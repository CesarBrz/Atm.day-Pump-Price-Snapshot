# --------------------------------------------------------------------------------------
# PumpFun / PumpSwap – Multi-mint price monitor with automatic source switching
# --------------------------------------------------------------------------------------
# GUI inspired by `zzz - teste busca preco pumpswap.py`.
# - User enters one mint per line (max 10).
# - For each mint the program first uses PumpFun (bonding curve) to fetch price and
#   detects whether the token is already `complete`.
# - While `complete` is False, price is queried from PumpFun.
# - Once `complete` flips to True, the script automatically switches to PumpSwap for
#   that mint and keeps using Pool reserves to calculate price.
#
# Requirements (install with pip if missing):
#   solana==0.30.0  solders==0.18.3  tkinter (built-in)  typing-extensions
# --------------------------------------------------------------------------------------

from __future__ import annotations

import base64
import struct
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Tuple, Optional
import csv
import os
from datetime import datetime
import requests  # for fetching SOL price

# --- new imports for integration with KoTH monitor ---
import queue  # thread-safe queue for cross-thread communication
import threading
import asyncio

from solana.rpc.api import Client
from solana.rpc.types import MemcmpOpts, DataSliceOpts
from solders.pubkey import Pubkey

# Matplotlib for real-time charts (pip install matplotlib)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# After other imports, import the calculator and BuyHandler
from token_metrics import TokenMetricsCalculator
from buy_handler import BuyHandler

# --- env helper ---
# Garante que o .env na MESMA pasta do script seja carregado, mesmo quando o
# programa é executado a partir de outro diretório.
try:
    from pathlib import Path
    from dotenv import load_dotenv  # type: ignore

    env_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=env_path, override=True, encoding="utf-8-sig")
except ImportError:
    # python-dotenv não instalado; variáveis de ambiente devem estar definidas externamente
    pass

# --------------------------------------------------------------------------------------
# Constants & RPC endpoints
# --------------------------------------------------------------------------------------

# Program IDs
PUMPFUN_PROGRAM_ID = Pubkey.from_string("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
PUMPSWAP_PROGRAM_ID = Pubkey.from_string("pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA")

# Main RPC (used for regular queries)
RPC_URL = "https://api.mainnet-beta.solana.com"

# Helius API key now loaded from environment (.env)
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")

# Build discovery RPC list dynamically (Helius first if key provided)
DISCOVERY_RPC_CANDIDATES: List[str] = []
if HELIUS_API_KEY:
    DISCOVERY_RPC_CANDIDATES.append(f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}")
DISCOVERY_RPC_CANDIDATES.extend([
    "https://solana-api.projectserum.com",
    "https://ssc-dao.genesysgo.net",
])

LAMPORTS_PER_SOL = 10**9
USD_SOL = 0.0  # will be fetched dynamically at runtime

# Global polling interval (milliseconds)
POLL_PERIOD_MS = 500  # ajuste aqui para alterar a periodicidade das consultas

# Chart refresh interval (milliseconds)
CHART_REFRESH_MS = 500  # intervalo de atualização do gráfico

# Máximo de tokens acompanhados simultaneamente
MAX_MINTS = 30  # ajuste aqui conforme necessidade

# Snapshot interval for full table dump (milliseconds)
SNAPSHOT_INTERVAL_MS = 1000  # 1 segundo

# Directory where snapshot CSVs will be stored
SNAPSHOT_DIR = "snapshots"

# SOL price refresh interval (milliseconds - 5 to 10 requests per minute is recommended)
SOL_PRICE_REFRESH_MS = 20000

# ------------------------- Auto-buy default filters ------------------------
DEFAULT_INIT_POOL = "Any"  # values: "Any", "Pump", "Swap"
DEFAULT_MCAP_MIN = 200
DEFAULT_MCAP_MAX = 9999
DEFAULT_SCORE_MIN = 42
DEFAULT_SCORE_MAX = 9999
DEFAULT_WR_MIN = 50
DEFAULT_WR_MAX = 100
DEFAULT_TRADES_MIN = 100
DEFAULT_TRADES_MAX = 9999

# --------------------------------------------------------------------------------------
# PumpFun helpers
# --------------------------------------------------------------------------------------

def decode_curve_buf(buf: bytes) -> Tuple[int, int, bool]:
    """Decode raw bonding-curve account data and return (vtok, vsol, complete)."""
    vtok, vsol, *_ = struct.unpack_from("<QQQQQ", buf, 8)
    complete = struct.unpack_from("<?", buf, 48)[0]
    return vtok, vsol, complete

def get_curve_state(client: Client, curve_pubkey: Pubkey) -> Tuple[int, int, bool]:
    """Return (virtual_token_reserves, virtual_sol_reserves, complete_flag)."""
    resp = client.get_account_info(curve_pubkey, encoding="base64")

    if hasattr(resp, "value"):
        account_info = resp.value
    else:
        account_info = resp.get("result", {}).get("value")

    if not account_info:
        raise RuntimeError("Curve account not found")

    data_field = account_info.data if hasattr(account_info, "data") else account_info.get("data")

    if isinstance(data_field, (bytes, bytearray, memoryview)):
        buf = bytes(data_field)
    else:
        b64_data = data_field[0] if isinstance(data_field, (list, tuple)) else data_field
        buf = base64.b64decode(b64_data)

    vtok, vsol, *_ = struct.unpack_from("<QQQQQ", buf, 8)
    complete = struct.unpack_from("<?", buf, 48)[0]
    return vtok, vsol, complete


def pumpfun_price(vtok: int, vsol: int, decimals: int) -> Optional[float]:
    if vtok == 0:
        return None
    return (vsol / LAMPORTS_PER_SOL) / (vtok / (10 ** decimals))

# --------------------------------------------------------------------------------------
# PumpSwap helpers
# --------------------------------------------------------------------------------------

def find_pool_account(discovery_client: Client, mint_pubkey: Pubkey) -> Tuple[Pubkey, bytes]:
    """Return (pool_pda, raw_account_data)."""
    BASE_MINT_OFFSET = 43  # struct layout offset
    filters = [MemcmpOpts(offset=BASE_MINT_OFFSET, bytes=str(mint_pubkey))]

    resp = discovery_client.get_program_accounts(
        PUMPSWAP_PROGRAM_ID,
        filters=filters,
        data_slice=DataSliceOpts(offset=0, length=0),
    )
    keyed_accounts = resp.value
    if not keyed_accounts:
        raise RuntimeError("Pool not found for this mint (no PumpSwap liquidity)")

    pool_pubkey = keyed_accounts[0].pubkey
    acc_full = discovery_client.get_account_info(pool_pubkey, encoding="base64")
    data_field = acc_full.value.data

    if isinstance(data_field, (list, tuple)):
        buf = base64.b64decode(data_field[0])
    elif isinstance(data_field, str):
        buf = base64.b64decode(data_field)
    else:
        buf = bytes(data_field)

    return pool_pubkey, buf


def decode_pool_accounts(buf: bytes) -> Tuple[Pubkey, Pubkey]:
    POOL_BASE_TOKEN_ACC_OFFSET = 139
    POOL_QUOTE_TOKEN_ACC_OFFSET = 171
    base_acc = Pubkey.from_bytes(buf[POOL_BASE_TOKEN_ACC_OFFSET : POOL_BASE_TOKEN_ACC_OFFSET + 32])
    quote_acc = Pubkey.from_bytes(buf[POOL_QUOTE_TOKEN_ACC_OFFSET : POOL_QUOTE_TOKEN_ACC_OFFSET + 32])
    return base_acc, quote_acc


def pumpswap_price(base_raw: int, quote_raw: int, decimals: int) -> Optional[float]:
    if base_raw == 0:
        return None
    return (quote_raw / LAMPORTS_PER_SOL) / (base_raw / (10 ** decimals))

# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------

def pick_discovery_client() -> Client:
    for url in DISCOVERY_RPC_CANDIDATES:
        try:
            cli = Client(url, timeout=10)
            # Removido para economizar 1 requisição; a própria primeira chamada real confirmará disponibilidade
            # cli.get_version()
            return cli
        except Exception:
            continue
    raise RuntimeError("No public RPC available for getProgramAccounts right now.")

# --------------------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------------------

class MintTracker:
    """Holds state for a single mint across PumpFun/PumpSwap."""

    def __init__(
        self,
        mint: Pubkey,
        decimals: int,
        curve_pda: Pubkey,
        completed: bool,
        base_acc: Optional[Pubkey] = None,
        quote_acc: Optional[Pubkey] = None,
    ) -> None:
        self.mint = mint
        self.decimals = decimals
        self.curve_pda = curve_pda
        self.completed = completed  # True = use PumpSwap
        self.base_acc = base_acc
        self.quote_acc = quote_acc
        self.last_price: Optional[float] = None
        self.history: List[Tuple[float, float]] = []  # (elapsed_seconds, mcap_sol)
        self.initial_mcap_sol: Optional[float] = None  # definido na primeira leitura válida
        self.initial_source: Optional[str] = None  # Adiciona coluna "Fonte Descoberta" (estado inicial)
        self.ath_mcap_sol: Optional[float] = None  # maior Mcap (ATH) em SOL capturado
        # Armazena dados extras (KoTH) para uso em métricas, se disponíveis
        self.extra_data: dict = {}
        self.buy_evaluated = False
        self.buy_pass: bool = False  # result of filter evaluation

    # Convenience properties
    @property
    def use_pumpswap(self) -> bool:
        return self.completed and self.base_acc is not None and self.quote_acc is not None

    # ------------------------------------------------------------------
    # Métricas derivadas (TokenMetricsCalculator)
    # ------------------------------------------------------------------
    def get_metrics(self, start_timestamp: float) -> dict:
        """Calcula e retorna métricas atuais usando TokenMetricsCalculator."""
        from datetime import datetime  # import local to avoid circular issues

        rows = []
        for elapsed, mcap_sol in self.history:
            dt = datetime.fromtimestamp(start_timestamp + elapsed)
            rows.append((dt, {"mcap_sol": mcap_sol, "status_init": self.initial_source or "-"}))

        # ensure wallets field is JSON string
        extra = dict(self.extra_data) if self.extra_data else {}
        if "wallets" in extra and not isinstance(extra["wallets"], str):
            try:
                import json
                extra["wallets"] = json.dumps(extra["wallets"], ensure_ascii=False)
            except Exception:
                pass

        calc = TokenMetricsCalculator(rows, extra)
        return calc.get_metrics()

# --------------------------------------------------------------------------------------
# GUI application
# --------------------------------------------------------------------------------------

class PriceMonitorApp(ttk.Frame):
    POLL_INTERVAL_MS = POLL_PERIOD_MS  # usa variável global

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        self.client = Client(RPC_URL)
        self.discovery_client = pick_discovery_client()

        # Dynamic SOL price (USD)
        global USD_SOL
        try:
            USD_SOL = self.fetch_sol_price()
            self.last_sol_fetch_ts = datetime.now()
        except Exception as exc:
            print("Falha ao obter preço inicial do SOL:", exc)
        # Schedule periodic updates according to global interval
        self.after(SOL_PRICE_REFRESH_MS, self._update_sol_price)

        self._build_ui()

        self.trackers: List[MintTracker] = []

        # start timestamp for elapsed time axis (application launch)
        self.start_time: Optional[float] = time.time()

        # --- integration state ---
        self._new_mints_queue: "queue.Queue[str]" = queue.Queue()
        self._polling_active: bool = False  # ensures _poll is scheduled only once
        # periodic check for new mints coming from KoTH monitor
        self.after(100, self._process_queue)

        # snapshot logging
        self._snapshot_file: Optional[str] = None
        self._snapshot_task_started: bool = False

        # Footer status bar
        self.status_var = tk.StringVar(value="")
        status_lbl = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_lbl.pack(side="bottom", fill="x", padx=5, pady=2)

        # Stats helpers
        self.total_detected: int = 0  # cumulative tokens ever detected since start

        # Store KoTH extra data per mint
        self.koth_extra: dict[str, dict] = {}

        # periodic footer refresh
        self.after(1000, self._refresh_status)

        # Buy handler instance (encapsulates BUY logic)
        self.buy_handler = BuyHandler(self)

    # ---------------------------------------- UI
    def _build_ui(self):
        # Placeholder frame now replaced by filter controls
        self.filter_frame = ttk.LabelFrame(self, text="Auto-buy filters")
        self.filter_frame.pack(side="top", fill="x", padx=10, pady=10)

        # --- Init Pool combobox ---
        ttk.Label(self.filter_frame, text="Init Pool:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.pool_var = tk.StringVar(value=DEFAULT_INIT_POOL)
        pool_cb = ttk.Combobox(self.filter_frame, textvariable=self.pool_var, state="readonly",
                               values=["Any", "Pump", "Swap"], width=6)
        pool_cb.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        # Helper to create min/max entries
        def add_range(row, label):
            ttk.Label(self.filter_frame, text=label+":").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            vcmd = (self.register(self._validate_float), '%P')
            e_min = ttk.Entry(self.filter_frame, validate="key", validatecommand=vcmd, width=10)
            e_max = ttk.Entry(self.filter_frame, validate="key", validatecommand=vcmd, width=10)
            e_min.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            ttk.Label(self.filter_frame, text="to").grid(row=row, column=2, padx=2)
            e_max.grid(row=row, column=3, sticky="w", padx=5, pady=2)
            return e_min, e_max

        self.mcap_min, self.mcap_max = add_range(1, "Init Mcap")
        self.score_min, self.score_max = add_range(2, "Wallets Score")
        self.wr_min, self.wr_max = add_range(3, "Top WR Wallet")
        self.trades_min, self.trades_max = add_range(4, "Top WR Trades")

        # Populate defaults if not None
        def set_default(entry, value):
            if value is not None:
                entry.insert(0, str(value))

        set_default(self.mcap_min, DEFAULT_MCAP_MIN)
        set_default(self.mcap_max, DEFAULT_MCAP_MAX)
        set_default(self.score_min, DEFAULT_SCORE_MIN)
        set_default(self.score_max, DEFAULT_SCORE_MAX)
        set_default(self.wr_min, DEFAULT_WR_MIN)
        set_default(self.wr_max, DEFAULT_WR_MAX)
        set_default(self.trades_min, DEFAULT_TRADES_MIN)
        set_default(self.trades_max, DEFAULT_TRADES_MAX)

        update_btn = ttk.Button(self.filter_frame, text="Update Filters", command=self._update_filters)
        update_btn.grid(row=5, column=0, columnspan=4, pady=4)

        # Input widgets previously hidden are no longer needed; keep stubs
        self.txt_mints = tk.Text(self)  # invisible stub
        self._start_btn = ttk.Button(self)

        self._headers = [
            ("mint", "Mint", 300, False),
            ("status_init", "Init. Pool", 120, False),
            ("status", "Current Pool", 100, False),
            ("price_sol", "Price (SOL)", 120, True),
            ("price_usd", "Price (USD)", 120, True),
            ("ath_sol", "ATH Mcap (SOL)", 140, True),
            ("mcap_sol", "Curr. Mcap (SOL)", 140, True),
            ("mcap_init", "Init. Mcap (SOL)", 140, True),
            ("var_pct", "% Var", 80, True),
            ("ath_pct", "ATH %", 90, True),
            ("indice", "Wallets Score", 100, True),
            ("top_wr", "Top WR Wallet", 90, True),
            ("top_trades", "Top WR Trades", 90, True),
            ("buy", "Buy?", 60, False),
        ]

        columns = tuple(h[0] for h in self._headers)
        self.tree = ttk.Treeview(self, columns=columns, show="headings")

        for col, header, width, is_num in self._headers:
            self.tree.heading(col, text=header, command=lambda c=col, n=is_num: self._sort_by_column(c, n))
            anchor = "e" if is_num else "w"
            self.tree.column(col, width=width, anchor=anchor)

        # Track sort order per column
        self._sort_desc: dict[str, bool] = {}

        self.tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Bind double-click to open chart
        self.tree.bind("<Double-1>", self._on_row_dbl_click)

        # Initial filter configuration dict using defaults
        self.filter_config: dict[str, object] = {
            "pool": DEFAULT_INIT_POOL,
            "mcap_min": DEFAULT_MCAP_MIN,
            "mcap_max": DEFAULT_MCAP_MAX,
            "score_min": DEFAULT_SCORE_MIN,
            "score_max": DEFAULT_SCORE_MAX,
            "wr_min": DEFAULT_WR_MIN,
            "wr_max": DEFAULT_WR_MAX,
            "trades_min": DEFAULT_TRADES_MIN,
            "trades_max": DEFAULT_TRADES_MAX,
        }

    # ---------------------------------------- Events
    def _on_start(self):
        # Allows calling this method multiple times: if the monitor is already running,
        # we simply try to add only the new mints that are not yet being tracked.
        raw_lines = [ln.strip() for ln in self.txt_mints.get("1.0", tk.END).splitlines() if ln.strip()]
        if not raw_lines:
            messagebox.showerror("Empty input", "Please enter at least one mint.")
            return
        if len(raw_lines) > MAX_MINTS:
            messagebox.showerror("Limit exceeded", f"Maximum of {MAX_MINTS} mints allowed at once.")
            return

        failed: List[str] = []
        for mint_str in raw_lines:
            try:
                # Skip if already tracking
                if any(str(t.mint) == mint_str for t in self.trackers):
                    continue

                mint_pk = Pubkey.from_string(mint_str)
            except Exception as e:
                failed.append(f"{mint_str}: invalid mint – {e}")
                continue

            # Fetch decimals
            try:
                mint_info = self.client.get_account_info_json_parsed(mint_pk)
                decimals = int(mint_info.value.data.parsed["info"]["decimals"])
            except Exception:
                failed.append(f"{mint_str}: unable to get decimals or mint does not exist")
                continue

            # Derive curve PDA and check state
            curve_pda, _ = Pubkey.find_program_address([b"bonding-curve", bytes(mint_pk)], PUMPFUN_PROGRAM_ID)
            try:
                _vtok, _vsol, complete_flag = get_curve_state(self.client, curve_pda)
            except Exception as exc:
                failed.append(f"{mint_str}: error reading bonding curve – {exc}")
                continue

            base_acc = quote_acc = None
            if complete_flag:
                try:
                    _pool_pda, pool_buf = find_pool_account(self.discovery_client, mint_pk)
                    base_acc, quote_acc = decode_pool_accounts(pool_buf)
                except Exception as exc:
                    failed.append(f"{mint_str}: token complete but no pool found – {exc}")
                    continue

            tracker = MintTracker(
                mint=mint_pk,
                decimals=decimals,
                curve_pda=curve_pda,
                completed=complete_flag,
                base_acc=base_acc,
                quote_acc=quote_acc,
            )
            tracker.extra_data = {}
            # Evict oldest if at capacity
            self._evict_oldest_if_needed()

            self.trackers.append(tracker)
            status_current = "Swap" if tracker.use_pumpswap else "Pump"
            tracker.initial_source = status_current  # type: ignore[attr-defined]
            self.total_detected += 1

            self.tree.insert("", "end", iid=str(mint_pk), values=("",)*len(self._headers))
            self._render_row(tracker)

        if failed:
            messagebox.showwarning("Some mints ignored", "\n".join(failed))

        # Clear the text box after processing inputs
        self.txt_mints.delete("1.0", tk.END)

        if not self.trackers:
            messagebox.showerror("No valid mints", "Unable to process any of the mints provided.")
            return

        self.after(self.POLL_INTERVAL_MS, self._poll)
        self._polling_active = True

        # Ensure snapshot logging is running even if 'Iniciar' nunca foi clicado
        if not self._snapshot_task_started:
            self._init_snapshot_file()
            self.after(SNAPSHOT_INTERVAL_MS, self._snapshot_save)
            self._snapshot_task_started = True

    # ---------------------------------------- Polling
    def _poll(self):
        if not self.trackers:
            return  # not running

        # --- PumpSwap batch ---
        swap_trackers = [t for t in self.trackers if t.use_pumpswap]
        swap_accounts: List[Pubkey] = []
        for t in swap_trackers:
            swap_accounts.extend([t.base_acc, t.quote_acc])  # type: ignore[arg-type]

        amounts: List[int] = []
        if swap_accounts:
            try:
                resp = self.client.get_multiple_accounts(swap_accounts, encoding="base64")
                values = resp.value
                for acc_info in values:
                    if acc_info is None:
                        amounts.append(0)
                        continue
                    df = acc_info.data
                    if isinstance(df, (list, tuple)):
                        buf = base64.b64decode(df[0])
                    elif isinstance(df, str):
                        buf = base64.b64decode(df)
                    else:
                        buf = bytes(df)
                    amt = int.from_bytes(buf[64:72], "little")
                    amounts.append(amt)
            except Exception as exc:
                print("Error reading PumpSwap accounts:", exc)
                amounts = [0] * len(swap_accounts)

        # Map back prices for PumpSwap tokens
        for idx, t in enumerate(swap_trackers):
            base_raw = amounts[idx * 2]
            quote_raw = amounts[idx * 2 + 1]
            price_sol = pumpswap_price(base_raw, quote_raw, t.decimals)
            t.last_price = price_sol

            # store history (mcap)
            if price_sol is not None and self.start_time is not None:
                mcap_sol_val = price_sol * 1_000_000_000
                t.history.append((time.time() - self.start_time, mcap_sol_val))

            # Update ATH if necessary
            if price_sol is not None:
                mcap_val = price_sol * 1_000_000_000
                if t.ath_mcap_sol is None or mcap_val > t.ath_mcap_sol:
                    t.ath_mcap_sol = mcap_val

        # --- PumpFun batch read & completion check ---
        pump_trackers = [t for t in self.trackers if not t.use_pumpswap]
        if pump_trackers:
            pdas = [t.curve_pda for t in pump_trackers]
            try:
                resp = self.client.get_multiple_accounts(pdas, encoding="base64")
                values = resp.value
            except Exception as exc:
                print("Error reading bonding curves (batch):", exc)
                values = [None] * len(pdas)

            for t, acc_info in zip(pump_trackers, values):
                if acc_info is None:
                    t.last_price = None
                    continue

                df = acc_info.data
                if isinstance(df, (list, tuple)):
                    buf = base64.b64decode(df[0])
                elif isinstance(df, str):
                    buf = base64.b64decode(df)
                else:
                    buf = bytes(df)

                vtok, vsol, complete_flag = decode_curve_buf(buf)
                # Check completion switch
                if not t.completed and complete_flag:
                    try:
                        _pool, pool_buf = find_pool_account(self.discovery_client, t.mint)
                        t.base_acc, t.quote_acc = decode_pool_accounts(pool_buf)
                        t.completed = True
                        self.tree.set(str(t.mint), column="status", value="Swap")
                        # Will be included in next PumpSwap batch
                    except Exception as exc:
                        print(f"Token {t.mint} completed but no pool found: {exc}")

                t.last_price = pumpfun_price(vtok, vsol, t.decimals)

                if t.last_price is not None and self.start_time is not None:
                    mcap_sol_val = t.last_price * 1_000_000_000
                    t.history.append((time.time() - self.start_time, mcap_sol_val))

                # Update ATH if necessary
                if t.last_price is not None:
                    mcap_val = t.last_price * 1_000_000_000
                    if t.ath_mcap_sol is None or mcap_val > t.ath_mcap_sol:
                        t.ath_mcap_sol = mcap_val

        # --- Update Treeview lines ---
        for t in self.trackers:
            self._render_row(t)

        self.after(self.POLL_INTERVAL_MS, self._poll)
        self._polling_active = True

    # ---------------------------------------- Chart window

    def _on_row_dbl_click(self, event):
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        tracker = next((t for t in self.trackers if str(t.mint) == item_id), None)
        if tracker is None:
            return

        PriceChartWindow(self, tracker)

    # ---------------------------------------- Table sorting

    def _sort_by_column(self, col: str, is_numeric: bool):
        # Determine sort direction
        desc = self._sort_desc.get(col, False)
        self._sort_desc[col] = not desc  # toggle for next click

        # Build list of (value, iid)
        def parse(val: str):
            if not is_numeric:
                return val.lower()
            if val in ("–", ""):
                return float("nan")
            # remove thousand separators etc.
            val = val.replace("%", "").replace(",", "").replace("+", "")
            try:
                return float(val)
            except ValueError:
                return float("nan")

        pairs = []
        for iid in self.tree.get_children(""):
            val = self.tree.set(iid, col)
            pairs.append((parse(val), iid))

        pairs.sort(reverse=not desc, key=lambda x: (x[0] != x[0], x[0]))  # NaNs last

        # Reorder
        for index, (_, iid) in enumerate(pairs):
            self.tree.move(iid, "", index)

    # ---------------------------------------- Integration helpers
    def queue_new_mint(self, data):
        """Can receive just a mint string or the full KoTH dict."""
        if isinstance(data, dict):
            mint_str = data.get("mint", "").strip()
            extra = data
        else:
            mint_str = str(data).strip()
            extra = None
        if not mint_str:
            return
        self._new_mints_queue.put((mint_str, extra))

    def _process_queue(self):
        while not self._new_mints_queue.empty():
            mint_str, extra = self._new_mints_queue.get()
            if extra:
                # save extra for later use
                self.koth_extra[mint_str] = extra
                # if already tracking, update tracker
                for t in self.trackers:
                    if str(t.mint) == mint_str:
                        t.extra_data = extra
            if mint_str:
                self._try_add_mint(mint_str, self.koth_extra.get(mint_str))
        self.after(100, self._process_queue)  # keep polling the queue

    def _try_add_mint(self, mint_str: str, extra: dict | None = None):
        """Add a single mint to the tracker list if valid and not already present."""
        # Skip if already tracked
        if any(str(t.mint) == mint_str for t in self.trackers):
            return
        # Respect global capacity limit
        if len(self.trackers) >= MAX_MINTS:
            # Automatically evict oldest to make room
            self._evict_oldest_if_needed()

        try:
            mint_pk = Pubkey.from_string(mint_str)
        except Exception:
            print(f"[ATMDAY] Invalid mint ignored: {mint_str}")
            return

        # Fetch decimals
        try:
            mint_info = self.client.get_account_info_json_parsed(mint_pk)
            decimals = int(mint_info.value.data.parsed["info"]["decimals"])
        except Exception:
            print(f"[ATMDAY] Unable to get decimals or mint does not exist: {mint_str}")
            return

        # Derive curve PDA and check state
        curve_pda, _ = Pubkey.find_program_address([b"bonding-curve", bytes(mint_pk)], PUMPFUN_PROGRAM_ID)
        try:
            _vtok, _vsol, complete_flag = get_curve_state(self.client, curve_pda)
        except Exception as exc:
            print(f"[ATMDAY] Error reading bonding curve for {mint_str}: {exc}")
            return

        base_acc = quote_acc = None
        if complete_flag:
            try:
                _pool_pda, pool_buf = find_pool_account(self.discovery_client, mint_pk)
                base_acc, quote_acc = decode_pool_accounts(pool_buf)
            except Exception as exc:
                print(f"[ATMDAY] Token complete but no pool found ({mint_str}): {exc}")
                return

        tracker = MintTracker(
            mint=mint_pk,
            decimals=decimals,
            curve_pda=curve_pda,
            completed=complete_flag,
            base_acc=base_acc,
            quote_acc=quote_acc,
        )
        tracker.extra_data = extra or {}
        # Evict oldest if at capacity
        self._evict_oldest_if_needed()

        self.trackers.append(tracker)
        status_current = "Swap" if tracker.use_pumpswap else "Pump"
        tracker.initial_source = status_current  # type: ignore[attr-defined]
        self.total_detected += 1

        self.tree.insert("", "end", iid=str(mint_pk), values=("",)*len(self._headers))
        self._render_row(tracker)

        # ensure polling is active
        if not self._polling_active:
            self.after(self.POLL_INTERVAL_MS, self._poll)
            self._polling_active = True
            if self.start_time is None:
                self.start_time = time.time()

        # Ensure snapshot logging is running even if 'Iniciar' never clicked
        if not self._snapshot_task_started:
            self._init_snapshot_file()
            self.after(SNAPSHOT_INTERVAL_MS, self._snapshot_save)
            self._snapshot_task_started = True

    # ---------------------------------------- Snapshot logging
    def _init_snapshot_file(self):
        """Create snapshot CSV file with headers."""
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Ensure directory exists
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        filename = os.path.join(SNAPSHOT_DIR, f"snapshots_{dt_str}.csv")
        headers = [h[0] for h in self._headers]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + headers)
        self._snapshot_file = filename

    def _snapshot_save(self):
        """Append current table snapshot to CSV and reschedule."""
        if self._snapshot_file is None:
            # should not happen, but safeguard
            self.after(SNAPSHOT_INTERVAL_MS, self._snapshot_save)
            return

        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self._snapshot_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for iid in self.tree.get_children(""):
                    row_vals = list(self.tree.item(iid)["values"])
                    writer.writerow([ts_str] + row_vals)
        except Exception as exc:
            print(f"Error saving snapshot CSV: {exc}")

        self.after(SNAPSHOT_INTERVAL_MS, self._snapshot_save)

    # ---------------------------------------- Capacity management
    def _evict_oldest_if_needed(self):
        """Ensure the number of tracked mints does not exceed MAX_MINTS by
        removing the oldest entries (FIFO order)."""
        while len(self.trackers) >= MAX_MINTS:
            oldest = self.trackers.pop(0)
            try:
                self.tree.delete(str(oldest.mint))
            except tk.TclError:
                # The row might already be gone; ignore
                pass

    # ---------------------------------------- SOL price updater
    def _update_sol_price(self):
        """Fetch latest SOL price in USD and reschedule itself."""
        global USD_SOL
        try:
            USD_SOL = self.fetch_sol_price()
            self.last_sol_fetch_ts = datetime.now()
            # Optionally, refresh price_usd column immediately by triggering a lightweight redraw
            # We avoid heavy RPC calls; just recompute USD strings from existing SOL prices
            for t in self.trackers:
                price_sol = t.last_price
                if price_sol is None:
                    continue
                price_usd_str = f"{price_sol * USD_SOL:.10f}"
                iid = str(t.mint)
                self.tree.set(iid, column="price_usd", value=price_usd_str)
        except Exception as exc:
            print("Error updating SOL price:", exc)
        # reschedule
        self.after(SOL_PRICE_REFRESH_MS, self._update_sol_price)

    def fetch_sol_price(self) -> float:
        """Return current SOL price in USD using CoinGecko simple price endpoint."""
        url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("solana", {}).get("usd", 0.0))

    # ---------------------------------------- Footer status refresh
    def _refresh_status(self):
        sol_str = f"${USD_SOL:.2f}" if USD_SOL else "-"
        ts_str = self.last_sol_fetch_ts.strftime("%H:%M:%S") if self.last_sol_fetch_ts else "-"
        run_str = "-"
        if self.start_time:
            elapsed = int(time.time() - self.start_time)
            run_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        tracked = len(self.trackers)
        total = self.total_detected
        self.status_var.set(
            f"Running: {run_str} | Tracking {tracked}/{total} tokens | SOL price: {sol_str} (updated {ts_str})"
        )
        self.after(1000, self._refresh_status)

    # ---------------------------------------- Validation helper
    def _validate_float(self, text: str) -> bool:
        if text in ("", "-"):
            return True
        try:
            float(text)
            return True
        except ValueError:
            return False

    # ---------------------------------------- Update filters
    def _update_filters(self):
        def parse(entry):
            val = entry.get().strip()
            if val == "":
                return None
            try:
                return float(val)
            except ValueError:
                return None

        self.filter_config = {
            "pool": self.pool_var.get(),
            "mcap_min": parse(self.mcap_min),
            "mcap_max": parse(self.mcap_max),
            "score_min": parse(self.score_min),
            "score_max": parse(self.score_max),
            "wr_min": parse(self.wr_min),
            "wr_max": parse(self.wr_max),
            "trades_min": parse(self.trades_min),
            "trades_max": parse(self.trades_max),
        }
        messagebox.showinfo("Filters", "Filter values updated.")

    # ---------------------------------------- Filter evaluation
    def _apply_filters(self, metrics: dict):
        cfg = self.filter_config
        # Pool filter
        pool_sel = cfg.get("pool", "Any")
        if pool_sel != "Any" and metrics.get("source") != pool_sel:
            return False
        def check_range(key_min, key_max, metric_key):
            v = metrics.get(metric_key)
            if v is None:
                return None  # incomplete
            lo = cfg.get(key_min)
            hi = cfg.get(key_max)
            if lo is not None and v < lo:
                return False
            if hi is not None and v > hi:
                return False
            return True
        for kmin,kmax,metric in (
            ("mcap_min","mcap_max","mcap0"),
            ("score_min","score_max","indice"),
            ("wr_min","wr_max","top_wr"),
            ("trades_min","trades_max","top_trades"),
        ):
            res = check_range(kmin,kmax,metric)
            if res is None:
                return None  # wait for data
            if res is False:
                return False
        return True

    # ---------------------------------------- Tree row helper
    def _render_row(self, t: 'MintTracker'):
        price_sol = t.last_price
        price_sol_str = f"{price_sol:.9f}" if price_sol is not None else "–"
        price_usd_str = f"{price_sol * USD_SOL:.10f}" if price_sol is not None else "–"
        mcap_current = price_sol * 1_000_000_000 if price_sol is not None else None

        # set initial mcap if not yet set
        if mcap_current is not None and t.initial_mcap_sol is None:
            t.initial_mcap_sol = mcap_current

        mcap_sol_str = f"{mcap_current:.2f}" if mcap_current is not None else "–"
        mcap_init_str = f"{t.initial_mcap_sol:.2f}" if t.initial_mcap_sol is not None else "–"
        var_pct_str = "–"
        if mcap_current is not None and t.initial_mcap_sol:
            var_pct = (mcap_current / t.initial_mcap_sol - 1) * 100
            var_pct_str = f"{var_pct:+.2f}%"
        ath_sol_str = f"{t.ath_mcap_sol:.2f}" if t.ath_mcap_sol is not None else "–"

        metrics = t.get_metrics(self.start_time or time.time())
        ath_pct_val = metrics.get("ath_pct")
        indice_val = metrics.get("indice")
        top_wr_val = metrics.get("top_wr")
        top_trades_val = metrics.get("top_trades")
        ath_pct_str = f"{ath_pct_val:+.2f}%" if ath_pct_val is not None else "–"
        indice_str = f"{indice_val:.2f}" if indice_val is not None else "–"
        top_wr_str = f"{top_wr_val:.2f}" if top_wr_val is not None else "–"
        top_trades_str = str(top_trades_val) if top_trades_val is not None else "–"

        buy_str = "Yes" if t.buy_pass else "No"

        self.tree.item(
            str(t.mint),
            values=(
                str(t.mint),
                getattr(t, "initial_source", "–"),
                "Swap" if t.use_pumpswap else "Pump",
                price_sol_str,
                price_usd_str,
                ath_sol_str,
                mcap_sol_str,
                mcap_init_str,
                var_pct_str,
                ath_pct_str,
                indice_str,
                top_wr_str,
                top_trades_str,
                buy_str,
            ),
        )
        # warn if wallet metrics unavailable
        if top_wr_val is None:
            print(f"[WARN] Wallet metrics unavailable yet for {t.mint} – extra_data missing or malformed")

        if not getattr(t, 'buy_evaluated', False):
            res = self._apply_filters(metrics)
            if res is None:
                # missing data, keep for next iteration
                pass
            else:
                if res:
                    # Mark tracker and invoke BuyHandler (no internal row refresh to avoid recursion)
                    t.buy_pass = True
                    self.buy_handler.execute_buy(t, metrics)
                    # Update 'Buy' column directly
                    self.tree.set(str(t.mint), column="buy", value="Yes")
                t.buy_evaluated = True

class PriceChartWindow(tk.Toplevel):
    def __init__(self, master: tk.Widget, tracker: MintTracker):
        super().__init__(master)
        self.title(f"Gráfico – {tracker.mint}")
        self.tracker = tracker
        self.start_timestamp = getattr(master, "start_time", time.time())

        fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Tempo (s)")
        self.ax.set_ylabel("Mcap (SOL)")

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.line, = self.ax.plot([], [], color="blue")
        self.after(0, self._update_chart)

        # reference line (detection mcap)
        ref_y = tracker.initial_mcap_sol if tracker.initial_mcap_sol is not None else 0
        self.ref_line = self.ax.axhline(ref_y, color="red", linestyle="--")

        # -- métricas label --
        self.metrics_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.metrics_var).pack(pady=5)

        # Button to save CSV
        save_btn = ttk.Button(self, text="Salvar CSV", command=self._save_csv)
        save_btn.pack(pady=5)

    def _update_chart(self):
        if not self.tracker.history:
            self.after(CHART_REFRESH_MS, self._update_chart)
            return

        xs, ys = zip(*self.tracker.history)
        self.line.set_data(xs, ys)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

        # update reference line if initial mcap now defined
        if self.tracker.initial_mcap_sol is not None:
            y = self.tracker.initial_mcap_sol
            self.ref_line.set_ydata([y, y])

        # Atualiza métricas na label
        metrics = self.tracker.get_metrics(self.start_timestamp)
        ath_pct_val = metrics.get("ath_pct")
        indice_val = metrics.get("indice")
        top_wr_val = metrics.get("top_wr")
        top_trades_val = metrics.get("top_trades")

        ath_pct_str = f"{ath_pct_val:+.2f}%" if ath_pct_val is not None else "–"
        indice_str = f"{indice_val:.2f}" if indice_val is not None else "–"
        top_wr_str = f"{top_wr_val:.2f}" if top_wr_val is not None else "–"
        top_trades_str = str(top_trades_val) if top_trades_val is not None else "–"

        self.metrics_var.set(f"ATH%: {ath_pct_str}, Índice: {indice_str}, Top WR: {top_wr_str} ({top_trades_str})")

        self.after(CHART_REFRESH_MS, self._update_chart)

    # -------------------------------------------------- CSV Saving
    def _save_csv(self):
        """Save current tracker's history to a timestamped CSV file."""
        if not self.tracker.history:
            messagebox.showwarning("Sem dados", "Ainda não há dados para salvar.")
            return

        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.tracker.mint}_{dt_str}.csv"

        try:
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["elapsed_seconds", "mcap_sol"])
                writer.writerows(self.tracker.history)
            messagebox.showinfo("CSV salvo", f"Série temporal salva em {filename}")
        except Exception as exc:
            messagebox.showerror("Erro", f"Falha ao salvar CSV: {exc}")

# --------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------

def main():
    root = tk.Tk()
    root.title("PumpFun / PumpSwap – Multi Mint Price Monitor")
    app = PriceMonitorApp(root)

    # --------------------------------------------------------------
    # Inicia monitor KoTH (Telegram) em thread separada
    # --------------------------------------------------------------
    try:
        from koth_monitor_atm import KothMonitorATMDAY
    except ImportError:
        print("koth_monitor_atm.py não encontrado; monitor Telegram desabilitado.")
    else:
        def koth_callback(new_koth, old_mint):
            app.queue_new_mint(new_koth)

        def run_koth_monitor():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            monitor = KothMonitorATMDAY(interval_ms=0, callback=koth_callback, include_nsfw=True, show_raw=False)
            monitor.start()
            loop.run_forever()

        threading.Thread(target=run_koth_monitor, daemon=True).start()

    root.mainloop()


if __name__ == "__main__":
    main() 