import csv
import json
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Tuple

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from token_metrics import TokenMetricsCalculator

SNAPSHOT_DIR = "snapshots"
ATM_DETECTIONS_FILE = "atm_detections_pump.csv"
LAMPORTS_PER_SOL = 10 ** 9


class SnapshotViewer(tk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.pack(fill="both", expand=True)

        self.snapshot_data: Dict[str, List[Tuple[datetime, dict]]] = {}
        self.koth_extra: Dict[str, dict] = {}
        self.token_metrics: Dict[str, dict] = {}

        self._build_ui()
        self._load_koth_extra()
        self._prompt_snapshot_file()

    # ----------------------------- UI ---------------------------------------
    def _build_ui(self):
        lbl = ttk.Label(self, text="Tokens tracked:")
        lbl.pack(anchor="w", padx=10, pady=(10, 0))

        cols = ("ts0", "mint", "source", "mcap0", "ath_pct", "indice", "top_wr", "top_trades", "wallets")
        self.tree = ttk.Treeview(self, columns=cols, show="headings")
        self.tree.heading("ts0", text="Timestamp", command=lambda: self._sort_by("ts0", False))
        self.tree.heading("mint", text="Mint", command=lambda: self._sort_by("mint", False))
        self.tree.heading("source", text="Init. Pool", command=lambda: self._sort_by("source", False))
        self.tree.heading("mcap0", text="Mcap0 (SOL)", command=lambda: self._sort_by("mcap0", True))
        self.tree.heading("ath_pct", text="ATH %", command=lambda: self._sort_by("ath_pct", True))
        self.tree.heading("indice", text="Wallets Score", command=lambda: self._sort_by("indice", True))
        self.tree.heading("top_wr", text="Top WR Wallet %", command=lambda: self._sort_by("top_wr", True))
        self.tree.heading("top_trades", text="Trades Top WR", command=lambda: self._sort_by("top_trades", True))
        self.tree.heading("wallets", text="# Wallets", command=lambda: self._sort_by("wallets", True))

        self.tree.column("ts0", width=145, anchor="w")
        self.tree.column("mint", width=300, anchor="w")
        self.tree.column("source", width=80, anchor="center")
        self.tree.column("mcap0", width=120, anchor="e")
        self.tree.column("ath_pct", width=100, anchor="e")
        self.tree.column("indice", width=130, anchor="e")
        self.tree.column("top_wr", width=120, anchor="e")
        self.tree.column("top_trades", width=120, anchor="e")
        self.tree.column("wallets", width=90, anchor="e")

        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        self.tree.bind("<Double-1>", self._on_double_click)

        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        vsb.pack(side="left", fill="y", pady=10)
        self.tree.configure(yscrollcommand=vsb.set)

        self._sort_desc = {}

    # ----------------------------- Data loading -----------------------------
    def _prompt_snapshot_file(self):
        initial_dir = SNAPSHOT_DIR if os.path.isdir(SNAPSHOT_DIR) else "."
        path = filedialog.askopenfilename(
            title="Selecione o arquivo de snapshots",
            initialdir=initial_dir,
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            messagebox.showinfo("Sem arquivo", "Nenhum arquivo selecionado. Saindo.")
            self.master.destroy()
            return
        try:
            self._load_snapshot_file(path)
        except Exception as exc:
            messagebox.showerror("Erro", f"Falha ao carregar CSV: {exc}")
            self.master.destroy()

    def _load_snapshot_file(self, path: str):
        self.snapshot_data.clear()
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mint = row.get("mint")
                if not mint:
                    continue
                ts_str = row.get("timestamp") or row.get("Timestamp")
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                self.snapshot_data.setdefault(mint, []).append((ts, row))
        # sort each mint list
        for rows in self.snapshot_data.values():
            rows.sort(key=lambda x: x[0])

        # compute metrics per mint
        for mint, rows in self.snapshot_data.items():
            calc = TokenMetricsCalculator(rows, self.koth_extra.get(mint))
            self.token_metrics[mint] = calc.get_metrics()

        # populate treeview
        self.tree.delete(*self.tree.get_children())
        for mint in sorted(self.snapshot_data.keys()):
            m = self.token_metrics.get(mint, {})
            extra = self.koth_extra.get(mint, {})
            wallets_raw = extra.get("wallets", "[]")
            try:
                wallets_count = len(json.loads(wallets_raw))
            except Exception:
                wallets_count = "-"
            self.tree.insert("", "end", iid=mint, values=(
                m.get('ts0','-'),
                mint,
                m.get('source','-'),
                f"{m.get('mcap0', '-'):.2f}" if m.get('mcap0') else "-",
                f"{m.get('ath_pct', '-'): .2f}%" if m.get('ath_pct') is not None else "-",
                f"{m.get('indice', '-'): .2f}" if m.get('indice') is not None else "-",
                f"{m.get('top_wr', '-'): .2f}" if m.get('top_wr') is not None else "-",
                m.get('top_trades', '-') if m.get('top_trades') is not None else "-",
                wallets_count,
            ))

    def _load_koth_extra(self):
        if not os.path.isfile(ATM_DETECTIONS_FILE):
            return
        try:
            with open(ATM_DETECTIONS_FILE, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mint = row.get("mint")
                    if mint:
                        self.koth_extra[mint] = row
        except Exception as exc:
            print(f"Falha ao ler {ATM_DETECTIONS_FILE}: {exc}")

    # ----------------------------- Events -----------------------------------
    def _on_double_click(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        mint = sel[0]  # iid is mint
        DataWindow(self.master, mint, self.snapshot_data[mint], self.koth_extra.get(mint))

    def _sort_by(self, col: str, is_numeric: bool):
        # Determine current sort order
        desc = self._sort_desc.get(col, False)
        self._sort_desc[col] = not desc

        def parse(val):
            if is_numeric:
                try:
                    return float(str(val).replace('%', '').replace(',', '').strip())
                except Exception:
                    return float('nan')
            return str(val).lower()

        items = []
        for iid in self.tree.get_children(''):
            val = self.tree.set(iid, col)
            items.append((parse(val), iid))

        items.sort(reverse=not desc, key=lambda x: (x[0] != x[0], x[0]))

        for idx, (_, iid) in enumerate(items):
            self.tree.move(iid, '', idx)


class DataWindow(tk.Toplevel):
    def __init__(self, master: tk.Widget, mint: str, rows: List[Tuple[datetime, dict]], extra: dict | None):
        super().__init__(master)
        self.title(f"Histórico – {mint}")
        self.mint = mint
        self.rows = rows
        self.extra = extra or {}

        self._build_ui()
        self._populate_graph()
        self._populate_info()

    def _build_ui(self):
        # split into top (graph) and bottom (info)
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="both", expand=True, padx=10, pady=10)
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.bottom_frame = bottom_frame

        fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel("Tempo (s)")
        self.ax.set_ylabel("Mcap (SOL)")
        self.line, = self.ax.plot([], [], color="blue")

        canvas = FigureCanvasTkAgg(fig, master=top_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas = canvas

    def _populate_graph(self):
        if not self.rows:
            return
        base_ts = self.rows[0][0]
        xs = [ (ts - base_ts).total_seconds() for ts, _ in self.rows ]
        ys = []
        for _ts, row in self.rows:
            mcap_s = row.get("mcap_sol") or row.get("mcap_current")
            if not mcap_s or mcap_s == "–":
                # try compute from price_sol
                price_s = row.get("price_sol") or row.get("price_sol_str")
                try:
                    price_f = float(price_s)
                    mcap_val = price_f * 1_000_000_000
                except Exception:
                    mcap_val = None
            else:
                try:
                    mcap_val = float(mcap_s)
                except Exception:
                    mcap_val = None
            ys.append(mcap_val or float("nan"))
        self.line.set_data(xs, ys)

        # reference line at initial Mcap (first valid y)
        initial_mcap = next((y for y in ys if y == y), None)  # skip NaN
        if initial_mcap is not None:
            self.ax.axhline(initial_mcap, color="red", linestyle="--")

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _populate_info(self):
        # token info
        info_text = ttk.Label(self.bottom_frame, text="Informações do KoTH", font=("Segoe UI", 10, "bold"))
        info_text.pack(anchor="w")

        if not self.extra:
            ttk.Label(self.bottom_frame, text="Sem dados no CSV de detecções.").pack(anchor="w")
            return

        # Token name
        ttk.Label(self.bottom_frame, text=f"Token: {self.extra.get('token_name', '-')}").pack(anchor="w")

        # --- Métricas calculadas via classe reutilizável ---
        metrics = TokenMetricsCalculator(self.rows, self.extra).get_metrics()

        initial_mcap = metrics.get("mcap0")
        ath_val = metrics.get("ath_sol")
        ath_index = metrics.get("ath_index")
        valoriz_pct = metrics.get("ath_pct")
        indice_carteiras = metrics.get("indice")

        ttk.Label(self.bottom_frame, text=f"Market Cap0 (SOL): {initial_mcap:.2f}" if initial_mcap else "Market Cap0 (SOL): -").pack(anchor="w")

        if ath_val is not None:
            ttk.Label(self.bottom_frame, text=f"Market Cap ATH (SOL): {ath_val:.2f} – Time idx {ath_index}").pack(anchor="w")
            if valoriz_pct is not None:
                ttk.Label(self.bottom_frame, text=f"Valorização ATH %: {valoriz_pct:+.2f}%").pack(anchor="w")
        else:
            ttk.Label(self.bottom_frame, text="Market Cap ATH: -").pack(anchor="w")

        ttk.Label(self.bottom_frame, text=f"Índice Carteiras: {indice_carteiras:.2f}" if indice_carteiras is not None else "Índice Carteiras: -").pack(anchor="w")

        # wallets table permanece inalterada
        wallets_raw = self.extra.get("wallets", "[]")
        try:
            wallets = json.loads(wallets_raw)
        except Exception:
            wallets = []

        if wallets:
            ttk.Label(self.bottom_frame, text="Carteiras:").pack(anchor="w", pady=(5, 0))
            tree = ttk.Treeview(self.bottom_frame, columns=("addr", "wr", "trades"), show="headings", height=5)
            tree.heading("addr", text="Endereço")
            tree.heading("wr", text="Winrate")
            tree.heading("trades", text="Trades")
            tree.column("addr", width=260)
            tree.column("wr", width=80, anchor="e")
            tree.column("trades", width=80, anchor="e")
            for w in wallets:
                tree.insert("", "end", values=(
                    w.get("wallet_address", "-"),
                    w.get("wallet_winrate", "-"),
                    w.get("wallet_trade_number", "-"),
                ))
            tree.pack(fill="x", expand=False, pady=(0, 5))


# ------------------------------ Main ----------------------------------------

def main():
    root = tk.Tk()
    root.title("Snapshot Viewer – PumpFun/PumpSwap")
    app = SnapshotViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main() 