"""Token metrics utilities.

This module provides the `TokenMetricsCalculator` class, responsible for calculating all
indices related to a token (initial Market Cap/ATH, percentage increase, wallet metrics, etc.).
It is independent of the graphical interface and can be reused in scripts, notebooks, or any other
Python application.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 1 SOL = 10 ** 9 lamports  (default on Solana network)
LAMPORTS_PER_SOL: int = 10 ** 9

__all__ = ["TokenMetricsCalculator", "LAMPORTS_PER_SOL"]


class TokenMetricsCalculator:
    """Calculates metrics of a token snapshot.

    Parameters
    ----------
    rows : List[Tuple[datetime, dict]]
        List of tuples (timestamp, dict) containing the snapshot lines of the
        token, already ordered or not. If not ordered, the class does not
        depend on the order for calculations, except for the ATH index.
    extra : dict | None, default None
        Line of the CSV of detections (KoTH) for the token, containing the
        `wallets` field (json) and other fields.
    lamports_per_sol : int, default LAMPORTS_PER_SOL
        Conversion factor from lamports to SOL.
    """

    def __init__(
        self,
        rows: List[Tuple[datetime, dict]],
        extra: Optional[dict] = None,
        lamports_per_sol: int = LAMPORTS_PER_SOL,
    ) -> None:
        self.rows = rows or []
        self.extra = extra or {}
        self.lamports_per_sol = lamports_per_sol
        self._metrics: Dict[str, object] = {}
        self._compute()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, object]:
        """Returns the dictionary with all calculated indices."""
        return self._metrics

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------
    def _compute(self) -> None:
        # --- Mcap series ------------------------------------------------
        mcaps: List[Optional[float]] = [self._extract_mcap(row) for _ts, row in self.rows]

        mcap0 = next((v for v in mcaps if v is not None), None)
        ath_val = None
        ath_index = None
        if mcap0 is not None:
            for idx, val in enumerate(mcaps):
                if val is not None and (ath_val is None or val > ath_val):
                    ath_val = val
                    ath_index = idx
            ath_pct = (ath_val / mcap0 - 1) * 100 if ath_val is not None else None
        else:
            ath_pct = None

        # --- Wallet metrics --------------------------------------------
        wallets_raw = self.extra.get("wallets", "[]")
        try:
            wallets: List[dict] = json.loads(wallets_raw)
        except Exception:
            wallets = []

        total_trades = 0
        weighted_sum = 0.0
        top_wr = None
        top_trades = None
        for w in wallets:
            try:
                wr_val = float(w.get("wallet_winrate", 0))
                trades_val = int(w.get("wallet_trade_number", 0))
            except Exception:
                continue

            total_trades += trades_val
            weighted_sum += wr_val * trades_val

            if top_wr is None or wr_val > top_wr:
                top_wr = wr_val
                top_trades = trades_val

        indice = weighted_sum / total_trades if total_trades else None

        # --- Metadata ---------------------------------------------------
        ts0_str = self.rows[0][0].strftime("%Y-%m-%d %H:%M:%S") if self.rows else "-"
        source_val = self.rows[0][1].get("status_init", "-") if self.rows else "-"

        self._metrics = {
            "mcap0": mcap0,
            "ath_sol": ath_val,
            "ath_pct": ath_pct,
            "ath_index": ath_index,
            "indice": indice,
            "top_wr": top_wr,
            "top_trades": top_trades,
            "ts0": ts0_str,
            "source": source_val,
        }

    # ------------------------------------------------------------------
    def _extract_mcap(self, row: dict) -> Optional[float]:
        """Extracts or calculates the market cap in SOL from a snapshot line."""
        mcap_s = row.get("mcap_sol") or row.get("mcap_current")
        if not mcap_s or mcap_s == "–":  # caractere de "dash" vindo de CSV
            price_s = row.get("price_sol") or row.get("price_sol_str")
            try:
                price_f = float(price_s)
                return price_f * self.lamports_per_sol
            except Exception:
                return None
        try:
            return float(mcap_s)
        except Exception:
            return None 