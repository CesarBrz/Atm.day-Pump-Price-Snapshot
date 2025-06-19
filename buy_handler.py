from __future__ import annotations

"""Buy handler module.

This module defines the `BuyHandler` class, responsible for executing the BUY
signal logic decoupled from the GUI application. It displays a message,
updates tracker state, and triggers a row re-render on the main app.
"""

from tkinter import messagebox
from typing import TYPE_CHECKING, Any

# Forward type hints without importing the module (invalid name as module)
if TYPE_CHECKING:  # pragma: no cover
    MintTracker = Any  # type: ignore
    PriceMonitorApp = Any  # type: ignore


class BuyHandler:
    """Encapsulates BUY signal actions.

    Parameters
    ----------
    app : PriceMonitorApp
        Reference to the main GUI so we can ask it to refresh the row after a
        successful buy.
    """

    def __init__(self, app: 'PriceMonitorApp') -> None:  # type: ignore[name-defined]
        self.app = app

    # ------------------------------------------------------------------
    def execute_buy(self, tracker: 'MintTracker', metrics: dict) -> None:  # type: ignore[name-defined]
        """Perform the buy action: show message, mark tracker, refresh row."""
        msg = (
            f"BUY signal – Mint: {tracker.mint}\n"
            f"Init Pool: {metrics.get('source')} | Mcap0: {metrics.get('mcap0')} SOL\n"
            f"Indice: {metrics.get('indice')} | Top WR: {metrics.get('top_wr')} ({metrics.get('top_trades')})"
        )
        # Log to console
        print(msg)

        # Show GUI alert (may fail when running headless)
        try:
            messagebox.showinfo("BUY", msg)
        except Exception:
            pass

        # Update tracker state (row will be refreshed by caller)
        tracker.buy_pass = True 