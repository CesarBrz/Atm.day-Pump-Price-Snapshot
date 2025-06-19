# PumpFun / PumpSwap Multi-Mint Toolkit

## Overview
This repository provides a **real-time toolkit for monitoring, analysing and (in the future) trading newly launched Solana tokens** that start on the _PumpFun_ bonding curve and later migrate to the _PumpSwap_ liquidity pool.

Key components:

| File | Purpose |
|------|---------|
| `1-price-monitor.py` | GUI application that tracks up to 30 mints simultaneously, automatically switching from PumpFun to PumpSwap once liquidity is active, calculates live metrics and can fire *BUY* signals based on configurable filters. |
| `koth_monitor_atm.py` | Background Telegram listener that parses KoTH (King of the Hill) messages and feeds fresh mints — together with rich wallet statistics — directly into the price monitor. |
| `token_metrics.py` | Stand-alone utility that turns raw snapshot rows + KoTH data into advanced metrics such as initial market-cap, ATH %, weighted wallet score, etc. |
| `buy_handler.py` | Skeleton class used by the GUI to react to *BUY* signals (currently shows a message box only). A complementary **`SellHandler`** is planned for future releases. |
| `2-snapshot-viewer.py` | Desktop tool to open the CSV snapshots produced by the monitor, review historical performance of each token and visualise growth charts. |
| `get solana price.py` | Minimal example on how to fetch the current SOL ↔ USD price from CoinGecko (already integrated in the main GUI). |

---

## Features
* Track **PumpFun** bonding-curve and **PumpSwap** pool prices side-by-side.
* Automatic source switching the millisecond a token becomes _complete_.
* Configurable auto-buy filters: pool type, initial market-cap range, wallet score, top win-rate, trade count and more.
* Matplotlib charts embedded in Tkinter for instant visual feedback.
* 1-second CSV **snapshots** of the full table for later back-testing.
* Optional Telegram integration so you never miss a new ATM-DAY alert.
* Re-usable metrics class for notebooks, scripts or dashboards.

> **Planned** – Full trade execution pipelines will live in dedicated `BuyHandler` / `SellHandler` classes. At the moment they are placeholders: only GUI pop-ups are triggered, no orders are sent.

---

## Installation
1. Install Python ≥3.10.
2. Clone the repository and `cd` into it.
3. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt  # or manually:
   pip install solana==0.30.0 solders==0.18.3 telethon matplotlib python-dotenv requests typing-extensions
   ```
5. Copy `.env.example` to `.env` and fill in:
   ```dotenv
   HELIUS_API_KEY=your_helius_key_here
   TELEGRAM_API_ID=123456
   TELEGRAM_API_HASH=abcdef...
   TELEGRAM_USER_ID=987654321
   ```

---

## Usage
### 1. Real-Time Price Monitor
```bash
python 1-price-monitor.py
```
* Paste one mint per line or just wait for the KoTH monitor to inject new tokens automatically.
* Adjust the **Auto-buy Filters** pane at the top to fine-tune your strategy.
* Double-click any row to open an individual Mcap chart.

### 2. Snapshot Viewer
After running the monitor you will find CSV files inside `snapshots/`.

```bash
python 2-snapshot-viewer.py
```
Pick a CSV file and explore historical metrics and wallet data.

---

## Roadmap
* Wire the `BuyHandler` to an on-chain trading backend (Jupiter, Raydium, etc.).
* Implement a symmetric `SellHandler` with take-profit / stop-loss logic.
* Add headless / CLI mode for server deployments.
* Export metrics to Prometheus & Grafana dashboards.

---

## License
Distributed under the MIT License – see `LICENSE` for details. 
