import asyncio
import logging
import re
import os
import csv
import json
from datetime import datetime
# --- env helper ---
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass
from telethon import TelegramClient, events

# If True, only mints ending with "pump" will trigger the callback and
# are considered valid KoTH. Other mints are ignored for callback purposes,
# but will still be recorded in a separate csv.
filter_pump = True


TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID", "0"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")
ID_USER = int(os.getenv("TELEGRAM_USER_ID", "0"))

class KothMonitorATMDAY:
    """
    Class to monitor ATMDAY PUMP ALGO signals through Telegram messages.
    Now automatically registers each new token detected in 'atm_detections_pump.csv' and 'atm_detections_others.csv'.

    Callback receives:
        - new_koth (dict): Dictionary with all parsed token information.
        - old_koth_mint (str|None): Previous mint.
    """
    CSV_FILE_PUMP = 'atm_detections_pump.csv'
    CSV_FILE_OTHERS = 'atm_detections_others.csv'
    FIELDNAMES = [
        'mint', 'token_name', 'price', 'market_cap', 'launched_at',
        'wallet_in', 'wallet_out', 'volume_in', 'volume_out',
        'avg_buy', 'avg_sell', 'is_mintable', 'is_freezable', 'is_dex_paid',
        'wallets'
    ]

    def __init__(self, interval_ms: int, callback, include_nsfw: bool = True, show_raw: bool = False):
        self.callback = callback
        self.api_id = TELEGRAM_API_ID
        self.api_hash = TELEGRAM_API_HASH
        self.session_name = 'atmdaysocket'
        self.user_id = ID_USER
        self.show_raw = show_raw
        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
        self._running = False
        self._last_koth = {}

        # Initialize CSV files if they don't exist
        self._ensure_csv(self.CSV_FILE_PUMP)
        self._ensure_csv(self.CSV_FILE_OTHERS)

        if not (self.api_id and self.api_hash and self.user_id):
            raise RuntimeError("Missing Telegram credentials. Define TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_USER_ID in environment or .env")

        @self.client.on(events.NewMessage(from_users=self.user_id))
        async def _message_handler(event):
            raw_text = event.message.raw_text

            # Show the full message dump before any processing when enabled
            if self.show_raw:
                print("\n=== Raw message received ===")
                print(raw_text)
                print("=== End of raw message ===\n")

            logging.info(f"New message received from user {self.user_id}.")
            await self.processar_mensagem(raw_text)

    def _ensure_csv(self, path: str):
        """Create the CSV file with header if it doesn't exist."""
        if not os.path.exists(path):
            with open(path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def _write_to_csv(self, data: dict, csv_path: str):
        row = data.copy()
        # launched_at is already a string; no conversion needed

        # Format price values to avoid scientific notation
        for key in ('price', 'avg_buy', 'avg_sell'):
            val = row.get(key)
            if isinstance(val, float):
                # 10 decimal places, no scientific notation, remove zeros/trailing dot
                row[key] = (f"{val:.10f}").rstrip('0').rstrip('.')

        row['wallets'] = json.dumps(row.get('wallets', []), ensure_ascii=False)
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow({k: row.get(k, '') for k in self.FIELDNAMES})

    async def processar_mensagem(self, raw_text: str):
        data = self.parse_koth_message(raw_text)
        if not data:
            logging.warning("Unable to parse the message.")
            return
        new_mint = data.get('mint')
        old_mint = self._last_koth.get('mint')

        # Define if mint ends with "pump" (case-insensitive)
        is_pump = isinstance(new_mint, str) and new_mint.lower().endswith('pump')

        if new_mint != old_mint:
            logging.info(f"Change in token detected: {old_mint or 'None'} -> {new_mint}")

            # Save to corresponding CSV
            csv_target = self.CSV_FILE_PUMP if is_pump else self.CSV_FILE_OTHERS
            try:
                self._write_to_csv(data, csv_target)
            except Exception as e:
                logging.error(f"Failed to write to CSV: {e}")

            # If filter is active and token is not pump, only register in CSV and exit
            if filter_pump and not is_pump:
                logging.info(f"Token '{new_mint}' ignored by filter_pump.")
                return  # Do not trigger callback nor update _last_koth

            # Otherwise, process normally (trigger callback and update state)
            print(f"✨✨✨ New token detected: {new_mint} ✨✨✨")
            self.callback(data, old_mint)
            self._last_koth = data
        else:
            logging.debug(f"Same token received again: {new_mint}")

    def parse_koth_message(self, text: str) -> dict:
        # Remove empty lines and apply strip to facilitate parsing
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        header = {}
        wallets = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # If we reach the wallets section, stop collecting header
            if line.startswith('🧠') or line.lower().startswith('7d stats'):
                i += 1
                break
            # ----------- New patterns/adjustments (case-insensitive) ------------
            # Mint / CA
            if re.match(r'ca\s*:', line, re.I):
                header['mint'] = line.split(':', 1)[1].strip()
            # Token name
            elif re.match(r'token name\s*:', line, re.I):
                token_part = line.split(':', 1)[1].strip()
                # Remove dollar sign if present
                if token_part.startswith('$'):
                    token_part = token_part[1:].strip()
                header['token_name'] = token_part
            # Price
            elif re.match(r'price\s*:', line, re.I):
                try:
                    header['price'] = float(line.split(':', 1)[1].strip().lstrip('$'))
                except ValueError:
                    pass
            # Market Cap / marketCap (aceita variações de espaço e maiúsc/minúsc)
            elif re.match(r'market\s*cap\s*:', line, re.I) or re.match(r'marketcap\s*:', line, re.I):
                cap_val = line.split(':', 1)[1].strip().lstrip('$')
                header['market_cap'] = self._parse_abbrev_number(cap_val)
            # Keep
            elif line.startswith('Launched at:'):
                header['launched_at'] = line.split(':', 1)[1].strip()
            elif re.match(r'\d+\/\d+ wallets in\/out', line):
                w_in, w_out = line.split()[0].split('/')
                header['wallet_in'] = int(w_in)
                header['wallet_out'] = int(w_out)
            elif re.match(r'[\d\.]+[kKmMbB]?/[\d\.]+ volume in/out', line):
                v_in, v_out = line.split()[0].split('/')
                header['volume_in'] = self._parse_abbrev_number(v_in)
                header['volume_out'] = self._parse_abbrev_number(v_out)
            elif re.match(r'[\d\.]+\/[\d\.]+ avg\. buy\/sell price', line):
                a_buy, a_sell = line.split()[0].split('/')
                header['avg_buy'] = float(a_buy)
                header['avg_sell'] = float(a_sell)
            elif line == 'Not Mintable':
                header['is_mintable'] = line
            elif line == 'Not Freezable':
                header['is_freezable'] = line
            elif line == 'Dex not paid':
                header['is_dex_paid'] = line
            i += 1
        header.setdefault('is_mintable', '')
        header.setdefault('is_freezable', '')
        header.setdefault('is_dex_paid', '')
        while i < len(lines):
            if not lines[i].endswith(':'):
                i += 1
                continue
            addr = lines[i].rstrip(':')
            w = {'wallet_address': addr}
            i += 1
            if i < len(lines) and lines[i].startswith('WinRate:'):
                winrate_raw = lines[i].split(':',1)[1].strip().rstrip('%')
                try:
                    # Trata valores como 'null' ou strings vazias
                    w['wallet_winrate'] = float(winrate_raw) if winrate_raw and winrate_raw.lower() != 'null' else None
                except ValueError:
                    w['wallet_winrate'] = None
                i += 1
            if i < len(lines) and lines[i].startswith('Median Trade Duration:'):
                w['wallet_median_trade_duration'] = lines[i].split(':',1)[1].strip()
                i += 1
            if i < len(lines) and lines[i].startswith('Trades:'):
                w['wallet_trade_number'] = int(lines[i].split(':',1)[1].strip())
                i += 1
            hits = {'6x+':0,'3x-6x':0,'1x-3x':0,'losses':0}
            for _ in range(4):
                part = lines[i]
                num = int(re.search(r'(\d+)', part).group(1))
                if '6x+' in part: hits['6x+'] = num
                elif '3x-6x' in part: hits['3x-6x'] = num
                elif '1x-3x' in part: hits['1x-3x'] = num
                elif 'Loss' in part: hits['losses'] = num
                i += 1
            w.update({
                'wallet_7d_6x_hits': hits['6x+'],
                'wallet_7d_3-6x_hits': hits['3x-6x'],
                'wallet_7d_1-3x_hits': hits['1x-3x'],
                'wallet_7d_losses': hits['losses']
            })
            if i < len(lines) and lines[i].startswith('Historically tied to:'):
                w['wallet_history'] = lines[i].split(':',1)[1].strip()
                i += 1
            wallets.append(w)
        # Build result containing only requested fields
        result = {
            'token_name': header.get('token_name', ''),
            'mint': header.get('mint', ''),
            'market_cap': header.get('market_cap', 0.0),
            'price': header.get('price', 0.0),
            'wallets': wallets
        }
        return result

    async def _run_client(self):
        try:
            await self.client.start()
            logging.info(f"✅ KothMonitorATMDAY listening to messages from user {self.user_id}...")
            await self.client.run_until_disconnected()
        except Exception as e:
            logging.exception(f"Telegram client error: {e}")
        finally:
            self._running = False

    def start(self):
        if not self._running:
            self._running = True
            loop = asyncio.get_event_loop()
            loop.create_task(self._run_client())
            logging.debug("KothMonitorATMDAY started.")

    def stop(self):
        if self._running and self.client.is_connected():
            self._running = False
            loop = asyncio.get_event_loop()
            loop.create_task(self.client.disconnect())
            logging.debug("KothMonitorATMDAY stopped.")

    # --- Utilitário interno -------------------------------------------------
    @staticmethod
    def _parse_abbrev_number(value: str) -> float:
        """Convert strings like '2.3k', '4M', '1.2B' to absolute float.

        If no recognized suffix is found, try to convert directly to float.
        """
        if not isinstance(value, str):
            return float(value)

        value = value.strip()
        multiplier = 1.0
        if value.lower().endswith('k'):
            multiplier = 1e3
            value = value[:-1]
        elif value.lower().endswith('m'):
            multiplier = 1e6
            value = value[:-1]
        elif value.lower().endswith('b'):
            multiplier = 1e9
            value = value[:-1]
        try:
            return float(value) * multiplier
        except ValueError:
            # Return 0.0 if unable to convert, avoiding exceptions at runtime
            logging.warning(f"_parse_abbrev_number: unable to convert '{value}'. Returning 0.0")
            return 0.0

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    def my_callback(new_koth, old_koth_mint):
        print("\n=== New token detected (via Telegram) ===")
        for k, v in new_koth.items():
            if k != 'wallets':
                # Ensure price is printed without scientific notation
                if k == 'price' and isinstance(v, float):
                    print(f"{k}: {(f'{v:.10f}').rstrip('0').rstrip('.')}")
                else:
                    print(f"{k}: {v}")
        print("Wallets:")
        for w in new_koth['wallets']:
            print(w)
        print("Previous token:", old_koth_mint)
        print("==========================================")

    monitor = KothMonitorATMDAY(interval_ms=0, callback=my_callback, include_nsfw=False, show_raw=True)
    monitor.start()
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        monitor.stop()
