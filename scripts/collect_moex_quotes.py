from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import os
import signal
import sys
import time
from zoneinfo import ZoneInfo

import requests

MOSCOW = ZoneInfo("Europe/Moscow")
ISS_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
    "?iss.meta=off&iss.only=marketdata"
    "&marketdata.columns=SECID,BID,OFFER,SPREAD,LAST,NUMTRADES,VALTODAY,UPDATETIME,TIME,SYSTIME"
)
QUOTE_FIELDS = ["poll_utc", "SECID", "BID", "OFFER", "SPREAD", "LAST", "NUMTRADES", "VALTODAY", "UPDATETIME", "TIME", "SYSTIME"]
POLL_FIELDS = ["poll_utc", "status", "n_securities", "n_changed", "latency_s"]

log = logging.getLogger("collect_moex_quotes")


class Collector:
    def __init__(self, out_dir: str, interval: float, timeout: float):
        self.out_dir = out_dir
        self.interval = interval
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "edge-project-m2 (research seminar, HSE)"
        self.last_state: dict[str, tuple] = {}
        self.current_day: dt.date | None = None
        self.quote_file = None
        self.quote_writer = None
        self.poll_file = None
        self.poll_writer = None
        self.stop = False

    def _open_day(self, day: dt.date) -> None:
        self._close_day()
        os.makedirs(self.out_dir, exist_ok=True)
        qpath = os.path.join(self.out_dir, f"moex_quotes_{day.isoformat()}.csv")
        ppath = os.path.join(self.out_dir, f"moex_polls_{day.isoformat()}.csv")
        q_new = not os.path.exists(qpath) or os.path.getsize(qpath) == 0
        p_new = not os.path.exists(ppath) or os.path.getsize(ppath) == 0
        self.quote_file = open(qpath, "a", newline="", encoding="utf-8")
        self.poll_file = open(ppath, "a", newline="", encoding="utf-8")
        self.quote_writer = csv.DictWriter(self.quote_file, fieldnames=QUOTE_FIELDS)
        self.poll_writer = csv.DictWriter(self.poll_file, fieldnames=POLL_FIELDS)
        if q_new:
            self.quote_writer.writeheader()
        if p_new:
            self.poll_writer.writeheader()
        self.current_day = day
        self.last_state = {}
        log.info("writing to %s", qpath)

    def _close_day(self) -> None:
        for f in (self.quote_file, self.poll_file):
            if f is not None:
                f.flush()
                f.close()
        self.quote_file = self.poll_file = None
        self.quote_writer = self.poll_writer = None

    def poll_once(self) -> None:
        now_utc = dt.datetime.now(dt.timezone.utc)
        day = now_utc.astimezone(MOSCOW).date()
        if day != self.current_day:
            self._open_day(day)
        t0 = time.monotonic()
        status = "ok"
        n_sec = 0
        n_changed = 0
        try:
            r = self.session.get(ISS_URL, timeout=self.timeout)
            r.raise_for_status()
            block = r.json()["marketdata"]
            cols = block["columns"]
            rows = block["data"]
            idx = {c: i for i, c in enumerate(cols)}
            n_sec = len(rows)
            for row in rows:
                secid = row[idx["SECID"]]
                state = tuple(row[idx[c]] for c in ("BID", "OFFER", "LAST", "NUMTRADES", "TIME"))
                if self.last_state.get(secid) == state:
                    continue
                self.last_state[secid] = state
                n_changed += 1
                rec = {"poll_utc": now_utc.isoformat(timespec="seconds")}
                rec.update({c: row[idx[c]] for c in QUOTE_FIELDS[1:]})
                self.quote_writer.writerow(rec)
        except (requests.RequestException, ValueError, KeyError) as exc:
            status = f"error:{type(exc).__name__}"
            log.warning("poll failed: %r", exc)
        latency = time.monotonic() - t0
        self.poll_writer.writerow({
            "poll_utc": now_utc.isoformat(timespec="seconds"),
            "status": status,
            "n_securities": n_sec,
            "n_changed": n_changed,
            "latency_s": f"{latency:.3f}",
        })
        self.quote_file.flush()
        self.poll_file.flush()

    def run(self) -> None:
        def _stop(signum, frame):
            log.info("signal %s received, stopping", signum)
            self.stop = True

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        log.info("collector started, interval %.1fs", self.interval)
        while not self.stop:
            started = time.monotonic()
            self.poll_once()
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, self.interval - elapsed))
        self._close_day()
        log.info("collector stopped")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/moex/quotes", help="output directory")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between polls")
    ap.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    Collector(args.out, args.interval, args.timeout).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
