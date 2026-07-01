from __future__ import annotations
import argparse
import torch
import os
import time

from engine.ensemble.ensemle import (backfill_cold_start,
                                     run_cycle)


def main():
    ap = argparse.ArgumentParser(description="Live severe-weather inference engine (guts only).")
    ap.add_argument("--model", required=True, help="dir with model_1h.pt / model_24h.pt")
    ap.add_argument("--arch", default="lstm", choices=["lstm", "lstm_delta", "lstm_attn"])
    ap.add_argument("--knn-csv", default=None, help="KNN reference CSV (emb_*, class[, prior_*])")
    ap.add_argument("--knn-ks", default="15,60", help="comma-separated K values")
    ap.add_argument("--station", required=True, help="ASOS station id, e.g. ORD")
    ap.add_argument("--location", default=None, help="human label for the station")
    ap.add_argument("--warn-wfo", default=None,
                    help="NWS WFO for ground-truth watches/warnings, e.g. LOT (Chicago)")
    ap.add_argument("--warn-ugc", default=None,
                    help="comma-separated UGC codes to keep, e.g. ILC031 (Cook)")
    ap.add_argument("--scaler-npz", default=None, help="build npz to read scaler_mean/std from")
    ap.add_argument("--work-dir", default="live_work", help="scratch dir for fetched CSVs")
    ap.add_argument("--out", default="live_log.json", help="JSON log to append to")
    ap.add_argument("--embeddings", default=None,
                    help="JSONL embedding store (default: embeddings_<station>.jsonl)")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    ap.add_argument("--interval-min", type=int, default=60, help="loop period in minutes")
    ap.add_argument("--backfill-days", type=int, default=7,
                    help="on first run (no log yet), reconstruct this many days of "
                         "hourly history before going live (0 disables)")
    args = ap.parse_args()
    args.location = args.location or args.station
    args.warn_ugc = ([u.strip() for u in args.warn_ugc.split(",")]
                     if args.warn_ugc else None)
    args.embeddings = args.embeddings or f"embeddings_{args.station}.jsonl"
    os.makedirs(args.work_dir, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[live] device={device.type} arch={args.arch} station={args.station} "
          f"interval={args.interval_min}min once={args.once}")
    
    cold = not os.path.exists(args.out)
    if cold and args.backfill_days > 0:
        try:
            backfill_cold_start(args, device)
        except Exception as e:
            print(f"[backfill] failed ({e!r}); proceeding to live without history")

    if args.once:
        run_cycle(args, device)
        return
    while True:
        try:
            run_cycle(args, device)
        except Exception as e:
            
            print(f"[live] cycle error: {e!r} (will retry next interval)")
            raise e
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
