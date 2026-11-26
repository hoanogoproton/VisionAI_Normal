"""Standalone shared-directory acquisition probe. No model/COM/PLC/source writes."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.image_watcher import ImageWatcher
from modules.ai_worker import decode_packets

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--poll-ms", type=float, default=20)
    parser.add_argument("--queue-images", type=int, default=32)
    parser.add_argument("--queue-mb", type=float, default=128)
    parser.add_argument("--simulate-ai-ms", type=float, default=0,
                        help="Pause consumer to test independent capture/backlog")
    args = parser.parse_args()
    folder = Path(args.folder)
    if not folder.is_dir():
        parser.error("folder must already exist; probe does not create input folders")
    if min(args.seconds, args.poll_ms, args.queue_images, args.queue_mb) <= 0:
        parser.error("duration, poll and queue limits must be positive")
    if args.simulate_ai_ms < 0:
        parser.error("simulate-ai-ms must be >= 0")
    budget = int(args.queue_mb * 1024 * 1024)
    watcher = ImageWatcher(
        folder, poll_interval=args.poll_ms / 1000,
        queue_max_images=args.queue_images, queue_max_bytes=budget,
        max_image_bytes=min(32 * 1024 * 1024, budget))
    exit_code = 0
    decoded_count = 0
    started = time.monotonic()
    print("PROBE: existing files are ignored; create NEW images during this run.", flush=True)
    print("No source writes, deletion, models, COM or PLC output.", flush=True)
    watcher.start()
    try:
        next_report = time.monotonic() + 5
        while time.monotonic() - started < args.seconds:
            for code, message in watcher.drain_events():
                print(f"{code}: {message}", flush=True)
            watcher.check_health()
            packet = watcher.get_packet_nowait()
            if packet is not None:
                images = decode_packets({"PROBE": (watcher, packet)})
                if images:
                    decoded_count += 1
                    print(f"DECODED: {packet.filename}; bytes={len(packet.data)}", flush=True)
                    if args.simulate_ai_ms:
                        time.sleep(args.simulate_ai_ms / 1000)
            else:
                time.sleep(0.001)
            if time.monotonic() >= next_report:
                print("STATS: " + json.dumps(watcher.stats()), flush=True)
                next_report = time.monotonic() + 5
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        exit_code = 2
        print(f"PROBE_ERROR: {exc}", flush=True)
    finally:
        watcher.stop()
        watcher.discard_pending("probe ended before processing")
        for code, message in watcher.drain_events():
            print(f"{code}: {message}", flush=True)
        report = watcher.stats()
        report["decoded_count"] = decoded_count
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["exit_code"] = exit_code
        print("FINAL: " + json.dumps(report), flush=True)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
