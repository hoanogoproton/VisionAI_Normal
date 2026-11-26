"""Independent acquisition worker: scandir -> shared read -> bounded RAM FIFO.

No OpenCV, AI, source mutation, temporary copies or GUI work in this module.
Acknowledge a packet only after decoding it successfully.
"""
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from modules.shared_image_reader import (
    file_identity, read_shared_bytes, ImageNotReady, ImageTooLarge)

class AcquisitionOverload(RuntimeError):
    pass

@dataclass(frozen=True)
class ImagePacket:
    token: int
    generation: int
    filename: str
    original_path: str
    identity: tuple
    captured_at: float
    data: bytes

@dataclass
class _Candidate:
    identity: tuple
    first_seen: float
    next_try: float
    attempts: int = 0
    state: str = "candidate"

class ImageWatcher:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, watch_folder, temp_folder=None, poll_interval=0.02,
                 auto_cleanup=True, queue_max_images=32,
                 queue_max_bytes=128 * 1024 * 1024,
                 max_image_bytes=32 * 1024 * 1024,
                 retry_delays=(0.005, 0.010, 0.020, 0.040),
                 retry_timeout=0.25):
        if float(poll_interval) <= 0:
            raise ValueError("poll_interval must be > 0")
        if min(int(queue_max_images), int(queue_max_bytes),
               int(max_image_bytes)) <= 0:
            raise ValueError("Queue/image limits must be > 0")
        if int(max_image_bytes) > int(queue_max_bytes):
            raise ValueError("max_image_bytes cannot exceed queue_max_bytes")
        if float(retry_timeout) <= 0 or any(float(x) < 0 for x in retry_delays):
            raise ValueError("Invalid retry configuration")
        self.watch_folder = Path(watch_folder)
        # Compatibility metadata ONLY. No temp file is written/deleted.
        self.temp_folder = Path(temp_folder) if temp_folder else None
        self.auto_cleanup = auto_cleanup
        self.poll_interval = float(poll_interval)
        self.queue_max_images = int(queue_max_images)
        self.queue_max_bytes = int(queue_max_bytes)
        self.max_image_bytes = int(max_image_bytes)
        self.retry_delays = tuple(float(x) for x in retry_delays)
        self.retry_timeout = float(retry_timeout)
        self.watch_folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._generation = 0
        self._sequence = 0
        self._queue = deque()
        self._leases = {}  # token -> (packet, candidate-key)
        self._records = {}
        self.processed_files = {}
        self._events = deque(maxlen=256)
        self._fault = None
        self._stats = {
            "scans": 0, "captured": 0, "accepted": 0, "retries": 0,
            "read_errors": 0, "decode_errors": 0, "missing": 0,
            "exhausted": 0, "boundary_discarded": 0,
            "queue_bytes": 0, "queue_items": 0, "overloads": 0,
            "events_coalesced": 0, "read_ms_total": 0.0, "read_ms_max": 0.0,
        }
        self.reset()

    def _event(self, code, message):
        if len(self._events) == self._events.maxlen:
            self._stats["events_coalesced"] += 1
        self._events.append((code, str(message)))

    def drain_events(self):
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def stats(self):
        with self._lock:
            result = dict(self._stats)
            result["generation"] = self._generation
            result["ready_items"] = len(self._queue)
            result["running"] = bool(self._thread and self._thread.is_alive())
            return result

    def check_health(self):
        with self._lock:
            if self._fault is not None:
                raise self._fault

    def _snapshot(self):
        result = {}
        # Exactly one directory enumeration; no four glob passes.
        with os.scandir(self.watch_folder) as entries:
            for entry in entries:
                if Path(entry.name).suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue
                try:
                    if entry.is_file(follow_symlinks=False):
                        result[entry.name] = file_identity(
                            entry.stat(follow_symlinks=False))
                except FileNotFoundError:
                    continue  # ephemeral file vanished during enumeration
        return result

    def reset(self):
        """Explicit product boundary: discard queued OLD packets, ignore existing files.

        In-flight reads/decodes carry a generation and cannot enter the new batch.
        This intentional boundary discard is counted and logged, not silent.
        """
        with self._lock:
            baseline = self._snapshot()
            discarded = len(self._leases)
            self._generation += 1
            self._queue.clear()
            self._leases.clear()
            self._records.clear()
            self.processed_files = baseline
            self._stats["queue_bytes"] = 0
            self._stats["queue_items"] = 0
            self._stats["boundary_discarded"] += discarded
            self._event("BOUNDARY", f"generation={self._generation}; "
                        f"discarded={discarded}; ignored_existing={len(baseline)}")

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self.check_health()
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name=f"capture:{self.watch_folder.name}",
                daemon=True)
            self._thread.start()

    def stop(self, timeout=2.0):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                with self._lock:
                    self._event("STOP_TIMEOUT", "Read/scan still blocked in OS; "
                                "Python cannot cancel synchronous filesystem I/O")

    def get_packet_nowait(self):
        self.check_health()
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def get_new_images(self):
        """Compatibility wrapper: returns ALL queued packets, never copies files."""
        result = []
        while True:
            packet = self.get_packet_nowait()
            if packet is None:
                return result
            result.append({
                "filename": packet.filename,
                "original_path": packet.original_path,
                "bytes": packet.data, "packet": packet,
            })

    def cleanup_temp_file(self, temp_path):
        # Source safety: the new RAM pipeline does not own any temp paths.
        return None

    def _release(self, token):
        lease = self._leases.pop(token, None)
        if lease:
            self._stats["queue_bytes"] -= len(lease[0].data)
            self._stats["queue_items"] -= 1
        return lease

    def accept(self, packet):
        """Decoded successfully. Returns False for an obsolete generation."""
        with self._lock:
            lease = self._release(packet.token)
            if not lease or packet.generation != self._generation:
                return False
            candidate = self._records.get(lease[1])
            if candidate:
                candidate.state = "accepted"
            self.processed_files[packet.filename] = packet.identity
            self._stats["accepted"] += 1
            return True

    def reject(self, packet, reason):
        """Decode failed: release RAM and schedule bounded reacquisition."""
        with self._lock:
            lease = self._release(packet.token)
            if not lease or packet.generation != self._generation:
                return
            candidate = self._records.get(lease[1])
            self._stats["decode_errors"] += 1
            self._event("DECODE_ERROR", f"{packet.filename}: {reason}")
            if candidate:
                self._retry(candidate, packet.filename, time.monotonic())

    def discard(self, packet, reason):
        """Explicit policy discard (e.g. shutdown/product boundary), never success."""
        with self._lock:
            lease = self._release(packet.token)
            if lease:
                candidate = self._records.get(lease[1])
                if candidate:
                    candidate.state = "failed"
                self._stats["boundary_discarded"] += 1
                self._event("DISCARD", f"{packet.filename}: {reason}")

    def discard_pending(self, reason):
        with self._lock:
            packets = [lease[0] for lease in self._leases.values()]
            self._queue.clear()
            for packet in packets:
                self.discard(packet, reason)

    def _retry(self, candidate, name, now):
        index = candidate.attempts - 1
        if (index >= len(self.retry_delays)
                or now - candidate.first_seen >= self.retry_timeout):
            candidate.state = "failed"
            self._stats["exhausted"] += 1
            self._event("RETRY_EXHAUSTED", name)
            return
        candidate.state = "candidate"
        candidate.next_try = now + self.retry_delays[max(0, index)]
        self._stats["retries"] += 1

    def _overload(self, message):
        self._stats["overloads"] += 1
        self._fault = AcquisitionOverload(
            f"{self.watch_folder}: {message}. Acquisition stopped; "
            "no overwrite/drop-oldest policy is used. Producer is NOT stopped.")
        self._event("OVERLOAD", self._fault)
        self._stop.set()

    def _scan_once(self):
        with self._lock:
            generation = self._generation
        snapshot = self._snapshot()
        now = time.monotonic()
        with self._lock:
            if generation != self._generation:
                return
            self._stats["scans"] += 1
            # Prune absent terminal versions so memory does not grow per filename.
            for name in list(self.processed_files):
                if name not in snapshot:
                    del self.processed_files[name]
            for key, candidate in list(self._records.items()):
                current = snapshot.get(key[0])
                if current != key[1] and candidate.state != "queued":
                    if candidate.state == "candidate":
                        self._stats["missing"] += 1
                        self._event("MISSING_OR_REPLACED", key[0])
                    del self._records[key]
            for name, identity in snapshot.items():
                key = (name, identity)
                if self.processed_files.get(name) == identity:
                    continue
                if key not in self._records:
                    self._records[key] = _Candidate(identity, now, now)
            ready = [
                key for key, candidate in self._records.items()
                if candidate.state == "candidate"
                and candidate.next_try <= now and snapshot.get(key[0]) == key[1]
            ]

            # Ordering is deterministic within a scan, not a producer batch guarantee.
            ready.sort(key=lambda key: (
                self._records[key].first_seen, key[1][2], key[0].casefold()))
        # One acquisition worker per folder; never hold state lock during read.
        for key in ready:
            if self._stop.is_set():
                break
            with self._lock:
                candidate = self._records.get(key)
                if (generation != self._generation or candidate is None
                        or candidate.state != "candidate"):
                    continue
                if (candidate.attempts and
                        time.monotonic() - candidate.first_seen >= self.retry_timeout):
                    candidate.state = "failed"
                    self._stats["exhausted"] += 1
                    self._event("RETRY_EXHAUSTED", key[0])
                    continue
                if key[1][-1] > self.max_image_bytes:
                    candidate.state = "failed"
                    self._stats["read_errors"] += 1
                    self._event("IMAGE_TOO_LARGE", key[0])
                    continue
                # Capacity includes both queued and leased (being decoded) bytes.
                if (self._stats["queue_items"] >= self.queue_max_images
                        or self._stats["queue_bytes"] + key[1][-1]
                        > self.queue_max_bytes):
                    self._overload("RAM FIFO capacity exhausted")
                    break
                candidate.attempts += 1
                candidate.state = "reading"
            read_started = time.monotonic()
            try:
                raw, identity = read_shared_bytes(
                    self.watch_folder / key[0], self.max_image_bytes, key[1])
            except OSError as exc:
                with self._lock:
                    if generation != self._generation:
                        continue
                    elapsed_ms = (time.monotonic() - read_started) * 1000
                    self._stats["read_ms_total"] += elapsed_ms
                    self._stats["read_ms_max"] = max(self._stats["read_ms_max"], elapsed_ms)
                    self._stats["read_errors"] += 1
                    error = getattr(exc, "winerror", None)
                    self._event("READ_ERROR", f"{key[0]}: winerror={error}; {exc}")
                    if isinstance(exc, (FileNotFoundError, ImageTooLarge)):
                        candidate.state = "failed"
                        if isinstance(exc, FileNotFoundError):
                            self._stats["missing"] += 1
                    elif error == 5 or (isinstance(exc, PermissionError) and error not in (32, 33)):
                        candidate.state = "failed"
                    else:
                        self._retry(candidate, key[0], time.monotonic())
                continue
            with self._lock:
                elapsed_ms = (time.monotonic() - read_started) * 1000
                self._stats["read_ms_total"] += elapsed_ms
                self._stats["read_ms_max"] = max(self._stats["read_ms_max"], elapsed_ms)
                if generation != self._generation:
                    self._stats["boundary_discarded"] += 1
                    continue
                if (self._stats["queue_items"] >= self.queue_max_images
                        or self._stats["queue_bytes"] + len(raw) > self.queue_max_bytes):
                    candidate.state = "failed"
                    self._overload("RAM FIFO capacity exhausted after read")
                    break
                self._sequence += 1
                packet = ImagePacket(
                    self._sequence, generation, key[0],
                    str(self.watch_folder / key[0]), identity,
                    time.monotonic(), raw)
                self._leases[packet.token] = (packet, key)
                self._queue.append(packet)
                candidate.state = "queued"
                self._stats["queue_items"] += 1
                self._stats["queue_bytes"] += len(raw)
                self._stats["captured"] += 1

    def _run(self):
        next_scan = time.monotonic()
        while not self._stop.is_set():
            remaining = next_scan - time.monotonic()
            if remaining > 0 and self._stop.wait(remaining):
                break
            try:
                self._scan_once()
            except Exception as exc:
                with self._lock:
                    self._fault = RuntimeError(
                        f"Acquisition worker failed for {self.watch_folder}: {exc}")
                    self._event("WORKER_ERROR", self._fault)
                self._stop.set()
                break
            # Real scan schedule. Never execute catch-up bursts after a slow scan.
            next_scan = max(next_scan + self.poll_interval,
                            time.monotonic() + 0.001)
