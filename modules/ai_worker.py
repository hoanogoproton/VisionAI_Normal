"""Jobs for the single AI executor. No GUI, COM or source-file access."""
from pathlib import Path
import time
import cv2
import numpy as np

def decode_packets(items, require_complete_markers=True,
                   max_decoded_pixels=40_000_000):
    """items: camera -> (watcher, packet). Returns valid images only.

    Marker checks reject common partial writes; they are not an atomicity proof.
    Failed packets are NACKed for bounded retry, not marked processed.
    """
    decoded = {}
    for cam, (watcher, packet) in items.items():
        try:
            raw = packet.data
            suffix = Path(packet.filename).suffix.lower()
            if require_complete_markers:
                if suffix in (".jpg", ".jpeg"):
                    if not raw.startswith(b"\xff\xd8") or b"\xff\xd9" not in raw[2:]:
                        raise ValueError("JPEG missing SOI/EOI")
                elif suffix == ".png":
                    if (not raw.startswith(b"\x89PNG\r\n\x1a\n")
                            or not raw.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")):
                        raise ValueError("PNG missing signature/IEND")
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("cv2.imdecode returned None")
            if image.shape[0] * image.shape[1] > max_decoded_pixels:
                raise ValueError("Decoded image exceeds max_decoded_pixels")
            if watcher.accept(packet):
                decoded[cam] = image
        except Exception as exc:
            watcher.reject(packet, str(exc))
    return decoded

def detect_images(images, roi_rules, batch_num, detect_callback):
    start = time.time()
    results, gui_items = {}, {}
    # One executor thread serializes model/cache use across batches and cameras.
    for cam, image in images.items():
        results[cam], gui_items[cam] = detect_callback(
            cam, image, roi_rules, batch_num)
    return {
        "images": images, "results": results,
        "gui_items": gui_items, "started_at": start,
    }
