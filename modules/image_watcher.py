"""
Module: image_watcher
Chuc nang: Theo doi folder va phat hien anh moi (based on user's approach)
Khong phu thuoc: Chi import pathlib, os, shutil
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class ImageWatcher:
    """
    Giám sát folder cho ảnh mới.

    - Chỉ xử lý mỗi phiên bản file một lần.
    - Instant copy sang temp folder để bảo vệ ảnh.
    - Khi đổi mã hàng, các file đã có trong folder được đánh dấu là ảnh cũ.
    """

    IMAGE_PATTERNS = ("*.jpg", "*.JPG", "*.png", "*.PNG")

    def __init__(self, watch_folder: str, temp_folder: str = "images/process_temp",
                 poll_interval: float = 0.5, auto_cleanup: bool = True):
        self.watch_folder = Path(watch_folder)
        self.temp_folder = Path(temp_folder)
        self.poll_interval = poll_interval
        self.auto_cleanup = auto_cleanup

        # filename -> (mtime_ns, ctime_ns, size). A changed file with the same
        # filename is a new image and must be processable.
        self.processed_files: Dict[str, tuple] = {}

        self.watch_folder.mkdir(parents=True, exist_ok=True)
        self.temp_folder.mkdir(parents=True, exist_ok=True)

        # Scan folder hiện tại - đánh dấu để bỏ qua
        self._mark_existing_files()

        print(f"[WATCHER INIT]")
        print(f"  watch_folder: {self.watch_folder.absolute()}")
        print(f"  temp_folder: {self.temp_folder.absolute()}")
        print(f"  poll_interval: {poll_interval}s")

    def _iter_image_files(self) -> List[Path]:
        """Return image files in a deterministic order."""
        files = []
        for pattern in self.IMAGE_PATTERNS:
            files.extend(self.watch_folder.glob(pattern))
        return sorted(files, key=lambda path: path.name.lower())

    @staticmethod
    def _get_file_identity(img_path: Path) -> tuple:
        """Identity changes when a camera overwrites a filename with a new image."""
        stat = img_path.stat()
        return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size

    def _mark_existing_files(self) -> None:
        """Mark files currently present as already handled/ignored."""
        existing = self._iter_image_files()

        for file_path in existing:
            try:
                self.processed_files[file_path.name] = self._get_file_identity(file_path)
            except OSError as e:
                print(f"[WATCHER ERROR] Cannot stat existing file {file_path.name}: {e}")

        if existing:
            print(f"[WATCHER] Ignored {len(existing)} existing files")

    def get_new_images(self) -> List[Dict[str, str]]:
        """
        Lấy danh sách ảnh mới (chưa được xử lý).

        Chỉ trả về file xuất hiện hoặc bị ghi đè sau lần reset watcher gần nhất.
        """
        new_images = []

        try:
            for img_path in self._iter_image_files():
                try:
                    file_identity = self._get_file_identity(img_path)
                except OSError as e:
                    print(f"[WATCHER ERROR] Cannot stat {img_path.name}: {e}")
                    continue

                previous_identity = self.processed_files.get(img_path.name)
                if previous_identity == file_identity:
                    continue

                self.processed_files[img_path.name] = file_identity

                try:
                    temp_path = self.temp_folder / img_path.name
                    shutil.copy2(img_path, temp_path)

                    new_images.append({
                        'filename': img_path.name,
                        'original_path': str(img_path),
                        'temp_path': str(temp_path)
                    })
                except Exception as e:
                    print(f"[WATCHER ERROR] Copy failed: {img_path.name} - {e}")
                    # Retry later unless a newer version has already replaced it.
                    if self.processed_files.get(img_path.name) == file_identity:
                        self.processed_files.pop(img_path.name, None)

        except Exception as e:
            print(f"[WATCHER ERROR] Scan folder failed: {e}")

        return new_images

    def cleanup_temp_file(self, temp_path: str) -> None:
        """Xóa file temp sau khi xử lý xong."""
        try:
            if self.auto_cleanup and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"[WATCHER] Cleanup error: {e}")

    def reset(self) -> None:
        """
        Start a new image batch.

        Every image present in the input folder at this instant is marked as
        old. Only files that appear or are overwritten after this reset are
        returned by ``get_new_images``.
        """
        self.processed_files.clear()
        self._mark_existing_files()
        print("[WATCHER] Reset - ignored images currently in folder")


# ============================================================================
# TEST
# ============================================================================

def main():
    """Test watcher"""
    print("[TEST] ImageWatcher\n")
    
    # Test 1: Create watcher
    print("[1] Create watcher:")
    watcher = ImageWatcher(
        watch_folder="test_watch",
        temp_folder="test_temp",
        poll_interval=0.5
    )
    print()
    
    # Test 2: Create dummy images
    print("[2] Create test images:")
        
    watch_path = Path("test_watch")
    watch_path.mkdir(exist_ok=True)
    
    for i in range(3):
        img_path = watch_path / f"test_{i:03d}.jpg"
        with open(img_path, 'wb') as f:
            f.write(b'FAKE_IMAGE_DATA')
        time.sleep(0.1)
        print(f"  Created: {img_path.name}")
    print()
    
    # Test 3: Get new images
    print("[3] Get new images first time:")
    new_imgs = watcher.get_new_images()
    print(f"  Found: {len(new_imgs)} new images")
    for img in new_imgs:
        print(f"    - {img['filename']}")
        print(f"      temp: {img['temp_path']}")
    print()
    
    # Test 4: Get new images again (should be empty)
    print("[4] Get new images again (should be empty):")
    new_imgs = watcher.get_new_images()
    print(f"  Found: {len(new_imgs)} new images (expected 0)\n")
    
    # Test 5: Add new image and check
    print("[5] Add new image:")
    new_img = watch_path / "test_new.jpg"
    with open(new_img, 'wb') as f:
        f.write(b'NEW_IMAGE_DATA')
    print(f"  Created: {new_img.name}")
    
    new_imgs = watcher.get_new_images()
    print(f"  Found: {len(new_imgs)} new images")
    for img in new_imgs:
        print(f"    - {img['filename']}")
    print()
    
    # Cleanup
    print("[CLEANUP]:")
    import shutil
    shutil.rmtree("test_watch", ignore_errors=True)
    shutil.rmtree("test_temp", ignore_errors=True)
    print("  Removed test folders")
    print()
    
    print("[DONE] Tests completed!")


if __name__ == "__main__":
    main()
