"""
Module: image_watcher
Chuc nang: Theo doi folder va phat hien anh moi (based on user's approach)
Khong phu thuoc: Chi import pathlib, os, shutil
"""

import os
import time
import shutil
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


class ImageWatcher:
    """
    Giám sát folder cho ảnh mới
    - Chỉ xử lý ảnh chưa được xử lý
    - Instant copy sang temp folder để bảo vệ ảnh
    """
    
    def __init__(self, watch_folder: str, temp_folder: str = "images/process_temp",
                 poll_interval: float = 0.5, auto_cleanup: bool = True):
        """
        Khoi tao watcher
        
        Args:
            watch_folder (str): Folder theo dõi (VD: "images/temp/CAM1")
            temp_folder (str): Folder lưu copy an toàn
            poll_interval (float): Thời gian kiểm tra (giây)
            auto_cleanup (bool): Tự động xóa file temp sau khi xử lý
        """
        self.watch_folder = Path(watch_folder)
        self.temp_folder = Path(temp_folder)
        self.poll_interval = poll_interval
        self.auto_cleanup = auto_cleanup
        
        # Nhớ file đã xử lý (tránh xử lý lại)
        self.processed_files = set()
        
        # Tạo folder nếu chưa có
        self.watch_folder.mkdir(parents=True, exist_ok=True)
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        
        # Scan folder hiện tại - đánh dấu để bỏ qua
        self._mark_existing_files()
        
        print(f"[WATCHER INIT]")
        print(f"  watch_folder: {self.watch_folder.absolute()}")
        print(f"  temp_folder: {self.temp_folder.absolute()}")
        print(f"  poll_interval: {poll_interval}s")
    
    def _mark_existing_files(self) -> None:
        """Đánh dấu file hiện có để bỏ qua lần đầu chạy"""
        existing = list(self.watch_folder.glob('*.jpg')) + \
                  list(self.watch_folder.glob('*.JPG')) + \
                  list(self.watch_folder.glob('*.png')) + \
                  list(self.watch_folder.glob('*.PNG'))
        
        for f in existing:
            self.processed_files.add(f.name)
        
        if existing:
            print(f"[WATCHER] Ignored {len(existing)} existing files")
    
    def get_new_images(self) -> List[Dict[str, str]]:
        """
        Lấy danh sách ảnh mới (chưa được xử lý)
        
        Returns:
            List[Dict]: Danh sách ảnh mới:
                [
                    {
                        'filename': str,
                        'original_path': str (đường dẫn gốc),
                        'temp_path': str (đường dẫn sau copy)
                    },
                    ...
                ]
        """
        new_images = []
        
        try:
            # Tìm tất cả ảnh trong watch folder
            image_files = list(self.watch_folder.glob('*.jpg')) + \
                         list(self.watch_folder.glob('*.JPG')) + \
                         list(self.watch_folder.glob('*.png')) + \
                         list(self.watch_folder.glob('*.PNG'))
            
            for img_path in image_files:
                # Bỏ qua nếu đã xử lý
                if img_path.name in self.processed_files:
                    continue
                
                # Đánh dấu đã xử lý
                self.processed_files.add(img_path.name)
                
                # Copy sang temp folder
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
                    # Bỏ đánh dấu để retry lần sau
                    self.processed_files.remove(img_path.name)
        
        except Exception as e:
            print(f"[WATCHER ERROR] Scan folder failed: {e}")
        
        return new_images
    
    def cleanup_temp_file(self, temp_path: str) -> None:
        """
        Xóa file temp sau khi xử lý xong
        
        Args:
            temp_path (str): Đường dẫn file temp
        """
        try:
            if self.auto_cleanup and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"[WATCHER] Cleanup error: {e}")
    
    def reset(self) -> None:
        """Reset tracking - gọi lại từ đầu"""
        self.processed_files.clear()
        self._mark_existing_files()
        print(f"[WATCHER] Reset - will re-scan all files")


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
    import time
    
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
