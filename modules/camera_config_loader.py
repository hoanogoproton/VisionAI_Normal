"""
Module: camera_config_loader
Chức năng: Load cấu hình camera từ CSV
Không phụ thuộc: Chỉ import csv, pathlib, os

CÁCH DÙNG:
    from modules.camera_config_loader import load_camera_config
    
    config = load_camera_config("config/camera_config.csv")
    
    # Lấy đường dẫn CAM1
    cam1_input = config["CAM1"]["input_folder"]
    cam1_temp = config["CAM1"]["temp_folder"]
"""

import csv
import os
from pathlib import Path
from typing import Dict, Optional


def validate_camera_folder(folder_path: str) -> bool:
    """
    Kiểm tra folder có tồn tại không
    
    Args:
        folder_path (str): Đường dẫn folder
        
    Returns:
        bool: True nếu tồn tại, False nếu không
    """
    return os.path.exists(folder_path) and os.path.isdir(folder_path)


def load_camera_config(
    csv_file: str = "config/camera_config.csv",
    create_folders: bool = True,
    verbose: bool = True
) -> Dict[str, Dict[str, str]]:
    """
    Load cấu hình camera từ CSV file
    
    Args:
        csv_file (str): Đường dẫn file CSV (mặc định: config/camera_config.csv)
        create_folders (bool): Tự động tạo folder nếu chưa có
        verbose (bool): In log khi chạy
        
    Returns:
        Dict: 
        {
            "CAM1": {
                "input_folder": "images/temp/CAM1",
                "temp_folder": "images/process_temp/CAM1",
                "enabled": True
            },
            "CAM2": {...},
            ...
        }
        
    Ví dụ:
        >>> config = load_camera_config()
        >>> config["CAM1"]["input_folder"]
        'images/temp/CAM1'
        >>> config["CAM1"]["temp_folder"]
        'images/process_temp/CAM1'
    """
    
    config = {}
    
    # Check file CSV có tồn tại không
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"[CAMERA_CONFIG] File not found: {csv_file}")
    
    if verbose:
        print(f"[CAMERA_CONFIG] Loading: {csv_file}")
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                camera_name = row.get("camera_name", "").strip()
                input_folder = row.get("input_folder", "").strip()
                temp_folder = row.get("temp_folder", "").strip()
                enabled = row.get("enabled", "true").strip().lower() == "true"
                
                # Bỏ qua dòng với camera_name trống
                if not camera_name:
                    continue
                
                # Tạo folder nếu cần
                if create_folders and enabled:
                    os.makedirs(input_folder, exist_ok=True)
                    os.makedirs(temp_folder, exist_ok=True)
                
                # Lưu vào dict
                config[camera_name] = {
                    "input_folder": input_folder,
                    "temp_folder": temp_folder,
                    "enabled": enabled
                }
                
                if verbose:
                    status = "✓" if enabled else "✗"
                    print(f"  {status} {camera_name}")
                    print(f"      Input:  {input_folder}")
                    print(f"      Temp:   {temp_folder}")
    
    except Exception as e:
        raise Exception(f"[CAMERA_CONFIG] Error reading CSV: {e}")
    
    if verbose:
        print(f"[CAMERA_CONFIG] Total cameras: {len(config)}\n")
    
    return config


def get_camera_folder(
    camera_name: str,
    config: Dict[str, Dict[str, str]],
    folder_type: str = "input",
    verbose: bool = False
) -> Optional[str]:
    """
    Lấy đường dẫn folder của camera
    
    Args:
        camera_name (str): Tên camera (VD: "CAM1")
        config (dict): Dict từ load_camera_config()
        folder_type (str): "input" hoặc "temp"
        verbose (bool): In log
        
    Returns:
        str: Đường dẫn folder hoặc None nếu không tìm thấy
        
    Ví dụ:
        >>> config = load_camera_config()
        >>> get_camera_folder("CAM1", config, "input")
        'images/temp/CAM1'
        >>> get_camera_folder("CAM1", config, "temp")
        'images/process_temp/CAM1'
    """
    
    if camera_name not in config:
        if verbose:
            print(f"[CAMERA_CONFIG] Camera not found: {camera_name}")
        return None
    
    cam_config = config[camera_name]
    
    if not cam_config["enabled"]:
        if verbose:
            print(f"[CAMERA_CONFIG] Camera disabled: {camera_name}")
        return None
    
    if folder_type == "input":
        folder = cam_config["input_folder"]
    elif folder_type == "temp":
        folder = cam_config["temp_folder"]
    else:
        raise ValueError(f"Invalid folder_type: {folder_type}")
    
    if verbose:
        print(f"[CAMERA_CONFIG] {camera_name} ({folder_type}): {folder}")
    
    return folder


def get_enabled_cameras(config: Dict[str, Dict[str, str]]) -> list:
    """
    Lấy danh sách camera được bật (enabled=true)
    
    Args:
        config (dict): Dict từ load_camera_config()
        
    Returns:
        list: Danh sách tên camera, VD: ["CAM1", "CAM2", "CAM3"]
    """
    return [cam for cam, cfg in config.items() if cfg["enabled"]]


def print_camera_config_summary(config: Dict[str, Dict[str, str]]) -> None:
    """
    In bảng tóm tắt cấu hình camera
    
    Args:
        config (dict): Dict từ load_camera_config()
    """
    print("=" * 100)
    print("CAMERA CONFIG SUMMARY")
    print("=" * 100)
    
    if not config:
        print("  [NO CAMERAS CONFIGURED]")
    else:
        for idx, (cam_name, cam_cfg) in enumerate(config.items(), 1):
            status = "✓ ENABLED" if cam_cfg["enabled"] else "✗ DISABLED"
            print(f"\n  {idx}. {cam_name} - {status}")
            print(f"     Input Folder:  {cam_cfg['input_folder']}")
            print(f"     Temp Folder:   {cam_cfg['temp_folder']}")
    
    print("\n" + "=" * 100 + "\n")


# ============================================================================
# TEST
# ============================================================================

def test_camera_config_loader():
    """Test module"""
    
    print("\n" + "=" * 100)
    print("TEST: camera_config_loader")
    print("=" * 100 + "\n")
    
    # ===== TEST 1: Load config =====
    print("[TEST 1] Load camera config:")
    config = load_camera_config("config/camera_config.csv")
    print()
    
    # ===== TEST 2: Get enabled cameras =====
    print("[TEST 2] Get enabled cameras:")
    enabled = get_enabled_cameras(config)
    print(f"  Enabled cameras: {enabled}\n")
    
    # ===== TEST 3: Get specific camera folders =====
    print("[TEST 3] Get specific camera folders:")
    for cam in ["CAM1", "CAM2", "CAM3"]:
        input_path = get_camera_folder(cam, config, "input", verbose=True)
        temp_path = get_camera_folder(cam, config, "temp", verbose=True)
        print()
    
    # ===== TEST 4: Print summary =====
    print("[TEST 4] Print config summary:")
    print_camera_config_summary(config)
    
    # ===== TEST 5: Test non-existent camera =====
    print("[TEST 5] Get non-existent camera:")
    cam99 = get_camera_folder("CAM99", config, "input", verbose=True)
    print()
    
    print("=" * 100)
    print("TESTS COMPLETED")
    print("=" * 100)


if __name__ == "__main__":
    test_camera_config_loader()