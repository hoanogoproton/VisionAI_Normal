"""
Module: model_manager
Chuc nang: Quan ly va cache model YOLO
Khong phu thuoc: Khong import module khac tu project
"""

from ultralytics import YOLO
from typing import Optional
import os

_model_cache = {}

def get_model(model_name: str) -> YOLO:
    """
    Tải hay lấy model từ cache
    
    Args:
        model_name (str): Tên model (không .pt, VD: "MarkF")
        
    Returns:
        YOLO: Model YOLO đã tải
        
    Raises:
        FileNotFoundError: File model không tồn tại
    """
    if model_name not in _model_cache:
        model_path = f"models/{model_name}.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        _model_cache[model_name] = YOLO(model_path)
        print(f"[LOAD] Model '{model_name}' loaded")
    else:
        print(f"[CACHE] Model '{model_name}' from cache")
    
    return _model_cache[model_name]


def clear_cache() -> None:
    """Xóa tất cả model khỏi cache"""
    _model_cache.clear()
    print("[CLEAR] Cache cleared")


def get_cached_models() -> list:
    """Lấy danh sách model đang trong cache"""
    return list(_model_cache.keys())


# ============================================================================
# TEST
# ============================================================================

def main():
    """Test function"""
    print("[TEST] Model Manager\n")
    
    # Test 1: Load model
    print("[1] Test load model:")
    try:
        model = get_model("MarkF")
        print(f"[OK] Model loaded: {type(model).__name__}\n")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[INFO] Tao file test...\n")
        os.makedirs("models", exist_ok=True)
        print("[SKIP] Can't create dummy model, requires actual YOLO file\n")
    
    # Test 2: Load same model again (from cache)
    print("[2] Test cache (load same model again):")
    try:
        model = get_model("MarkF")
        cached = get_cached_models()
        print(f"[OK] Cached models: {cached}\n")
    except FileNotFoundError:
        print("[SKIP] Model file not found\n")
    
    # Test 3: Load different model
    print("[3] Test load different model:")
    try:
        model = get_model("MarkF2")
        cached = get_cached_models()
        print(f"[OK] Cached models: {cached}\n")
    except FileNotFoundError:
        print("[SKIP] Model file not found\n")
    
    # Test 4: Clear cache
    print("[4] Test clear cache:")
    clear_cache()
    cached = get_cached_models()
    print(f"[OK] Cached models after clear: {cached}\n")
    
    print("[DONE] Tests completed!")


if __name__ == "__main__":
    main()
