"""
Module: config_loader
Chuc nang: Load config tu file ngoai (YAML/JSON) thay vi hard-code trong main.py
    - Uu tien YAML (de doc, co comment)
    - Fallback JSON (neu khong co PyYAML)
    - Fallback default dict (neu khong co file)
    - Don gian, de debug

Khong phu thuoc module khac trong project.
"""

import os
import json
from typing import Dict, Any, Optional


def get_default_config() -> Dict[str, Any]:
    """
    Tra ve config mac dinh neu khong co file config.
    Dung khi:
      - File config khong ton tai
      - File config bi loi
      - Khong cai PyYAML hoac json bi loi
    """
    return {
        'product': {
            'mode': 'manual',
            'code': 'ABC123x',
            'default_code': 'ABC123x',
            'com_input_port': 'COM4',
            'com_input_baudrate': 9600,
            'com_input_timeout': 1.0,
            'com_input_poll_interval': 0.5,
            'log_root_dir': 'D:/anh',
            'log_filename': 'WH_AVIS_System.log',
            'log_date_folder_format': '%Y%m%d',
            'log_poll_interval': 0.5,
            'csv_path': 'config/products/{code}.csv'
        },
        'paths': {
            'output_dir': 'output/results',
            'log_dir': 'output/logs'
        },
        'camera': {
            'config_csv': 'config/camera_config.csv',
            'create_folders': True,
            'poll_interval': 0.5
        },
        'com_output': {
            'enabled': True,
            'port': 'COM5',
            'baudrate': 9600,
            'retry_count': 3
        },
        'gui': {
            'enabled': True,
            'window_name': 'VisionAI - {product_code}',
            'max_history': 10
        },
        'result': {
            'save_result': True
        }
    }


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """
    Load config tu file YAML hoac JSON.
    
    Priority:
      1. YAML file (neu co PyYAML)
      2. JSON file (neu co .json extension)
      3. Default dict (neu file khong ton tai)
    
    Args:
        config_file: Duong dan file config (mac dinh: "config.yaml")
        
    Returns:
        Dict chua config
        
    Raises:
        Khong raise exception, luon tra ve config (fallback to default)
    """
    
    # Kiem tra file ton tai
    if not os.path.exists(config_file):
        print(f"[CONFIG] WARNING: Config file not found: {config_file}")
        print(f"[CONFIG] Using default configuration")
        return get_default_config()
    
    # Thu load YAML
    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            print(f"[CONFIG] Loaded from YAML: {config_file}")
            return cfg
        except ImportError:
            print(f"[CONFIG] WARNING: PyYAML not installed")
            print(f"[CONFIG] Install: pip install pyyaml")
            print(f"[CONFIG] Using default configuration")
            return get_default_config()
        except Exception as e:
            print(f"[CONFIG] ERROR: Failed to load YAML: {e}")
            print(f"[CONFIG] Using default configuration")
            return get_default_config()
    
    # Thu load JSON
    elif config_file.endswith('.json'):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            print(f"[CONFIG] Loaded from JSON: {config_file}")
            return cfg
        except Exception as e:
            print(f"[CONFIG] ERROR: Failed to load JSON: {e}")
            print(f"[CONFIG] Using default configuration")
            return get_default_config()
    
    # File extension khong ho tro
    else:
        print(f"[CONFIG] WARNING: Unsupported file type: {config_file}")
        print(f"[CONFIG] Using default configuration")
        return get_default_config()


def save_config_template(output_file: str = "config_template.yaml") -> None:
    """
    Tao file config mau de tham khao.
    
    Args:
        output_file: Duong dan file output (mac dinh: "config_template.yaml")
    """
    cfg = get_default_config()
    
    if output_file.endswith('.yaml') or output_file.endswith('.yml'):
        try:
            import yaml
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            print(f"[CONFIG] Template saved: {output_file}")
        except ImportError:
            print(f"[CONFIG] ERROR: PyYAML not installed, cannot save YAML template")
    
    elif output_file.endswith('.json'):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[CONFIG] Template saved: {output_file}")
    
    else:
        print(f"[CONFIG] ERROR: Unsupported file type: {output_file}")


def get_value(cfg: Dict, key_path: str, default: Any = None) -> Any:
    """
    Lay gia tri tu config dict theo key path.
    
    Args:
        cfg: Config dict
        key_path: Duong dan key, VD: "com_output.port" hoac "product.code"
        default: Gia tri mac dinh neu khong tim thay
        
    Returns:
        Gia tri tim duoc hoac default
        
    Example:
        >>> cfg = {'com_output': {'port': 'COM5'}}
        >>> get_value(cfg, 'com_output.port')
        'COM5'
        >>> get_value(cfg, 'com_output.timeout', 1.0)
        1.0
    """
    keys = key_path.split('.')
    value = cfg
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


# ============================================================================
# TEST
# ============================================================================

def main():
    """Test config loader"""
    print("[TEST] Config Loader\n")
    
    # Test 1: Load default config
    print("[1] Test default config:")
    cfg = get_default_config()
    print(f"  Product code: {cfg['product']['code']}")
    print(f"  COM port: {cfg['com_output']['port']}")
    print()
    
    # Test 2: Load config file (se that bai neu khong co)
    print("[2] Test load config file:")
    cfg2 = load_config("config.yaml")
    print(f"  Loaded config: {type(cfg2)}")
    print()
    
    # Test 3: Get value helper
    print("[3] Test get_value helper:")
    port = get_value(cfg2, 'com_output.port', 'COM3')
    print(f"  com_output.port = {port}")
    timeout = get_value(cfg2, 'com_output.timeout', 1.0)
    print(f"  com_output.timeout = {timeout} (default)")
    print()
    
    # Test 4: Save template
    print("[4] Test save template:")
    save_config_template("config_template.yaml")
    save_config_template("config_template.json")
    print()
    
    print("[DONE] Test completed!")


if __name__ == "__main__":
    main()
