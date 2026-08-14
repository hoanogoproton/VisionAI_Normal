import csv
import os


def _parse_optional_int(value):
    """Parse optional int field - tra ve None neu trong/khong co"""
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _parse_optional_float(value):
    """Parse optional float field - tra ve None neu trong/khong co"""
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _parse_optional_bool(value):
    """Parse optional bool field - tra ve False neu trong/khong co"""
    if value is None or str(value).strip() == "":
        return False
    text = str(value).strip().lower()
    return text in ("true", "1", "yes")


def _parse_optional_str(value, default: str = ""):
    """Parse optional str field - tra ve default neu trong/khong co"""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def sort_product_rules(roi_rules):
    """Sort rules by model_name, class_id, then roi_id."""
    return sorted(
        roi_rules,
        key=lambda rule: (
            rule.get("model_name", ""),
            rule.get("class_id", 0),
            rule.get("roi_id", ""),
        ),
    )


def build_product_signal_plan(roi_rules):
    """Group loaded rules into ordered signal slots by com_index."""
    filtered = [r for r in roi_rules if r.get("com_index") is not None]
    filtered.sort(key=lambda r: (r["com_index"], r.get("roi_id", "")))

    grouped = {}
    for rule in filtered:
        ci = rule["com_index"]
        if ci not in grouped:
            grouped[ci] = {
                "model_name": rule.get("model_name", ""),
                "class_id": rule.get("class_id", 0),
                "class_name": rule.get("class_name", ""),
                "roi_ids": [],
            }
        grouped[ci]["roi_ids"].append(rule.get("roi_id", ""))

    signal_plan = []
    for com_index in sorted(grouped.keys()):
        group = grouped[com_index]
        signal_plan.append({
            "signal_index": com_index,
            "ok_signal": f"OK{com_index}",
            "ng_signal": f"NG{com_index}",
            "com_index": com_index,
            "model_name": group["model_name"],
            "class_id": group["class_id"],
            "class_name": group["class_name"],
            "roi_count": len(group["roi_ids"]),
            "roi_ids": group["roi_ids"],
        })

    return signal_plan


def load_product_csv(csv_path):
    """
    Đọc file CSV cấu hình mã hàng.
    
    Input:
        csv_path (str): đường dẫn file CSV
        
    Output:
        roi_rules (list[dict]): danh sách rule ROI
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    roi_rules = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for index, row in enumerate(reader, start=1):
            try:
                rule = {
                    "roi_id": row["roi_id"].strip(),
                    "camera": row["camera"].strip(),
                    "model_name": row["model_name"].strip(),
                    "class_id": int(row["class_id"]),
                    "class_name": row["class_name"].strip(),

                    # ROI detect
                    "detect_roi": (
                        int(row["detect_x_min"]),
                        int(row["detect_y_min"]),
                        int(row["detect_x_max"]),
                        int(row["detect_y_max"]),
                    ),

                    # ROI compare
                    "compare_roi": (
                        int(row["compare_x_min"]),
                        int(row["compare_y_min"]),
                        int(row["compare_x_max"]),
                        int(row["compare_y_max"]),
                    ),

                    "confidence": float(row["confidence"]),

                    # Check mode (optional)
                    "check_mode": _parse_optional_str(row.get("check_mode"), "compare").lower(),
                    "min_count": _parse_optional_int(row.get("min_count")),
                    "max_count": _parse_optional_int(row.get("max_count")),

                    # Keypoint angle check (optional - de trong neu khong can)
                    "keypoint_idx_1": _parse_optional_int(row.get("keypoint_idx_1")),
                    "keypoint_idx_2": _parse_optional_int(row.get("keypoint_idx_2")),
                    "expected_angle": _parse_optional_float(row.get("expected_angle")),
                    "angle_tolerance": _parse_optional_float(row.get("angle_tolerance")),

                    # Wire width measurement (optional - dung khi check_mode=measure_width)
                    "width_min_mm": _parse_optional_float(row.get("width_min_mm")),
                    "width_max_mm": _parse_optional_float(row.get("width_max_mm")),
                    "aruco_marker_size_mm": _parse_optional_float(row.get("aruco_marker_size_mm")),
                    "aruco_dict": _parse_optional_str(row.get("aruco_dict"), "DICT_4X4_50"),
                    "aruco_id": _parse_optional_int(row.get("aruco_id")),
                    "require_aruco": _parse_optional_bool(row.get("require_aruco")),
                    "trim_fraction": _parse_optional_float(row.get("trim_fraction")) if _parse_optional_float(row.get("trim_fraction")) is not None else 0.10,
                    "p_low": _parse_optional_float(row.get("p_low")) if _parse_optional_float(row.get("p_low")) is not None else 10,
                    "p_high": _parse_optional_float(row.get("p_high")) if _parse_optional_float(row.get("p_high")) is not None else 90,

                    # COM signal index (optional - de trong neu khong gui tin hieu)
                    "com_index": _parse_optional_int(row.get("com_index")),
                }

                roi_rules.append(rule)

            except Exception as e:
                print(f"[ERROR] Row {index} invalid: {e}")
                print("Row data:", row)

    return sort_product_rules(roi_rules)


# ==========================================================
# TEST NHANH MODULE
# ==========================================================
def main():
    """
    Test nhanh file CSV
    """

    # Ví dụ đường dẫn CSV
    csv_path = "config/products/ABC123.csv"

    print("Reading file:", csv_path)

    try:
        rules = load_product_csv(csv_path)

        print("\n===== PARSE RESULT =====")
        print(f"ROI count: {len(rules)}\n")

        for r in rules:
            print(r)

    except Exception as e:
        print("LỖI:", e)


if __name__ == "__main__":
    main()
