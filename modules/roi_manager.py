def crop_roi(image, roi):
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2]

def prepare_roi_data(image, rule):
    detect_img = crop_roi(image, rule["detect_roi"])

    return {
        "roi_id": rule["roi_id"],
        "camera": rule["camera"],
        "detect_image": detect_img,
        "compare_roi": rule["compare_roi"],
        "rule": rule
    }
