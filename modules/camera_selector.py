def get_used_cameras(roi_rules):
    cameras = set()
    for rule in roi_rules:
        cameras.add(rule["camera"])
    return sorted(cameras)
