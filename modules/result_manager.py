def aggregate_results(results):
    for r in results:
        if not r["pass"]:
            return "NG"
    return "OK"


def build_signal_summary(results):
    """Group ROI results into ordered signal slots by com_index."""
    filtered = [r for r in results if r.get("com_index") is not None]
    filtered.sort(key=lambda r: r["com_index"])

    grouped = {}
    for result in filtered:
        ci = result["com_index"]
        if ci not in grouped:
            grouped[ci] = {
                "com_index": ci,
                "model_name": result.get("model_name", ""),
                "class_id": result.get("class_id", 0),
                "class_name": result.get("class_name", ""),
                "passed": True,
                "roi_results": [],
            }
        group = grouped[ci]
        group["passed"] = group["passed"] and bool(result.get("pass", False))
        group["roi_results"].append(result)

    signal_summary = []
    for com_index in sorted(grouped.keys()):
        group = grouped[com_index]
        status = "OK" if group["passed"] else "NG"
        signal_summary.append({
            "signal_index": com_index,
            "com_index": com_index,
            "signal_code": f"{status}{com_index}",
            "status": status,
            "passed": group["passed"],
            "model_name": group["model_name"],
            "class_id": group["class_id"],
            "class_name": group["class_name"],
            "roi_count": len(group["roi_results"]),
            "roi_results": group["roi_results"],
        })

    return signal_summary


def build_signal_payload(results):
    """Return compact COM payload as comma-separated bits like 1,1,0 where 1=OK and 0=NG."""
    return ",".join("1" if item["passed"] else "0" for item in build_signal_summary(results))
