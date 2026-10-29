# Simplified image reset on product scan

When a valid product scan is accepted, the application starts a new image batch:

1. Temporary files from a partial previous batch are removed.
2. Each active camera watcher scans its input folder and marks all files currently present as old.
3. Only files that appear later, or are overwritten later, are collected for detection.

No image file timestamp is compared with the scan time. Therefore, a file that is copied into the camera folder after the reset is accepted even if that file preserves an older source timestamp.
