"""Short-lived read-only snapshots; Windows explicitly shares READ/WRITE/DELETE.

No source mutation. Metadata stability is a heuristic, NOT an atomic-write protocol.
"""
import os
import contextlib

class ImageNotReady(OSError):
    pass

class ImageTooLarge(OSError):
    pass

def file_identity(stat):
    """Directory version token. Do not compare Windows CRT fstat with this token."""
    return (stat.st_dev, stat.st_ino, stat.st_mtime_ns,
            stat.st_ctime_ns, stat.st_size)

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    _k32.CreateFileW.restype = wintypes.HANDLE
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.CloseHandle.restype = wintypes.BOOL
    _INVALID = ctypes.c_void_p(-1).value

    class _FileInfo(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("creation", wintypes.FILETIME),
            ("access", wintypes.FILETIME),
            ("write", wintypes.FILETIME),
            ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        ]
    _k32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_FileInfo)]
    _k32.GetFileInformationByHandle.restype = wintypes.BOOL

def _windows_path(path):
    path = os.path.abspath(os.fspath(path))
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path[2:]
    return "\\\\?\\" + path

def open_shared_fd(path):
    """Caller owns the returned fd and MUST close it."""
    if os.name != "nt":
        return os.open(os.fspath(path), os.O_RDONLY)
    handle = _k32.CreateFileW(
        _windows_path(path), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
    if handle == _INVALID:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # Ownership transfers to fd. Do not CloseHandle after success.
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        _k32.CloseHandle(handle)
        raise

@contextlib.contextmanager
def open_shared_binary(path):
    """Shared binary stream, with ownership/cleanup safe on fdopen failure."""
    fd = open_shared_fd(path)
    try:
        stream = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    with stream:
        yield stream

def _handle_identity(fd):
    if os.name != "nt":
        return file_identity(os.fstat(fd))
    # Native FILETIME avoids CRT fstat timestamp/inode differences on Windows.
    info = _FileInfo()
    handle = msvcrt.get_osfhandle(fd)
    if not _k32.GetFileInformationByHandle(handle, ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    def ns(value):
        ticks = (value.dwHighDateTime << 32) | value.dwLowDateTime
        return (ticks - 116444736000000000) * 100
    return (info.volume, (info.index_high << 32) | info.index_low,
            ns(info.write), ns(info.creation),
            (info.size_high << 32) | info.size_low)

def _matches_expected(actual, expected, windows=None):
    if windows is None:
        windows = os.name == "nt"
    if windows:
        # Compare common fields only. stat().st_dev/st_ctime meanings can differ
        # from Win32 volume/creation fields between Python/filesystem versions.
        return (actual[2], actual[-1]) == (expected[2], expected[-1])
    return actual == expected

def read_shared_bytes(path, max_bytes=32 * 1024 * 1024, expected_identity=None):
    """Return (bytes, directory-version token or native handle identity).

    Source closes before return. Read exactly the initial length, never follow
    a growing file indefinitely. Windows scan/open checks use mtime+size only:
    replacement with identical metadata can escape this heuristic.
    """
    if int(max_bytes) <= 0:
        raise ValueError("max_bytes must be > 0")
    fd = open_shared_fd(path)
    try:
        before = _handle_identity(fd)
        if expected_identity is not None and not _matches_expected(before, expected_identity):
            raise ImageNotReady("File changed between scan and open")
        size = before[-1]
        if size <= 0:
            raise ImageNotReady("Empty image")
        if size > max_bytes:
            raise ImageTooLarge("Image exceeds max_image_bytes")
        chunks = []
        remaining = size
        while remaining:
            chunk = os.read(fd, min(256 * 1024, remaining))
            if not chunk:
                raise ImageNotReady("Short read; producer changed the file")
            chunks.append(chunk)
            remaining -= len(chunk)
        if _handle_identity(fd) != before:
            raise ImageNotReady("Native handle metadata changed during read")
        # Preserve the scanner's token for deduplication; it is NOT a content hash.
        token = expected_identity if expected_identity is not None else before
        return b"".join(chunks), token
    finally:
        os.close(fd)
