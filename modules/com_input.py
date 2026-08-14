"""
Module: com_input
Chuc nang: Doc product code tu cong COM (tu PLC, barcode scanner, RFID reader,...)
    - 2 che do: manual (doc tu config) hoac auto (doc tu COM)
    - Polling thuong xuyen de cap nhat product code
    - Error handling: neu COM loi → dung default code, khong crash
    - Thread-safe: doc COM trong thread rieng

Khong phu thuoc module khac trong project.
Chi import: serial (pyserial), threading
"""

import time
import threading
from typing import Optional
from collections import deque


class COMProductReader:
    """
    Doc product code tu cong COM.
    
    Cach dung:
        # Che do AUTO (doc tu COM)
        reader = COMProductReader(
            port="COM4", 
            baudrate=9600, 
            mode="latest"
        )
        reader.start()
        
        # Trong vong lap chinh:
        product = reader.get_current()
        if product:
            print(f"Product: {product}")
        
        reader.stop()
    
    Mode:
        - "latest": Chi lay product code moi nhat (recommended)
        - "queue": Luu tat ca vao queue (it dung)
    """
    
    def __init__(self, 
                 port: str = "COM4",
                 baudrate: int = 9600,
                 timeout: float = 1.0,
                 mode: str = "latest",
                 poll_interval: float = 0.5):
        """
        Args:
            port: Ten cong COM (VD: "COM4")
            baudrate: Toc do baud (9600, 19200, ...)
            timeout: Timeout doc (giay)
            mode: "latest" hoac "queue"
            poll_interval: Thoi gian poll (giay)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.mode = mode
        self.poll_interval = poll_interval
        
        self.serial_port = None
        self.is_connected = False
        self.is_running = False
        
        # Du lieu
        self.current_product = None      # Product code moi nhat
        self.product_queue = deque(maxlen=100)  # Queue (neu mode=queue)
        
        # Thread
        self.read_thread = None
        self.lock = threading.Lock()
        
        # Thong ke
        self.total_received = 0
        self.total_errors = 0
        
        # Kiem tra pyserial
        try:
            import serial
            self.serial = serial
        except ImportError:
            print("[COM_INPUT] WARNING: pyserial not installed")
            print("[COM_INPUT] Install: pip install pyserial")
            print("[COM_INPUT] COM input disabled")
            return
    
    def _connect(self) -> bool:
        """Ket noi COM port"""
        try:
            self.serial_port = self.serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self.is_connected = True
            print(f"[COM_INPUT] Connected to {self.port} @ {self.baudrate} baud")
            return True
            
        except self.serial.SerialException as e:
            print(f"[COM_INPUT] WARNING: Cannot connect to {self.port}: {e}")
            print(f"[COM_INPUT] COM input disabled. Will use default product code.")
            self.is_connected = False
            return False
        except Exception as e:
            print(f"[COM_INPUT] ERROR: {e}")
            self.is_connected = False
            return False
    
    def _disconnect(self) -> None:
        """Ngat ket noi"""
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.is_connected = False
    
    def _read_loop(self) -> None:
        """Vong lap doc COM trong thread rieng"""
        while self.is_running:
            if not self.is_connected:
                # Thu ket noi lai
                if not self._connect():
                    time.sleep(5)  # Cho 5s roi thu lai
                    continue
            
            try:
                # Doc 1 dong
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline()
                    
                    # Decode + strip and take first 5 characters
                    try:
                        product_code = line.decode('ascii').strip()[:5]
                    except UnicodeDecodeError:
                        # Thu UTF-8
                        product_code = line.decode('utf-8', errors='ignore').strip()[:5]
                    
                    # Bo qua dong rong
                    if not product_code:
                        continue
                    
                    # Luu product code
                    with self.lock:
                        self.current_product = product_code
                        self.product_queue.append(product_code)
                        self.total_received += 1
                    
                    print(f"[COM_INPUT] Received: '{product_code}' (Total: {self.total_received})")
                
                # Cho poll_interval
                time.sleep(self.poll_interval)
                
            except self.serial.SerialException as e:
                print(f"[COM_INPUT] WARNING: Read error: {e}")
                self.total_errors += 1
                self._disconnect()
                time.sleep(2)  # Cho 2s roi thu ket noi lai
                
            except Exception as e:
                print(f"[COM_INPUT] ERROR: {e}")
                self.total_errors += 1
                time.sleep(1)
    
    def start(self) -> None:
        """Bat dau doc COM"""
        if self.is_running:
            print("[COM_INPUT] Already running")
            return
        
        # Kiem tra pyserial
        if not hasattr(self, 'serial'):
            print("[COM_INPUT] pyserial not available, cannot start")
            return
        
        # Ket noi
        if not self._connect():
            print("[COM_INPUT] Cannot connect, will retry in background")
        
        # Start thread
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        print(f"[COM_INPUT] Reader started (mode={self.mode})")
    
    def stop(self) -> None:
        """Dung doc COM"""
        self.is_running = False
        if self.read_thread:
            self.read_thread.join(timeout=2)
        self._disconnect()
        print(f"[COM_INPUT] Stopped. Stats: Received={self.total_received}, Errors={self.total_errors}")
    
    def get_current(self) -> Optional[str]:
        """
        Lay product code hien tai.
        
        Returns:
            Product code (str) hoac None neu chua nhan duoc
        """
        with self.lock:
            return self.current_product
    
    def get_queue(self) -> list:
        """Lay toan bo product queue (neu mode=queue)"""
        with self.lock:
            return list(self.product_queue)
    
    def clear(self) -> None:
        """Xoa product hien tai (de test)"""
        with self.lock:
            self.current_product = None
            self.product_queue.clear()
        print("[COM_INPUT] Cleared product data")
    
    def get_stats(self) -> dict:
        """Lay thong ke"""
        with self.lock:
            return {
                "current_product": self.current_product,
                "total_received": self.total_received,
                "total_errors": self.total_errors,
                "is_connected": self.is_connected,
                "is_running": self.is_running,
                "queue_size": len(self.product_queue)
            }
    
    def __del__(self):
        """Destructor"""
        try:
            self.stop()
        except Exception:
            pass


# ============================================================================
# TEST
# ============================================================================

def main():
    """Test COM input"""
    print("[TEST] COMProductReader\n")
    
    # Test 1: Khoi tao
    print("[1] Test khoi tao COM4...")
    reader = COMProductReader(
        port="COM999",  # Port khong ton tai (demo)
        baudrate=9600,
        mode="latest"
    )
    
    # Test 2: Start (se that bai neu khong co port)
    print("\n[2] Test start (se hien warning neu khong co port)...")
    reader.start()
    
    # Test 3: Gia lap nhan data
    print("\n[3] Gia lap nhan product code...")
    reader.current_product = "TEST_PRODUCT_123"
    reader.total_received = 1
    
    # Test 4: Lay product
    print("\n[4] Test get current product...")
    product = reader.get_current()
    print(f"  Current product: {product}")
    
    # Test 5: Stats
    print("\n[5] Stats:")
    stats = reader.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test 6: Stop
    print("\n[6] Test stop...")
    reader.stop()
    
    print("\n[DONE] Test completed!")
    print("\n[NOTE] De test that:")
    print("  1. Ket noi thiet bi vao COM port (VD: Arduino, PLC)")
    print("  2. Thiet bi gui text line, VD: 'ABC123\\r\\n'")
    print("  3. Chay lai test voi port dung")


if __name__ == "__main__":
    main()
