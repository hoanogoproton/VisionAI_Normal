"""
Module: com_output
Chuc nang: Gui tin hieu OK/NG qua cong COM den thiet bi ngoai (PLC, Arduino,...)
    - Retry mechanism: Thu lai neu that bai
    - Error handling: Log warning nhung khong crash he thong
    - Don gian, de debug

Khong phu thuoc module khac trong project.
Chi import: serial (pyserial)
"""

import time
from typing import Optional


class COMOutput:
    """
    Quan ly cong COM de gui tin hieu ket qua.
    
    Cach dung:
        com = COMOutput(port="COM3", baudrate=9600, enabled=True)
        
        # Gui ket qua
        com.send_result("OK")    # Gui "OK\r\n"
        com.send_result("NG")    # Gui "NG\r\n"
        
        # Dong port
        com.close()
    """
    
    def __init__(self, 
                 port: str = "COM3",
                 baudrate: int = 9600,
                 timeout: float = 1.0,
                 enabled: bool = True,
                 retry_count: int = 3,
                 retry_delay: float = 0.5):
        """
        Args:
            port: Ten cong COM (VD: "COM3", "COM5")
            baudrate: Toc do truyen (9600, 19200, 115200,...)
            timeout: Timeout cho tung thao tac (giay)
            enabled: Bat/tat chuc nang (de debug)
            retry_count: So lan thu lai neu that bai
            retry_delay: Thoi gian cho giua cac lan thu (giay)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.enabled = enabled
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        self.serial_port = None
        self.is_connected = False
        
        # Thong ke
        self.total_sent = 0
        self.total_failed = 0
        
        # Kiem tra pyserial
        try:
            import serial
            self.serial = serial
        except ImportError:
            print("[COM] WARNING: pyserial not installed. COM output disabled.")
            print("[COM] Install: pip install pyserial")
            self.enabled = False
            return
        
        # Ket noi port
        if self.enabled:
            self._connect()
    
    def _connect(self) -> bool:
        """
        Ket noi den cong COM.
        
        Returns:
            True neu thanh cong, False neu that bai
        """
        if not self.enabled:
            return False
        
        try:
            self.serial_port = self.serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            self.is_connected = True
            print(f"[COM] Connected to {self.port} @ {self.baudrate} baud")
            return True
            
        except self.serial.SerialException as e:
            print(f"[COM] WARNING: Cannot connect to {self.port}: {e}")
            print(f"[COM] COM output disabled. System will continue without COM.")
            self.is_connected = False
            self.enabled = False  # Tu dong tat neu khong ket noi duoc
            return False
        except Exception as e:
            print(f"[COM] ERROR: Unexpected error: {e}")
            self.is_connected = False
            self.enabled = False
            return False
    
    def send_result(self, status: str, extra_info: str = "") -> bool:
        """
        Gui ket qua qua COM.
        
        Args:
            status: "OK" hoac "NG"
            extra_info: Thong tin them (tuy chon)
            
        Returns:
            True neu gui thanh cong, False neu that bai
            
        Format gui:
            - Don gian: "OK\r\n" hoac "NG\r\n"
            - Co extra: "OK,BATCH5\r\n"
        """
        if not self.enabled:
            return True  # Khong bao loi neu da tat
        
        if not self.is_connected:
            # Thu ket noi lai 1 lan
            if not self._connect():
                return False
        
        # Tao message
        if extra_info:
            message = f"{status},{extra_info}\r\n"
        else:
            message = f"{status}\r\n"
        
        # Thu gui voi retry
        for attempt in range(1, self.retry_count + 1):
            try:
                self.serial_port.write(message.encode('ascii'))
                self.serial_port.flush()  # Dam bao du lieu duoc gui ngay
                
                self.total_sent += 1
                
                if attempt > 1:
                    print(f"[COM] Sent (retry {attempt-1}): {message.strip()}")
                else:
                    print(f"[COM] Sent: {message.strip()}")
                
                return True
                
            except self.serial.SerialException as e:
                print(f"[COM] WARNING: Send failed (attempt {attempt}/{self.retry_count}): {e}")
                
                # Ngat ket noi va thu ket noi lai
                self._disconnect()
                
                if attempt < self.retry_count:
                    time.sleep(self.retry_delay)
                    self._connect()
                else:
                    # Het retry
                    print(f"[COM] ERROR: Failed to send after {self.retry_count} attempts")
                    self.total_failed += 1
                    return False
                    
            except Exception as e:
                print(f"[COM] ERROR: Unexpected error: {e}")
                self.total_failed += 1
                return False
        
        return False
    
    def send_ok(self, extra_info: str = "") -> bool:
        """Gui tin hieu OK"""
        return self.send_result("OK", extra_info)
    
    def send_ng(self, extra_info: str = "") -> bool:
        """Gui tin hieu NG"""
        return self.send_result("NG", extra_info)
    
    def _disconnect(self) -> None:
        """Ngat ket noi COM port"""
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except Exception:
                pass
        self.is_connected = False
    
    def close(self) -> None:
        """Dong cong COM"""
        if self.serial_port:
            self._disconnect()
            print(f"[COM] Port closed. Stats: Sent={self.total_sent}, Failed={self.total_failed}")
    
    def get_stats(self) -> dict:
        """Lay thong ke"""
        return {
            "total_sent": self.total_sent,
            "total_failed": self.total_failed,
            "is_connected": self.is_connected,
            "enabled": self.enabled
        }
    
    def __del__(self):
        """Destructor: tu dong dong port"""
        try:
            self.close()
        except Exception:
            pass


# ============================================================================
# TEST
# ============================================================================

def main():
    """Test don gian"""
    print("[TEST] COMOutput\n")
    
    # Test 1: Khoi tao (se that bai neu khong co port)
    print("[1] Test khoi tao COM3...")
    com = COMOutput(port="COM999", baudrate=9600, enabled=True)
    # → Se hien warning nhung khong crash
    
    print("\n[2] Test gui tin hieu (se bi bo qua neu khong ket noi)...")
    com.send_ok()
    com.send_ng()
    com.send_result("OK", "BATCH5")
    
    print("\n[3] Stats:")
    stats = com.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n[4] Test tat COM (enabled=False)...")
    com2 = COMOutput(port="COM3", enabled=False)
    com2.send_ok()  # Khong gui gi, khong bao loi
    
    com.close()
    com2.close()
    
    print("\n[DONE] Test completed!")
    print("[NOTE] Neu muon test that, doi port thanh COM port that tren may.")


if __name__ == "__main__":
    main()