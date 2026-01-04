import serial
import time

print("Opening DATA port...")
data_port = serial.Serial('/dev/tty.usbserial-00ED1D3E1', 921600, timeout=2)
time.sleep(2)

print("Checking for any data...")
for i in range(5):
    bytes_available = data_port.in_waiting
    print(f"Check {i+1}: {bytes_available} bytes available")
    
    if bytes_available > 0:
        data = data_port.read(min(bytes_available, 100))  # Read first 100 bytes
        print("Raw data (hex):", data.hex())
        print("As text:", data.decode('utf-8', errors='ignore'))
        break
    
    time.sleep(1)

data_port.close()
