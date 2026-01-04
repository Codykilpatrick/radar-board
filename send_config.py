import serial
import time

# Open CLI port
cli_port = serial.Serial('/dev/tty.usbserial-00ED1D3E0', 115200)
time.sleep(1)

# Read and send config file
with open('profile_2d.cfg', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('%'):  # Skip comments and empty lines
            print(f"Sending: {line}")
            cli_port.write((line + '\n').encode())
            time.sleep(0.05)
            
            # Read response
            time.sleep(0.05)
            if cli_port.in_waiting > 0:
                response = cli_port.read(cli_port.in_waiting)
                print(f"Response: {response.decode('utf-8', errors='ignore')}")

cli_port.close()
print("\nConfig sent! Radar should be running now.")