import serial
import struct
import time

# Magic word to identify frame start
MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

def parse_header(data):
    """Parse the frame header (40 bytes)"""
    if len(data) < 40:
        return None
    
    # Correct format: 10 uint32s = 40 bytes
    header = struct.unpack('<10I', data[:40])
    return {
        'magic': header[0:2],
        'version': header[2],
        'totalPacketLen': header[3],
        'platform': header[4],
        'frameNumber': header[5],
        'timeCpuCycles': header[6],
        'numDetectedObj': header[7],
        'numTLVs': header[8],
        'subFrameNumber': header[9]
    }

def parse_detected_objects(data, num_objects):
    """Parse detected objects TLV"""
    objects = []
    offset = 0
    
    for i in range(num_objects):
        if offset + 16 > len(data):
            break
            
        obj = struct.unpack('<4f', data[offset:offset+16])
        objects.append({
            'range': obj[0],      # meters
            'azimuth': obj[1],    # radians
            'doppler': obj[2],    # m/s
            'snr': obj[3]         # dB
        })
        offset += 16
    
    return objects

# Open data port
data_port = serial.Serial('/dev/tty.usbserial-00ED1D3E1', 921600, timeout=1)

print("Listening for radar data... (Press Ctrl+C to stop)")
print("Wave your hand in front of the radar!\n")

buffer = bytes()

try:
    while True:
        # Read available data
        if data_port.in_waiting > 0:
            buffer += data_port.read(data_port.in_waiting)
        
        # Look for magic word
        magic_idx = buffer.find(MAGIC_WORD)
        if magic_idx >= 0:
            # Parse header
            header_data = buffer[magic_idx:magic_idx+40]
            header = parse_header(header_data)
            
            if header and header['totalPacketLen'] < 10000:  # Sanity check
                packet_len = header['totalPacketLen']
                
                # Wait for full packet
                while len(buffer) < magic_idx + packet_len:
                    if data_port.in_waiting > 0:
                        buffer += data_port.read(data_port.in_waiting)
                    time.sleep(0.01)
                
                # Extract full frame
                frame = buffer[magic_idx:magic_idx + packet_len]
                
                print(f"\n--- Frame {header['frameNumber']} ---")
                print(f"Detected objects: {header['numDetectedObj']}")
                
                if header['numDetectedObj'] > 0:
                    # Skip header and TLV header to get to object data
                    obj_data = frame[48:]  # 40 byte header + 8 byte TLV header
                    objects = parse_detected_objects(obj_data, header['numDetectedObj'])
                    
                    for i, obj in enumerate(objects):
                        print(f"  Object {i+1}: Range={obj['range']:.2f}m, "
                              f"Azimuth={obj['azimuth']*57.3:.1f}°, "
                              f"Velocity={obj['doppler']:.2f}m/s, "
                              f"SNR={obj['snr']:.1f}dB")
                
                # Remove processed frame from buffer
                buffer = buffer[magic_idx + packet_len:]
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\nStopped.")
    data_port.close()
