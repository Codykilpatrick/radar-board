import serial
import struct
import time

MAGIC_WORD = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

data_port = serial.Serial('/dev/tty.usbserial-00ED1D3E1', 921600, timeout=1)

print("Reading frames to analyze IWR6843ISK data format...\n")

buffer = bytes()
frames_read = 0

try:
    start = time.time()
    while frames_read < 10 and (time.time() - start) < 20:
        if data_port.in_waiting > 0:
            buffer += data_port.read(data_port.in_waiting)
        
        magic_idx = buffer.find(MAGIC_WORD)
        if magic_idx >= 0:
            header_data = buffer[magic_idx:magic_idx+40]
            if len(header_data) >= 40:
                header = struct.unpack('<10I', header_data)
                packet_len = header[3]
                num_det = header[7]
                num_tlvs = header[8]
                
                # Wait for full packet
                timeout_t = time.time() + 2
                while len(buffer) < magic_idx + packet_len and time.time() < timeout_t:
                    if data_port.in_waiting > 0:
                        buffer += data_port.read(data_port.in_waiting)
                    time.sleep(0.01)
                
                if len(buffer) >= magic_idx + packet_len:
                    frame = buffer[magic_idx:magic_idx + packet_len]
                    
                    # Only show frames with detections
                    if num_det > 0:
                        frames_read += 1
                        print(f"=== Frame {frames_read} (Detected: {num_det}) ===")
                        print(f"Packet length: {packet_len}, Num TLVs: {num_tlvs}")
                        
                        # Parse TLVs properly
                        offset = 40  # After header
                        for tlv_idx in range(num_tlvs):
                            if offset + 8 > len(frame):
                                break
                            
                            tlv_type, tlv_len = struct.unpack('<2I', frame[offset:offset+8])
                            
                            if tlv_type == 1:  # Detected points
                                print(f"\nTLV Type 1 (Detected Points): Length={tlv_len}")
                                tlv_payload = frame[offset+8:offset+tlv_len]
                                print(f"Payload size: {len(tlv_payload)} bytes")
                                
                                # Show raw bytes
                                print(f"Raw: {tlv_payload.hex()}")
                                
                                # Try different interpretations
                                if len(tlv_payload) >= 4:
                                    # Check if there's a descriptor first
                                    desc = struct.unpack('<HH', tlv_payload[:4])
                                    print(f"\nFirst 4 bytes as 2 uint16: {desc}")
                                    
                                # The IWR6843 SDK sometimes uses:
                                # - Point cloud format with x,y,z,doppler as int16 Q-format
                                # - Or with a descriptor header first
                                
                                # Try parsing as spherical (range, azimuth, elev, doppler) - compressed int16
                                if len(tlv_payload) >= 8:
                                    print("\nAs 4 int16 (maybe Q-format or indices):")
                                    for i in range(min(num_det, 3)):
                                        obj_start = i * 8
                                        if obj_start + 8 <= len(tlv_payload):
                                            vals = struct.unpack('<4h', tlv_payload[obj_start:obj_start+8])
                                            print(f"  Obj {i}: {vals}")
                                
                                # Try as 2 floats 
                                if len(tlv_payload) >= 8:
                                    print("\nAs 2 floats:")
                                    for i in range(min(num_det, 3)):
                                        obj_start = i * 8
                                        if obj_start + 8 <= len(tlv_payload):
                                            vals = struct.unpack('<2f', tlv_payload[obj_start:obj_start+8])
                                            print(f"  Obj {i}: {vals[0]:.4f}, {vals[1]:.4f}")
                                
                                # Try with descriptor (first 4 bytes = xyzQFormat, dopplerQFormat)
                                # Then data starts at byte 4
                                if len(tlv_payload) >= 12:
                                    print("\nWith 4-byte descriptor, then 4 int16 per object:")
                                    desc_raw = tlv_payload[:4]
                                    # Descriptor might be: xyzQFormat (uint16), dopplerQFormat (uint16)
                                    xyz_q, doppler_q = struct.unpack('<HH', desc_raw)
                                    print(f"  Descriptor: xyzQ={xyz_q}, dopplerQ={doppler_q}")
                                    
                                    data_part = tlv_payload[4:]
                                    for i in range(min(num_det, 3)):
                                        obj_start = i * 8
                                        if obj_start + 8 <= len(data_part):
                                            vals = struct.unpack('<4h', data_part[obj_start:obj_start+8])
                                            # Convert from Q-format
                                            x = vals[0] / (1 << xyz_q) if xyz_q < 16 else vals[0]
                                            y = vals[1] / (1 << xyz_q) if xyz_q < 16 else vals[1]
                                            z = vals[2] / (1 << xyz_q) if xyz_q < 16 else vals[2]
                                            doppler = vals[3] / (1 << doppler_q) if doppler_q < 16 else vals[3]
                                            print(f"  Obj {i} raw: {vals}")
                                            print(f"  Obj {i} converted: x={x:.3f}m, y={y:.3f}m, z={z:.3f}m, doppler={doppler:.3f}m/s")
                            
                            elif tlv_type == 7:  # Side info (SNR, noise)
                                print(f"\nTLV Type 7 (Side Info): Length={tlv_len}")
                                tlv_payload = frame[offset+8:offset+tlv_len]
                                if len(tlv_payload) >= 4:
                                    for i in range(min(num_det, 3)):
                                        obj_start = i * 4
                                        if obj_start + 4 <= len(tlv_payload):
                                            snr, noise = struct.unpack('<2H', tlv_payload[obj_start:obj_start+4])
                                            print(f"  Obj {i}: SNR={snr/10:.1f}dB, Noise={noise/10:.1f}dB")
                            
                            offset += tlv_len if tlv_len > 8 else 8
                        
                        print("\n" + "="*60 + "\n")
                    
                    buffer = buffer[magic_idx + packet_len:]
        
        time.sleep(0.01)

except KeyboardInterrupt:
    pass
finally:
    data_port.close()
    print("Done!")
