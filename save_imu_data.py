import serial
import csv
import time

# --- CONFIGURATION ---
SERIAL_PORT = 'COM7'   # Check your Arduino IDE for the correct port (e.g., /dev/ttyUSB0 on Linux/Mac)
BAUD_RATE = 115200
OUTPUT_FILE = 'C:\\Users\\RoboticsEngineer\\Desktop\\Kalman Filters\\saveimu_data.csv'
DURATION = 10          # How many seconds to record

def log_data():
    # Open Serial Port
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}")
    except serial.SerialException:
        print(f"Error: Could not open {SERIAL_PORT}. Is the Arduino Serial Monitor open?")
        return

    # Open CSV File
    with open(OUTPUT_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write Header
        writer.writerow(['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        
        print(f"Recording data for {DURATION} seconds...")
        start_time = time.time()
        line_count = 0

        # Clear buffer to avoid old data
        ser.reset_input_buffer()

        while (time.time() - start_time) < DURATION:
            if ser.in_waiting > 0:
                try:
                    # Read line, decode bytes to string, strip whitespace
                    line = ser.readline().decode('utf-8').strip()
                    
                    # Split by comma
                    data = line.split(',')
                    
                    # Ensure we have 6 values
                    if len(data) == 6:
                        writer.writerow(data)
                        line_count += 1
                        if line_count % 10 == 0:
                            print(f"Logged {line_count} lines...", end='\r')
                            
                except ValueError:
                    continue # Skip bad lines

    print(f"\nDone! Saved {line_count} data points to {OUTPUT_FILE}")
    ser.close()

if __name__ == "__main__":
    log_data()