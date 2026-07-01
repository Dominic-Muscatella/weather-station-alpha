import subprocess
import json
import csv
import time
import os
import time
import serial
import uuid


CSV_FILE = "/mnt/DeepData/live_LOCAL.csv"
INTERVAL_SECONDS = 285  
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 4800 
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
print(f"Connected to {SERIAL_PORT}")

def write_to_sensor_lock_file(sensor):
    try:
        with open(f"sensor{sensor}_rcv.lock", "w", encoding="utf-8") as file:
            
            file.write(str(uuid.uuid4()))
    except Exception as E:
        print(E)
        print('continuing anyways...')


def write_to_data_lock_file():
    try:
        with open(f"data_wrt.lock", "w", encoding="utf-8") as file:
            
            file.write(str(uuid.uuid4()))
    except Exception as E:
        print(E)
        print('continuing anyways...')


def celsius_to_fahrenheit(celsius):
    
    return (celsius * 9 / 5) + 32

def read_baro_pressure():
    try:
        
        if ser.in_waiting > 0:
            
            for _ in range(10): 
                if ser.in_waiting == 0:
                    break
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line.startswith('$') and 'XDR' in line:
                    parts = line.split(',')
                    try:
                        pressure_bars = float(parts[2])
                        return pressure_bars * 1000 
                    except (ValueError, IndexError):
                        pass
    except serial.SerialException as e:
        print(f"Serial error: {e}. Attempting reconnect...")
        try:
            ser.close()
            time.sleep(2)
            ser.open()
        except Exception as re:
            print(f"Reconnect failed: {re}")
    
    
    return None



def initialize_csv():
    
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["valid", "tmpf", "relh", "pressure_hpa"])

def write_to_csv(model, sensor_id, date_list, temp_list, hum_list, baro_list):
    
    avg_temp = round(sum(temp_list) / len(temp_list), 2)
    avg_hum = round(sum(hum_list) / len(hum_list), 2) if hum_list else "N/A"
    avg_baro = round(sum(baro_list) / len(baro_list), 2) if baro_list else "N/A"
    timestamp = date_list[-1]
    
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, avg_temp, avg_hum, avg_baro + 10])  
        
        
    write_to_data_lock_file()
    print(f"[{timestamp}] Saved 5-min average for ID {sensor_id}: Temp={avg_temp}°F, Hum={avg_hum}%, Baro={avg_baro+10}hPa ({len(temp_list)} readings)")

def listen_and_average():
    initialize_csv()
    
    
    cmd = ['/home/dom/rtl_433/build/src/rtl_433','-R','288', '-F', 'json']
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )

    
    
    buffer = {}
    
    
    start_time = time.time()
    print("Listening for sensors. Aggregating data every 5 minutes... Press Ctrl+C to stop.")

    try:
        while True:
            line = process.stdout.readline()
            if not line:
                continue
                
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    
                    if int(data["channel"]) == 1 or int(data["channel"]) == 2:
                    
                        print()
                        print('==============================================')
                        formatted_date = time.strftime('%Y-%m-%d %H:%M')
                        print(formatted_date)
                        model = data.get('model')
                        sensor_id = data.get('id')
                        temp = data.get('temperature_C')
                        temp = celsius_to_fahrenheit(temp)
                        humidity = data.get('humidity')
                        write_to_sensor_lock_file(data['channel'])
                        print(f"data logged from channel{int(data['channel'])}->t:{temp},h:{humidity}")
                        pressure = read_baro_pressure()
                        if pressure is not None:
                            print(f"Current Pressure: {pressure:.2f} mb (hPa)")
                        else:
                            print("Current Pressure: [No new serial data]")
                        print('==============================================')
                    
                    

                        
                        if model is None or sensor_id is None or temp is None:
                            continue

                        
                        sensor_key = (model, sensor_id)
                        if sensor_key not in buffer:
                            buffer[sensor_key] = {'temp': [], 'hum': [], 'baro':[], 'timestamp':[]}
                        if temp is not None:
                            buffer[sensor_key]['temp'].append(float(temp))
                        if humidity is not None:
                            buffer[sensor_key]['hum'].append(float(humidity))
                        if pressure is not None:
                            buffer[sensor_key]['baro'].append(float(pressure))
                        if formatted_date is not None:
                            buffer[sensor_key]['timestamp'].append(str(formatted_date))

                except (json.JSONDecodeError, ValueError) as e:
                    print('error:', e)
                     
            
            
            current_time = time.time()
            if current_time - start_time >= INTERVAL_SECONDS:
                
                acc_readings = {'timestamp':None, 
                                'temp':[],
                                'hum':[],
                                'baro':[]}
                sensor_id = 1
                for (model, sensor_id), readings in buffer.items():
                    if readings['temp']: 
                        acc_readings['timestamp'] = readings['timestamp']
                        acc_readings['temp'].extend(readings['temp'])
                        acc_readings['hum'].extend(readings['hum'])
                        acc_readings['baro'].extend(readings['baro'])
                    
                        
                write_to_csv(model, sensor_id, readings['timestamp'], acc_readings['temp'], acc_readings['hum'], acc_readings['baro'])
                
                
                buffer.clear()
                start_time = current_time

    except KeyboardInterrupt:
        print("\nStopping listener...")
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    listen_and_average()