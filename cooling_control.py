import RPi.GPIO as GPIO
import psutil
import json
import os

FAN1_SPEED_PIN = 26
FAN2_POWER_PIN = 20

def set_compute(data):
    with open("/home/dom/compute_data.json", "w") as file:
        json.dump(data, file, indent=4)

def set_pin(pin, value):
    val_map = {1:GPIO.HIGH,0:GPIO.LOW}
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, val_map[value])


def turn_off_high_cooling():
    # fan1 on 3.3 v
    # fan2 off
    set_pin(FAN1_SPEED_PIN, 0)
    set_pin(FAN2_POWER_PIN, 0)


def turn_on_high_cooling():
    # fan1 on 3.3 v
    # fan2 on 5v
    set_pin(FAN1_SPEED_PIN, 0)
    set_pin(FAN2_POWER_PIN, 1)


def turn_on_ultra_high_cooling():
    # fan1 on 5v
    # fan2 on 5v
    set_pin(FAN1_SPEED_PIN, 1)
    set_pin(FAN2_POWER_PIN, 1)



last_written = False
while True:
    cpu_usage = psutil.cpu_percent(interval=1)
    if cpu_usage >= 30.0:
        if last_written == True:
            pass
        else:
            print("adding...")
            set_compute({})
            last_written = True

    else:
        if last_written == False:
            pass
        else:
            last_written = False
            print("removing...")
            if os.path.exists("/home/dom/compute_data.json"):
                os.remove("/home/dom/compute_data.json")
    
    if cpu_usage > 75.0:
        print(f"System CPU Usage: {cpu_usage}%", end="\r")
        turn_on_high_cooling()
    if cpu_usage > 92.0:
        turn_on_ultra_high_cooling()
    if cpu_usage <= 75.0:
        turn_off_high_cooling()