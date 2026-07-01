import time
import board
import neopixel
import json
import urllib.request
import psutil
import signal
import sys
import os
import uuid
import copy
import gc
from pathlib import Path
import config as C
from system_management.sys_reporting import (write_to_data_alt_lock_file,
                                             get_file_write_time,
                                             write_to_sensor_alt_lock_file)
BRIGHTNESS = 0.0375


def alt_blink(pixels, blinks=1, delay=0.004, start=True):
    time.sleep(delay*20.0)
    old_brightness = float(pixels.brightness)
    old_pixels = [[int(a) for a in pixel] for pixel in pixels]

    for blink in range(blinks):
        pixels.brightness = BRIGHTNESS * 0.3333333
        for p in range(7):
            pixels[p] = old_pixels[p]
        pixels.show()
        time.sleep(delay*20.0)

        pixels.fill((0,0,0))
        pixels.show()
        time.sleep(0.0075)

        pixels.brightness = BRIGHTNESS * 2.0
        pixels[6] = C.BLINK_COLOR
        pixels.show()
        time.sleep(delay*10.0)

        pixels.brightness = BRIGHTNESS * 0.5
        pixels.fill(C.BLINK_COLOR)
        pixels.show()
        time.sleep(delay*15.0)

        pixels.fill((0,0,0))
        pixels.show()
        time.sleep(0.075)
    
    pixels.brightness = BRIGHTNESS * 0.3333333
    for p in range(7):
        pixels[p] = old_pixels[p]
    pixels.show()
    time.sleep(delay*5.0)
    pixels.brightness = old_brightness
    for p in range(7):
        pixels[p] = old_pixels[p]
    pixels.show()


def check_data_files_and_blink(pixels):
    write_to_data_alt_lock_file(tPath="now.lock")
    sensor1recieved_time = get_file_write_time("sensor1_rcv.lock")
    sensor1alert_time = get_file_write_time("sensor1_alt.lock")
    now_stamp = get_file_write_time("now.lock")
    present = [bool(sensor1recieved_time), bool(sensor1alert_time)]
    older_than_29s = now_stamp > sensor1alert_time + 5.0
    if sensor1recieved_time > sensor1alert_time and all(present) and older_than_29s:
        alt_blink(pixels)
        time.sleep(0.051) # helps prevent temporal collisions
        write_to_sensor_alt_lock_file(1)
        # time.sleep(0.01) # helps prevent temporal collisions
    # print(now_stamp, sensor1alert_time, sensor1recieved_time)
    sensor2recieved_time = get_file_write_time("sensor2_rcv.lock")
    sensor2alert_time = get_file_write_time("sensor2_alt.lock")
   
    present = [bool(sensor2recieved_time), bool(sensor2alert_time)]
    # write_to_data_alt_lock_file(tPath="now.lock")
    # now_stamp = get_file_write_time("now.lock")
    older_than_29s = now_stamp > sensor2alert_time + 5.0
    if sensor2recieved_time > sensor2alert_time and all(present) and older_than_29s:
        alt_blink(pixels, blinks=2)
        time.sleep(0.051) # helps prevent temporal collisions
        write_to_sensor_alt_lock_file(2)
        # time.sleep(0.01) # helps prevent temporal collisions

    datawrite_time = get_file_write_time("data_wrt.lock")
    dataalert_time = get_file_write_time("data_alt.lock")
    present = [bool(datawrite_time), bool(dataalert_time)]
    # write_to_data_alt_lock_file(tPath="now.lock")
    # now_stamp = get_file_write_time("now.lock")
    older_than_29s = now_stamp > dataalert_time + 5.0
    
    if datawrite_time > dataalert_time and all(present) and older_than_29s:
        alt_blink(pixels, blinks=5)
        time.sleep(0.051) # helps prevent temporal collisions
        write_to_data_alt_lock_file()
       
    ml_start_time = get_file_write_time("ml_str.lock")
    ml_alert_time = get_file_write_time("ml_alt.lock")
    present = [bool(ml_start_time), bool(ml_alert_time)]
    # write_to_data_alt_lock_file(tPath="now.lock")
    # now_stamp = get_file_write_time("now.lock")
    older_than_29s = now_stamp > ml_alert_time + 5.0
    if ml_start_time > ml_alert_time and all(present) and older_than_29s:
        alt_blink(pixels, blinks=16, delay=0.001, start=False)
        time.sleep(0.051) # helps prevent temporal collisions
        write_to_data_alt_lock_file(tPath="ml_alt.lock")
    
    monte_done_time = get_file_write_time("monte_fin.lock")
    monte_alert_time = get_file_write_time("monte_alt.lock")
    present = [bool(monte_done_time), bool(monte_alert_time)]
    # write_to_data_alt_lock_file(tPath="now.lock")
    # now_stamp = get_file_write_time("now.lock")
    older_than_29s = now_stamp > monte_alert_time + 5.0
    if monte_done_time > monte_alert_time and all(present) and older_than_29s:
        alt_blink(pixels, blinks=8, delay=0.001, start=False)
        time.sleep(0.051) # helps prevent temporal collisions
        write_to_data_alt_lock_file(tPath="monte_alt.lock")
    
    knn_done_time = get_file_write_time("knn_fin.lock")
    knn_alert_time = get_file_write_time("knn_alt.lock")
    present = [bool(knn_done_time), bool(knn_alert_time)]
    # write_to_data_alt_lock_file(tPath="now.lock")
    # now_stamp = get_file_write_time("now.lock")
    older_than_29s = now_stamp > knn_alert_time + 5.0
    if knn_done_time > knn_alert_time and all(present) and older_than_29s:
        alt_blink(pixels, blinks=16, delay=0.001, start=False)
        time.sleep(0.051) # helps prevent temporal collisions
        write_to_data_alt_lock_file(tPath="knn_alt.lock")
        


LED_PIN = board.D13

def knn_v1(rec: dict) -> dict | None:
    # If "knn" key is missing, or is empty/falsy, return None
    knn_data = rec.get("knn")
    if not knn_data:
        return None
        
    # Get the very first key in the dictionary
    kk = 'k60'
    
    # Extract the nested data structure
    target_node = knn_data[kk]
    
    return {
        "k": kk,
        "v1": target_node.get("v1_distance_weighted", {}),
        "conf": target_node.get("v1_confidence")
    }

def clamp(val, minimum, maximum):
    return max(minimum, min(val, maximum))


def blended(model: dict, none_prob: float, knn: dict, W: float, meta: dict) -> list:
    # Safely extract knn values using dict.get()
    kv = knn.get("v1") if knn else None
    knn_conf = knn.get("conf") if knn else None
    
    # Safely extract none_label from meta and check its value in kv
    none_label = meta["none_label"]
    kn = kv.get(none_label, 0) if kv else 0

    out = []
    
    for c in meta["hazard_classes"]:
        # model x none
        rr = (model[c]["prob"] / none_prob) if none_prob > 0 else 0
        
        # knn x none (capped at 5)
        kr = 0
        if kv:
            kr = min(kv.get(c, 0) / max(kn, 0.02), 5)
            
        # no knn -> model only
        w = W if kv else 0
        br = (1 - w) * rr + w * kr
        
        # confidence-discounted lower bound
        p = model[c]["prob"]
        lo = model[c].get("lo")
        
        if p > 0 and lo is not None:
            model_conf = 1 - clamp((p - lo) / p, 0, 1)
        else:
            model_conf = 0.5
            
        joint_conf = (model_conf * knn_conf) if knn_conf is not None else model_conf
        blo = br * joint_conf
        
        # Collect metric data structure
        out.append({
            "c": c,
            "prob": model[c]["prob"],
            "lo": model[c].get("lo"),
            "hi": model[c].get("hi"),
            "rr": rr,
            "kr": kr,
            "br": br,
            "modelConf": model_conf,
            "knnConf": knn_conf,
            "jointConf": joint_conf,
            "blo": blo
        })
        
    # Sort descending by the 'br' score
    out.sort(key=lambda x: x["br"], reverse=True)
    
    return out


def class_progress(br: float, t: dict) -> float:
    if not t or br <= 0:
        return 0.0

    w = t["watch"]
    r = t["warn"]
    
    # JavaScript: (t.advisory != null ? t.advisory : (t.watch + t.warn) / 2)
    if t.get("advisory") is not None:
        a = t["advisory"]
    else:
        a = (w + r) / 2

    if br < w:
        return (1 / 3) * (br / max(w, 1e-6))
        
    if br < a:
        return 1 / 3 + (1 / 3) * ((br - w) / max(a - w, 1e-6))
        
    if br < r:
        return 2 / 3 + (1 / 3) * ((br - a) / max(r - a, 1e-6))
        
    return 1.0 + min((br - r) / max(r, 0.2), 0.3)


def horizon_row(
    horizon_key: str,
    blend_rows: list,
    th: dict,
    knn: dict,
):
    # Extract helper functions passed via a dictionary (or use global scope)


    knn_conf = knn["conf"] if knn else None
    lit = []
    peak = 0

    watches =0
    advisories = 0
    warnings = 0
    for r in blend_rows:
        # Safely drill down into th dictionary
        t_dict = th.get(r["c"])
        t = t_dict.get(horizon_key) if t_dict else None
        if not t:
            continue

        # JavaScript: (t.advisory != null ? t.advisory : (t.watch + t.warn) / 2)
        if t.get("advisory") is not None:
            adv = t["advisory"]
        else:
            adv = (t["watch"] + t["warn"]) / 2

        # Gauge progress on the MEAN
        p = class_progress(r["br"], t)
        if p > peak:
            peak = p
            headline = r

        # Upper bound calculation
        bup = r["br"] * (2 - r["jointConf"])

        if r["blo"] >= t["warn"]:
            warnings +=1
        elif r["br"] >= adv:
            advisories +=1
        elif bup >= t["watch"]:
            watches +=1
    return watches, advisories, warnings

def fetch_model_meta_thresh():
    url = "http://weather-station-alpha.local:8000/api/thresholds"

    # Perform the HTTP GET call and parse the response directly
    with urllib.request.urlopen(url) as response:
        thresh_data = json.load(response)

    # print(thresh_data)


    url = "http://weather-station-alpha.local:8000/api/latest?loc=LOCAL"

    # Perform the HTTP GET call and parse the response directly
    with urllib.request.urlopen(url) as response:
        model_data = json.load(response)

    # print(model_data)

    url = "http://weather-station-alpha.local:8000/api/meta"

    # Perform the HTTP GET call and parse the response directly
    with urllib.request.urlopen(url) as response:
        meta_data = json.load(response)

    # print(meta_data)

    return model_data, meta_data, thresh_data


def load_and_interpolate(file):
    f_data = json.load(file)
    old_seq = f_data["sequence"]
    N = len(old_seq)
    new_seq = []

    for i in range(N):
        # 1. Add current frame to the even position
        current_frame = old_seq[i]
        new_seq.append(current_frame)
        
        # 2. Calculate the interpolated frame for the odd position
        if i < N - 1:
        # Standard interpolation between current frame and next frame
            next_frame = old_seq[i + 1]
            interp_time = (current_frame["time"] + next_frame["time"]) / 2
        else:
            # Final slot loop: Interpolate between the LAST frame and the FIRST frame
            next_frame = old_seq[0]
            # Extrapolate the time forward so it doesn't jump backwards mid-sequence
            interp_time = current_frame["time"] 
        interp_pixels = []
        for p1, p2 in zip(current_frame["pixels"], next_frame["pixels"]):
            rgb = [int((a + b) / 2) for a, b in zip(p1, p2)]
            interp_pixels.append(rgb)
            
        interp_frame = {
            "time": round(interp_time, 4),  # Rounded to prevent floating-point math issues
            "pixels": interp_pixels
        }

        new_seq.append(interp_frame)

    # Create the final matching output structure
    output_data = {"sequence": new_seq}
    return output_data


def check_compute():
    try:
        path = Path("/home/dom/compute_data.json")
        return path.exists()
    
    except Exception as e:
        print(e)
        print("resuming loop....")
        return False

with open(os.path.join(C.SYS_LOCK_DIR, 'system_management', 'light_animations', 'clear_light.json'), 'r') as file:
    clear_seq = load_and_interpolate(file)

with open(os.path.join(C.SYS_LOCK_DIR, 'system_management', 'light_animations', 'watch_light.json'), 'r') as file:
    watch_seq = load_and_interpolate(file)

with open(os.path.join(C.SYS_LOCK_DIR, 'system_management', 'light_animations', 'advisory_light.json'), 'r') as file:
    advisory_seq = load_and_interpolate(file)

with open(os.path.join(C.SYS_LOCK_DIR, 'system_management', 'light_animations', 'warning_light.json'), 'r') as file:
    warning_seq = load_and_interpolate(file)

with open(os.path.join(C.SYS_LOCK_DIR, 'system_management', 'light_animations', 'compute_running_light.json'), 'r') as file:
    compute_seq = load_and_interpolate(file)

animations = {"clear": clear_seq,
              "watch": watch_seq,
              "advisory": advisory_seq,
              "warning": warning_seq}
NUM_PIXELS = 7

# Color byte order (WS2812B usually uses GRB format)
ORDER = neopixel.GRB

# Initialize the LED strip object
pixels = neopixel.NeoPixel(
    LED_PIN, 
    NUM_PIXELS, 
    brightness=0.1,      # Global brightness control (0.0 to 1.0)
    auto_write=True,    # Set False to buffer changes, improving speed
    pixel_order=ORDER
)

def cleanup_and_exit(*args, **kwargs):
    pixels.fill(( 255, 0,0))
    pixels.show()
    time.sleep(0.25) 
    pixels.fill((0, 0, 0))
    pixels.show()
    time.sleep(0.25) 
    pixels.fill(( 255, 0,0))
    pixels.show()
    time.sleep(0.25) 
    pixels.fill((0, 0, 0))
    pixels.show()
    time.sleep(0.25) 
    pixels.fill((255, 0, 0))
    pixels.show()
    time.sleep(0.25) 
    pixels.fill((0, 0, 0))
    pixels.show()
    sys.exit(0)


signal.signal(signal.SIGTERM, cleanup_and_exit)
signal.signal(signal.SIGINT, cleanup_and_exit)


pixels.brightness = BRIGHTNESS
# let the system spin up first
alt_blink(pixels, blinks=7, delay=0.001)
pixels.fill((25, 0, 0))
pixels.show()
time.sleep(5) 
alt_blink(pixels,blinks=5, start=True)
alt_blink(pixels, blinks=5, delay=0.001)
pixels.fill((50, 0, 0))
pixels.show()
time.sleep(4) 
alt_blink(pixels, blinks=4, start=True)
alt_blink(pixels, blinks=5, delay=0.001)
pixels.brightness = 0.015
pixels.fill((75, 0, 0))
pixels.show()
time.sleep(3) 
alt_blink(pixels, blinks=3, start=True)
alt_blink(pixels, blinks=5, delay=0.001)

pixels.fill((100, 0, 0))
pixels.show()
time.sleep(2) 
alt_blink(pixels,blinks=2, start=True)

pixels.fill((125, 0, 0))
pixels.show()
time.sleep(3) 
alt_blink(pixels, start=True)
alt_blink(pixels, blinks=7, delay=0.001)
pixels.fill((150, 0, 0))
pixels.show()
time.sleep(2) 
alt_blink(pixels, blinks=2, start=True)

pixels.fill((175, 0, 0))
pixels.show()
time.sleep(3) 
alt_blink(pixels, start=True)
alt_blink(pixels, blinks=7, delay=0.001)
pixels.fill((200, 25,0))
pixels.show()
time.sleep(2) 
alt_blink(pixels, blinks=2, start=True)

pixels.fill((225, 50,0))
pixels.show()
time.sleep(3) 
alt_blink(pixels, start=True)
alt_blink(pixels, blinks=7, delay=0.001)
pixels.fill((230, 75,0))
pixels.show()
time.sleep(2) 
alt_blink(pixels, blinks=2, start=True)

pixels.fill((235, 100,0))
pixels.show()
time.sleep(3) 
alt_blink(pixels, start=True)
alt_blink(pixels, blinks=7, delay=0.001)
pixels.fill((240, 125,0))
pixels.show()
time.sleep(2) 
alt_blink(pixels, blinks=2, start=True)

pixels.fill((245, 150,0))
pixels.show()
time.sleep(1) 
alt_blink(pixels, start=True)
alt_blink(pixels, blinks=7, delay=0.001)
pixels.fill((250, 175,0))
pixels.show()
time.sleep(2) 
alt_blink(pixels, blinks=2, start=True)
alt_blink(pixels, blinks=10, delay=0.001)

pixels.fill((255, 200,0))
pixels.show()
time.sleep(1) 
alt_blink(pixels, blinks=7, delay=0.001)
alt_blink(pixels, start=True)
alt_blink(pixels, blinks=15, delay=0.001)
pixels.fill((255, 255,0))
pixels.show()
time.sleep(2) 
alt_blink(pixels, blinks=2, start=True)
alt_blink(pixels, blinks=5, delay=0.001)
pixels.fill((225, 255,0))
pixels.show()
time.sleep(1) 

alt_blink(pixels, start=True)
alt_blink(pixels, blinks=7, delay=0.001)
pixels.fill((200, 255,0))
pixels.show()
time.sleep(1) 
alt_blink(pixels, blinks=2, start=True)

pixels.fill((175, 255,0))
pixels.show()
time.sleep(0.5) 
alt_blink(pixels, start=True)
alt_blink(pixels, blinks=7, delay=0.001)
pixels.fill((150, 255,0))
pixels.show()
time.sleep(0.5) 
alt_blink(pixels, blinks=2, start=True)

pixels.fill((125, 255,0))
pixels.show()
time.sleep(0.5) 
alt_blink(pixels, start=True)
alt_blink(pixels, blinks=7, delay=0.001)
pixels.fill((100, 255,0))
pixels.show()
time.sleep(0.5) 
alt_blink(pixels, blinks=2, start=True)
alt_blink(pixels, blinks=10, delay=0.001)
pixels.fill((75, 255,0))
pixels.show()
time.sleep(0.5) 
alt_blink(pixels, start=True)
alt_blink(pixels, blinks=10, delay=0.001)

pixels.fill((50, 255,0))
pixels.show()
time.sleep(2) 
alt_blink(pixels, blinks=10, delay=0.001)

pixels.fill((25, 255,0))
pixels.show()
time.sleep(2) 
alt_blink(pixels, blinks=50, delay=0.00075)

pixels.fill((0, 255,0))
pixels.show()
time.sleep(3) 
alt_blink(pixels, blinks=50, delay=0.001)

pixels.fill((0, 0, 0))
pixels.show()
time.sleep(0.5) 


pixels.fill((0, 255, 0))
pixels.show()
time.sleep(0.5) 


pixels.fill((0, 0, 0))
pixels.show()
time.sleep(0.5) 


pixels.fill((0, 255, 0))
pixels.show()
time.sleep(0.5) 

pixels.fill((0, 0, 0))
pixels.show()
time.sleep(0.5) 

pixels.fill((0, 255, 0))
pixels.show()
time.sleep(0.5) 


pixels.fill((0, 0, 0))
pixels.show()
time.sleep(0.5) 


pixels.fill((0, 255, 0))
pixels.show()
time.sleep(0.5) 


pixels.fill((0, 0, 0))
pixels.show()
time.sleep(0.5) 


model_data, meta_data, thresh_data = fetch_model_meta_thresh()
n1=model_data['model_1h'][meta_data['none_label']]['prob']
knn=knn_v1(model_data)
W=meta_data['blend_knn_weight']
blend1=blended(model_data['model_1h'],n1,knn,W, meta_data)
wch, adv, wrn = horizon_row("h1", blend1,thresh_data,knn)

try:
    j = 0
    while True:
        j += 1
        
        if j % 1111 == 0:
            j=0
            pixels.deinit()
            del pixels
            pixels = neopixel.NeoPixel(
                LED_PIN,
                NUM_PIXELS,
                brightness=0.01,      # Global brightness control (0.0 to 1.0)
                auto_write=True,    # Set False to buffer changes, improving speed
                pixel_order=ORDER
            )
            time.sleep(0.5)
        compare = 15
        if int(wch + wrn + adv) == 0.0:
            compare = 2
        if j % compare == 0:
            
            model_data, meta_data, thresh_data = fetch_model_meta_thresh()
            n1=model_data['model_1h'][meta_data['none_label']]['prob']
            knn=knn_v1(model_data)
            W=meta_data['blend_knn_weight']
            blend1=blended(model_data['model_1h'],n1,knn,W, meta_data)
            print("update fetched!")
            wch, adv, wrn = horizon_row("h1", blend1,thresh_data,knn)
            # print(f"watches: {wch} \nadvisories: {adv}\nwarnings: {wrn}") 
     
        
        is_compute = False
        seq = clear_seq
        pixels.brightness = BRIGHTNESS
        if wch > 0:
            seq = watch_seq
            pixels.brightness = BRIGHTNESS
            is_compute = False
        if adv > 0:
            seq = advisory_seq
            pixels.brightness = BRIGHTNESS
            is_compute = False
        if wrn > 0:
            seq = warning_seq
            pixels.brightness = BRIGHTNESS
            is_compute = False
        
        if check_compute():
                seq = compute_seq
                pixels.brightness = BRIGHTNESS
                is_compute = True
        if "sequence" in seq.keys():
            for step in seq["sequence"]:
                if check_compute() and not is_compute:
                            break
                for i, pixel in enumerate(step['pixels']):
                    if i % 10 == 0:
                        if check_compute() and not is_compute:
                            
                            break
                        check_data_files_and_blink(pixels)
                    pixels[i] = pixel
                    pixels.show()
                    time.sleep(step['time'])
        else:
               pixels.fill((0, 0, 0))
               pixels.show()
               time.sleep(5)
except KeyboardInterrupt:
    # Safely shut off all LEDs when exiting with Ctrl+C
    pixels.fill((0, 0, 0))
    pixels.show()
    raise KeyboardInterrupt()

