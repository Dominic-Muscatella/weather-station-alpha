import uuid
import os
import config as C


def write_to_start_lock_file():
    try:
        with open(os.path.join(C.SYS_LOCK_DIR, f"ml_str.lock"), "w", encoding="utf-8") as file:
            
            file.write(str(uuid.uuid4()))
    except Exception as E:
        print(E)
        print('continuing anyways...')


def write_to_monte_lock_file():
    try:
        with open(os.path.join(C.SYS_LOCK_DIR, f"monte_fin.lock"), "w", encoding="utf-8") as file:
            
            file.write(str(uuid.uuid4()))
    except Exception as E:
        print(E)
        print('continuing anyways...')


def write_to_knn_lock_file():
    try:
        with open(os.path.join(C.SYS_LOCK_DIR, f"knn_fin.lock"), "w", encoding="utf-8") as file:
            
            file.write(str(uuid.uuid4()))
    except Exception as E:
        print(E)
        print('continuing anyways...')


def get_file_write_time(fpath):
    tmstp = 0
    try:
        tmstp = os.path.getmtime(fpath)
    except Exception as E:
        print(E)
        print('continuing anyways...')
    return tmstp


def write_to_sensor_alt_lock_file(sensor):
    try:
        with open(f"sensor{sensor}_alt.lock", "w", encoding="utf-8") as file:
            
            file.write(str(uuid.uuid4()))
    except Exception as E:
        print(E)
        print('continuing anyways...')


def write_to_data_alt_lock_file(tPath="data_alt.lock"):
    try:
        with open(tPath, "w", encoding="utf-8") as file:
            
            file.write(str(uuid.uuid4()))
    except Exception as E:
        print(E)
        print('continuing anyways...')