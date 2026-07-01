from __future__ import annotations
import os




ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")            
BUILD_DIR = os.path.join(ROOT, "build")          
PACKAGE_DIR = os.path.join(ROOT, "model_package")  
for _d in (DATA_DIR, BUILD_DIR, PACKAGE_DIR):
    os.makedirs(_d, exist_ok=True)




CLASSES = ["tornado", "flood", "wind", "t-storm", "blizzard", "severe heat", "severe cold"]
NONE_LABEL = "none"
ALL_LABELS = CLASSES + [NONE_LABEL]              
N_OUTPUTS = len(ALL_LABELS)


CLASS_DESCRIPTIONS = {
    "tornado":      "tornado funnel cloud waterspout",
    "flood":        "flood flash flooding heavy rain river flood urban flooding",
    "wind":         "high wind strong damaging straight line non-thunderstorm wind gusts",
    "t-storm":      "severe thunderstorm convective storm hail lightning thunderstorm wind",
    "blizzard":     "blizzard heavy snow winter storm ice storm whiteout lake effect snow",
    "severe heat":  "excessive heat heat wave dangerously high temperature",
    "severe cold":  "extreme cold wind chill hard freeze dangerously low temperature",
}




CHANNELS = ["temp", "pressure", "humidity"]
N_CHANNELS = len(CHANNELS)









TRAIN_UNITS = {"temp": "C", "pressure": "kPa", "humidity": "%"}








LIVE_UNITS = {"temp": "F", "pressure": "inHg", "humidity": "%"}



WEATHER_COLUMN_CANDIDATES = {
    "datetime": ["datetime", "date_time", "date", "time", "timestamp", "dt", "valid", " Date Time"],
    "temp":     ["temp", "temperature", "tempc", "tempf", "air_temp", "t", "temperature_c", "temp_c"],
    "pressure": ["pressure", "press", "baro", "barometric_pressure", "slp",
                 "sea_level_pressure", "mslp", "pres",
                 "atm_press", "atmospheric_pressure", "station_pressure", "stp", "atmp"],
    "humidity": ["humidity", "rel_humidity", "relative_humidity", "rh", "humid",
                 "relativehumidity", "hmdt", "rhum"],
}



WEATHER_DATETIME_PARTS = {
    "year":   ["year", "yr", "yyyy"],
    "month":  ["mo", "month", "mon", "mm"],
    "day":    ["dy", "day", "dd", "dom"],
    "hour":   ["hr", "hour", "hh"],
    "minute": ["mi", "min", "minute", "mm_"],   
}

















ASOS_STATION = "ORD"                 
ASOS_START_YEAR = 2005               

WEATHER_SOURCES = [
    
    {"path": os.path.join(DATA_DIR, "asos_weather.csv"),
     "units": {"temp": "F", "pressure": "hPa", "humidity": "%"},
     "columns": {"datetime": ["valid"], "temp": ["tmpf"],
                 "pressure": ["pressure_hpa", "mslp"], "humidity": ["relh"]},
     "optional": True},
    
    
    
    
    
    
]


SENTINEL_VALUES = (-999, -999.0, -9999, -9999.0, 9999, 9999.0)



PLAUSIBLE_RANGE = {
    "temp": (-60.0, 60.0),        
    "pressure": (850.0, 1100.0),  
    "humidity": (0.0, 100.0),     
}




HOURLY_WINDOW_DAYS = 7
HOURLY_WINDOW_LEN = HOURLY_WINDOW_DAYS * 24       
SUBHOURLY_WINDOW_HOURS = 48
SUBHOURLY_INTERVAL_MIN = 5
SUBHOURLY_WINDOW_LEN = SUBHOURLY_WINDOW_HOURS * (60 // SUBHOURLY_INTERVAL_MIN)  



ANCHOR_MINUTE = 30




INTERP_METHOD = "log"






WINDOW_STRIDE_HOURS = 1


HORIZON_1H = 1
HORIZON_24H = 24







LABEL_TOLERANCE_MIN = 60







HEAT_THRESHOLD_F = 85.0
COLD_THRESHOLD_F = 10.0



MAX_MISSING_FRACTION = 0.10
MAX_GAP_HOURS = 6                                 



SPLIT_EMBARGO_HOURS = HOURLY_WINDOW_DAYS * 24





TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1        





REFIT_ON_TRAINVAL = True








N_MONTHS = 12
N_MONTH_BINS = 0                                 
DATE_FEAT_DIM = N_MONTHS + N_MONTH_BINS          






WEIGHT_BASE        = 1.0
WEIGHT_STORM_EVENT = 0.25
WEIGHT_WARNING     = 0.25     
WEIGHT_WATCH_ONLY  = 0.75




CZ_NAME_CONTAINS = "cook"
STATE_CONTAINS = "illinois"




IGNORE_EVENT_TYPES = {
    "rip current", "high surf", "seiche", "lakeshore flood", "coastal flood",
    "astronomical low tide", "drought", "dense fog", "marine dense fog",
    "sneakerwave", "tsunami", "volcanic ash", "wildfire",
}





EVENT_MATCH_MIN_SCORE = 0.30





EVENT_CLASS_OVERRIDES = {
    "winter weather": "blizzard",
}





STORM_EVENTS_YEARS = list(range(ASOS_START_YEAR, 2027))   
WATCH_YEARS = list(range(ASOS_START_YEAR, 2027))          




MC_PASSES = 100
MC_THRESHOLD = 0.15          
MC_WEIGHT = 2                
RAW_WEIGHT = 1               
MC_CI = (2.5, 97.5)          




CONV_CHANNELS = [32, 64]     
CONV_KERNEL = 5
LEG_FC = 256                  
HEAD_FC = [256, 256]          
DROPOUT_P = 0.3333333333             




EPOCHS = 50
BATCH_SIZE = 1024             
LR = 1.5e-5                    
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 45
SEED = 1338





FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.5            


GRAD_CLIP_NORM = 5.0
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 6

VERSION = "0.1.0"


IEM_ASOS_RANGE = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    "?station={st}&data=tmpf&data=relh&data=mslp&data=alti"
    "&year1={y1}&month1={m1}&day1={d1}&hour1={h1}"
    "&year2={y2}&month2={m2}&day2={d2}&hour2={h2}"
    "&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no"
)
FETCH_DAYS = 10
ROLLING_DAYS = 30
ARCHIVE_WEEK_DAYS = 7
IEM_WARN_RANGE = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"
    "?accept=csv&timeopt=1"
    "&sts={sts}&ets={ets}"
    "&location_group=wfo&wfo[]={wfo}"
)
MC_BATCHES = 10

SYS_LOCK_DIR = "/home/dom/weather_station_alpha"

BLINK_COLOR = (0, 175, 0)
