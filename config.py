"""
config.py
=========
Single source of truth for the whole pipeline. Tweak here, not in the modules.

The overarching goal (per project brief):
    "Make a local weather station + robust-enough severe weather prediction
     models such that if all our fancy doppler radars shut off and the NWS were
     disbanded, I'd still have a half-decent severe-threat warning system based
     off of local measurements."
"""
from __future__ import annotations
import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")            # raw downloaded data lands here
BUILD_DIR = os.path.join(ROOT, "build")          # the assembled .npz dataset
PACKAGE_DIR = os.path.join(ROOT, "model_package")  # final saved bundle
for _d in (DATA_DIR, BUILD_DIR, PACKAGE_DIR):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------------
# Classes (multi-label). "none" is appended as an explicit abstain/no-event head.
# ----------------------------------------------------------------------------
CLASSES = ["tornado", "flood", "wind", "t-storm", "blizzard", "severe heat", "severe cold"]
NONE_LABEL = "none"
ALL_LABELS = CLASSES + [NONE_LABEL]              # 8 sigmoid outputs
N_OUTPUTS = len(ALL_LABELS)

# Richer text used to embed each class for the SBERT semantic match.
CLASS_DESCRIPTIONS = {
    "tornado":      "tornado funnel cloud waterspout",
    "flood":        "flood flash flooding heavy rain river flood urban flooding",
    "wind":         "high wind strong damaging straight line non-thunderstorm wind gusts",
    "t-storm":      "severe thunderstorm convective storm hail lightning thunderstorm wind",
    "blizzard":     "blizzard heavy snow winter storm ice storm whiteout lake effect snow",
    "severe heat":  "excessive heat heat wave dangerously high temperature",
    "severe cold":  "extreme cold wind chill hard freeze dangerously low temperature",
}

# ----------------------------------------------------------------------------
# Channels (the three local-station measurements). Order is fixed everywhere.
# ----------------------------------------------------------------------------
CHANNELS = ["temp", "pressure", "humidity"]
N_CHANNELS = len(CHANNELS)

# ----------------------------------------------------------------------------
# Units.  Everything is converted into these canonical units before scaling,
# so training data and the live feed can be in whatever the source emits.
#   temp -> C, pressure -> hPa, humidity -> %   (see units.py)
# ----------------------------------------------------------------------------
# What the HISTORICAL training CSV (Kaggle curiel/chicago-weather-database) is in.
# Confirmed from inspect_data.py: TEMP is Celsius (April-midnight ~1, med ~11,
# max ~31 = textbook Chicago C), ATM_PRESS is kPa (~100.5 -> 1005 hPa), HMDT %.
TRAIN_UNITS = {"temp": "C", "pressure": "kPa", "humidity": "%"}

# What your LIVE deployment emits, PER CHANNEL (sources can differ):
#   pressure <- local Pi USB barometer  (you set its output; you said inHg now)
#   temp     <- remote station, the TRANSMITTED value, NOT the LCD display
#   humidity <- remote station (almost always %)
# NOTE: a station's display toggle (F/C) often does NOT change what the radio
# transmits. Pin this to the transmitted value once you sniff it. If you decode
# with rtl_433, its JSON is commonly already temperature_C / humidity % — verify.
LIVE_UNITS = {"temp": "F", "pressure": "inHg", "humidity": "%"}

# Flexible column-name resolution for the weather CSV. The loader picks the first
# candidate present (case-insensitive). Add your file's names here if they differ.
WEATHER_COLUMN_CANDIDATES = {
    "datetime": ["datetime", "date_time", "date", "time", "timestamp", "dt", "valid", " Date Time"],
    "temp":     ["temp", "temperature", "tempc", "tempf", "air_temp", "t", "temperature_c", "temp_c"],
    "pressure": ["pressure", "press", "baro", "barometric_pressure", "slp",
                 "sea_level_pressure", "mslp", "pres",
                 "atm_press", "atmospheric_pressure", "station_pressure", "stp", "atmp"],
    "humidity": ["humidity", "rel_humidity", "relative_humidity", "rh", "humid",
                 "relativehumidity", "hmdt", "rhum"],
}

# If no single datetime column is found, assemble one from component columns
# (the Kaggle file splits it into YEAR / MO / DY / HR). First candidate wins.
WEATHER_DATETIME_PARTS = {
    "year":   ["year", "yr", "yyyy"],
    "month":  ["mo", "month", "mon", "mm"],
    "day":    ["dy", "day", "dd", "dom"],
    "hour":   ["hr", "hour", "hh"],
    "minute": ["mi", "min", "minute", "mm_"],   # optional
}

# ----------------------------------------------------------------------------
# Weather sources. Each is loaded, converted to canonical units (C/hPa/%) and
# concatenated onto one hourly timeline. On overlapping timestamps, sources
# EARLIER in this list win, so list your most-trusted source first.
#   - "units":   per-source unit dict (sources can differ).
#   - "columns": per-source column overrides; None -> use WEATHER_COLUMN_CANDIDATES.
#   - "optional": if True, silently skipped when the file is absent.
#
# DEFAULT = ASOS-only (one homogeneous O'Hare record, 2005->present). We do NOT
# mix in the Kaggle file because the two use different PRESSURE REFERENCES:
# Kaggle ATM_PRESS is station pressure (~995 hPa median at O'Hare's elevation)
# while ASOS pressure_hpa is sea-level/MSLP (~1020 hPa). Blending them would put
# a ~25 hPa source-dependent step into the pressure channel — the model's most
# important input — so we keep a single datum. To splice Kaggle back in for
# recent years, uncomment its entry below (but reconcile the pressure datum).
# ----------------------------------------------------------------------------
ASOS_STATION = "ORD"                 # Chicago O'Hare (use "MDW" for Midway)
ASOS_START_YEAR = 2005               # how far back to pull hourly ASOS history

WEATHER_SOURCES = [
    # IEM ASOS history (written by download_data.download_asos_weather)
    {"path": os.path.join(DATA_DIR, "asos_weather.csv"),
     "units": {"temp": "F", "pressure": "hPa", "humidity": "%"},
     "columns": {"datetime": ["valid"], "temp": ["tmpf"],
                 "pressure": ["pressure_hpa", "mslp"], "humidity": ["relh"]},
     "optional": True},
    # # Kaggle curiel/chicago-weather-database (~2021-04 -> present). Disabled:
    # # its ATM_PRESS is STATION pressure, ~25 hPa below ASOS MSLP. To use it,
    # # reconcile the datum (e.g. convert to MSLP) before re-enabling.
    # {"path": os.path.join(DATA_DIR, "chicago_weather.csv"),
    #  "units": {"temp": "C", "pressure": "kPa", "humidity": "%"},
    #  "columns": None, "optional": True},
]

# Missing-data sentinels to treat as NaN (the Kaggle file uses -999 in TEMP).
SENTINEL_VALUES = (-999, -999.0, -9999, -9999.0, 9999, 9999.0)

# After unit conversion, NaN out physically impossible readings (unit-agnostic
# guard; also mops up any sentinel that survived conversion). Canonical units.
PLAUSIBLE_RANGE = {
    "temp": (-60.0, 60.0),        # C
    "pressure": (850.0, 1100.0),  # hPa
    "humidity": (0.0, 100.0),     # %
}

# ----------------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------------
HOURLY_WINDOW_DAYS = 7
HOURLY_WINDOW_LEN = HOURLY_WINDOW_DAYS * 24       # 168 hourly points  -> X
SUBHOURLY_WINDOW_HOURS = 48
SUBHOURLY_INTERVAL_MIN = 5
SUBHOURLY_WINDOW_LEN = SUBHOURLY_WINDOW_HOURS * (60 // SUBHOURLY_INTERVAL_MIN)  # 576 -> X2

# Each hourly observation is treated as occurring at the :30 mark of its hour
# when we interpolate the sub-hourly (X2) series.
ANCHOR_MINUTE = 30

# "log" interpolates linearly in log-space (falls back to linear per-channel
# when that channel has non-positive values inside the window, e.g. sub-zero
# temperature). "linear" forces plain linear interpolation.
INTERP_METHOD = "log"

# Stride between consecutive window-end timestamps, in hours. 1 = densest.
# Adjacent windows overlap ~99% (a 168h window shares 165h with its neighbor at
# stride 3), so stride is the biggest wall-clock lever with little signal loss.
# 3 is a good default (3x fewer windows, ~3x faster). Drop to 1 for a final
# dense run if you want maximum positive-window coverage and have the patience.
WINDOW_STRIDE_HOURS = 1

# Forecast horizons (hours after the window end).
HORIZON_1H = 1
HORIZON_24H = 24

# Labeling tolerance (minutes): each event interval is widened by this much on
# BOTH sides before checking overlap with a forecast window. This is what makes
# the sparse 1-hour target trainable — an event that begins shortly after the
# window end (or ended shortly before it) still counts as a positive, i.e. the
# label means "event within ~tolerance of the forecast window" rather than
# "event strictly inside it". Set 0 for the strict (t, t+horizon] definition.
LABEL_TOLERANCE_MIN = 60

# Temperature-derived heat/cold labels (in addition to NOAA Heat/Cold events).
# If ANY hourly temperature in the forecast window is above HEAT_THRESHOLD_F,
# mark the 'severe heat' class; below COLD_THRESHOLD_F, mark 'severe cold'.
# Thresholds are in Fahrenheit (converted to canonical C internally). These
# count like a watch for sample weighting (WEIGHT_WATCH_ONLY), not a warned
# storm event. Set either to None to disable that side.
HEAT_THRESHOLD_F = 85.0
COLD_THRESHOLD_F = 10.0

# A window is dropped if more than this fraction of its required hourly history
# is missing (after gap interpolation up to MAX_GAP_HOURS).
MAX_MISSING_FRACTION = 0.10
MAX_GAP_HOURS = 6                                 # gaps <= this are interpolated

# Embargo (hours) inserted between train/val/test chronological splits so that
# overlapping windows don't leak across the split boundary.
SPLIT_EMBARGO_HOURS = HOURLY_WINDOW_DAYS * 24

# Chronological split fractions (no shuffling — the test slice is the most
# recent period). TEST gets the remainder: 1 - TRAIN_SPLIT - VAL_SPLIT.
#   - val  is the early-stopping signal (used DURING training).
#   - test is the untouched final scorecard (never seen in training).
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1        # test = 0.1

# After early stopping finds the epoch budget on val, refit the SHIPPED model on
# train+val combined for that many epochs (test stays untouched). This recovers
# the val data into the deployed model, so only ~TEST fraction is withheld from
# it, while metrics stay honest. Set False to ship the early-stopped model.
REFIT_ON_TRAINVAL = True

# ----------------------------------------------------------------------------
# Date features: month one-hot (12) + month-third one-hot (3) = 15 dims.
# Day-of-month (31 dims) was dropped — there's no weather reason the 14th differs
# from the 15th, so it was 31 dims of noise inviting overfitting. The 3-bin
# "third of the month" (beginning / middle / end) keeps coarse within-month
# position without the noise.
# ----------------------------------------------------------------------------
N_MONTHS = 12
N_MONTH_BINS = 0                                 # beginning / middle / end
DATE_FEAT_DIM = N_MONTHS + N_MONTH_BINS          # 15

# ----------------------------------------------------------------------------
# "Warned with a real event" emphasis.
# Storm-events overlaps are the strong ground truth -> upweighted in the loss.
# Watch overlaps (from IEM) are a weaker positive signal -> mild upweight.
# ----------------------------------------------------------------------------
WEIGHT_BASE        = 1.0
WEIGHT_STORM_EVENT = 0.25
WEIGHT_WARNING     = 0.25     # NWS warning overlap (between event and watch)
WEIGHT_WATCH_ONLY  = 0.75

# ----------------------------------------------------------------------------
# Region filter for NOAA Storm Events.
# ----------------------------------------------------------------------------
CZ_NAME_CONTAINS = "cook"
STATE_CONTAINS = "illinois"

# Event types to DROP before labeling: real NOAA records that aren't predictable
# from local temp/pressure/humidity (coastal/lake hazards, slow-onset, etc.).
# Seen in the real Cook/IL data; dropping them keeps the class labels clean.
IGNORE_EVENT_TYPES = {
    "rip current", "high surf", "seiche", "lakeshore flood", "coastal flood",
    "astronomical low tide", "drought", "dense fog", "marine dense fog",
    "sneakerwave", "tsunami", "volcanic ash", "wildfire",
}

# If the best class match for a (non-ignored) event scores below this, route it
# to "none" rather than forcing a weak/incorrect class. SBERT cosine in [0,1];
# the keyword fallback returns 1.0 on a hit, 0.0 otherwise, so the threshold
# only bites for SBERT. Set 0.0 to disable.
EVENT_MATCH_MIN_SCORE = 0.30

# Hard-coded event_type -> class overrides, applied AFTER the mapper (SBERT or
# keyword) and taking precedence over it. SBERT mis-files "Winter Weather" as
# severe cold; it's a snow/winter-storm phenomenon, so we pin it to blizzard.
# Keys are matched case-insensitively against the exact EVENT_TYPE.
EVENT_CLASS_OVERRIDES = {
    "winter weather": "blizzard",
}

# ----------------------------------------------------------------------------
# Years to pull. Extend these back to match however far your weather history
# goes (ASOS_START_YEAR) so the labels cover the added input years.
# ----------------------------------------------------------------------------
STORM_EVENTS_YEARS = list(range(ASOS_START_YEAR, 2027))   # default 2005..2026
WATCH_YEARS = list(range(ASOS_START_YEAR, 2027))          # default 2005..2026

# ----------------------------------------------------------------------------
# Monte-Carlo dropout confidence
# ----------------------------------------------------------------------------
MC_PASSES = 100
MC_THRESHOLD = 0.15          # sigmoid threshold per pass
MC_WEIGHT = 2                # weight on MC exceedance
RAW_WEIGHT = 1               # weight on raw (dropout-off) sigmoid
MC_CI = (2.5, 97.5)          # percentile band reported as the confidence interval

# ----------------------------------------------------------------------------
# Model architecture
# ----------------------------------------------------------------------------
CONV_CHANNELS = [32, 64]     # per-leg conv stack widths
CONV_KERNEL = 5
LEG_FC = 256                  # per-leg fully-connected size after pooling+date
HEAD_FC = [256, 256]          # shared head widths
DROPOUT_P = 0.3333333333             # this is the MC-dropout rate

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
EPOCHS = 50
BATCH_SIZE = 1024             # bigger batches = better CPU/GPU throughput
LR = 1.5e-5                    # lowered from 1e-3 — 1e-3 overfit in ~1 epoch
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 45
SEED = 1338

# Loss: multi-label FOCAL loss instead of pos_weight*BCE. With prevalences down
# to ~0.0004, a clipped pos_weight produced huge, unstable gradients (val loss
# exploded). Focal down-weights easy negatives via (1-p)^gamma and stays stable;
# alpha tilts weight toward the rare positive class.
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.5            # weight on the positive term (0.5 = neutral)

# Stabilizers: clip gradients, and drop LR when val loss plateaus.
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
