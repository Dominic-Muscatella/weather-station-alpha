from __future__ import annotations

CANONICAL = {"temp": "C", "pressure": "hPa", "humidity": "%"}


_TEMP = {                       
    "c": lambda v: v,
    "celsius": lambda v: v,
    "f": lambda v: (v - 32.0) * 5.0 / 9.0,
    "fahrenheit": lambda v: (v - 32.0) * 5.0 / 9.0,
    "k": lambda v: v - 273.15,
    "kelvin": lambda v: v - 273.15,
}
_PRESSURE = {                   
    "hpa": lambda v: v,
    "mb": lambda v: v,
    "mbar": lambda v: v,
    "millibar": lambda v: v,
    "inhg": lambda v: v * 33.8638866667,
    "inches_hg": lambda v: v * 33.8638866667,
    "pa": lambda v: v / 100.0,
    "kpa": lambda v: v * 10.0,
    "mmhg": lambda v: v * 1.33322387415,
    "psi": lambda v: v * 68.9475729318,
}
_HUMIDITY = {                   
    "%": lambda v: v,
    "percent": lambda v: v,
    "pct": lambda v: v,
    "rh": lambda v: v,
    "frac": lambda v: v * 100.0,
    "fraction": lambda v: v * 100.0,
    "ratio": lambda v: v * 100.0,
}
_TABLE = {"temp": _TEMP, "pressure": _PRESSURE, "humidity": _HUMIDITY}


def known_units(channel: str):
    return sorted(_TABLE[channel].keys())


def to_canonical(channel: str, values, from_unit: str):

    fam = _TABLE.get(channel)
    if fam is None:
        raise KeyError(f"no conversion table for channel {channel!r}")
    key = str(from_unit).strip().lower()
    if key not in fam:
        raise ValueError(
            f"unknown {channel} unit {from_unit!r}; known: {known_units(channel)}"
        )
    return fam[key](values)


def guess_unit(channel: str, lo: float, med: float, hi: float):

    if channel == "temp":
        
        if 230 <= med <= 330:
            return "K", True
        if med > 45:               
            return "F", True
        if -30 <= med <= 45:
            return "C", True
        return ("F", False) if med > 45 else ("C", False)
    if channel == "pressure":
        if 25 <= med <= 35:
            return "inHg", True
        if 900 <= med <= 1100:
            return "hPa", True
        if 90000 <= med <= 110000:
            return "Pa", True
        if 90 <= med <= 110:
            return "kPa", True
        return "hPa", False
    if channel == "humidity":
        if hi <= 1.5:
            return "fraction", True
        if 0 <= hi <= 100:
            return "%", True
        return "%", False
    return "?", False
