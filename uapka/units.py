"""Unit parsing and conversion.

Scenario inputs (ranges, sizes, altitudes) are typed by humans in whatever
units their source used (HUD readouts in feet / nautical miles, reports in km),
so every length input accepts an explicit unit suffix. A bare number is metres.
"""

from __future__ import annotations

import re

G0 = 9.80665  # standard gravity, m/s^2

LENGTH_UNITS = {
    "m": 1.0,
    "km": 1000.0,
    "ft": 0.3048,
    "kft": 304.8,
    "nmi": 1852.0,
    "nm": 1852.0,
    "mi": 1609.344,
}

MS_TO_KT = 1.0 / 0.514444
MS_TO_KMH = 3.6
MS_TO_MPH = 1.0 / 0.44704

_NUM_UNIT = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z]*)\s*$")


def parse_length(text: str, default_unit: str = "m") -> float:
    """Parse '12km', '30000 ft', '5nmi', '800' -> metres."""
    m = _NUM_UNIT.match(text)
    if not m:
        raise ValueError(f"Cannot parse length: {text!r}")
    value = float(m.group(1))
    unit = (m.group(2) or default_unit).lower()
    if unit not in LENGTH_UNITS:
        raise ValueError(f"Unknown length unit {unit!r} in {text!r}; use one of {sorted(LENGTH_UNITS)}")
    return value * LENGTH_UNITS[unit]


def parse_length_list(text: str, default_unit: str = "m") -> list[float]:
    """Parse a list or sweep of lengths, in metres.

    Accepted forms (may be mixed, comma/semicolon separated):
      '10km, 20km, 30nmi'         explicit values
      '5km:50km:5km'              start:stop:step sweep (stop inclusive)
      '1km:100km:x10'             not supported; use 'log:1km:100km:7' instead
      'log:1km:100km:7'           7 log-spaced values
    """
    out: list[float] = []
    for part in re.split(r"[;,]", text):
        part = part.strip()
        if not part:
            continue
        if part.lower().startswith("log:"):
            _, a, b, n = part.split(":")
            a_m, b_m, n_i = parse_length(a, default_unit), parse_length(b, default_unit), int(n)
            if a_m <= 0 or b_m <= 0 or n_i < 2:
                raise ValueError(f"Bad log sweep: {part!r}")
            ratio = (b_m / a_m) ** (1.0 / (n_i - 1))
            out.extend(a_m * ratio**i for i in range(n_i))
        elif ":" in part:
            a, b, s = part.split(":")
            a_m, b_m, s_m = (parse_length(x, default_unit) for x in (a, b, s))
            if s_m <= 0 or b_m < a_m:
                raise ValueError(f"Bad sweep: {part!r}")
            n = int(round((b_m - a_m) / s_m))
            out.extend(a_m + i * s_m for i in range(n + 1))
        else:
            out.append(parse_length(part, default_unit))
    if not out:
        raise ValueError("No values given")
    return out


def fmt_length(m: float) -> str:
    if abs(m) >= 1000:
        return f"{m / 1000:.3g} km"
    return f"{m:.3g} m"
