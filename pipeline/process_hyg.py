#!/usr/bin/env python3
"""
Von Neumann Probe Star Catalog Pipeline

Downloads the HYG star database and processes it into:
  1. A compact binary file (Float32Array) for WebGL rendering
  2. A JSON metadata file with named stars, bounds, and stats
  3. A JSON file of galactic landmarks

Usage:
    python pipeline/process_hyg.py
"""

import csv
import io
import json
import math
import os
import struct
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PC_TO_LY = 3.26156  # 1 parsec in light-years

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

HYG_URL = (
    "https://raw.githubusercontent.com/astronexus/HYG-Database/"
    "master/hyg/CURRENT/hygdata_v41.csv"
)
HYG_RAW = os.path.join(RAW_DIR, "hygdata_v41.csv")

BIN_OUT = os.path.join(OUT_DIR, "stars.bin")
META_OUT = os.path.join(OUT_DIR, "metadata.json")
LANDMARKS_OUT = os.path.join(OUT_DIR, "landmarks.json")

# Number of floats per star in the binary: x, y, z, absMag, r, g, b
FLOATS_PER_STAR = 7
BYTES_PER_STAR = FLOATS_PER_STAR * 4  # 28 bytes


# ---------------------------------------------------------------------------
# B-V color index to RGB conversion
# ---------------------------------------------------------------------------

def bv_to_rgb(bv: float) -> tuple[float, float, float]:
    """Convert B-V color index to an (r, g, b) tuple in [0, 1].

    Uses the Tanner Helland / Mitchell Charity approximation for
    blackbody-like stellar colors.
    """
    # Clamp B-V to a reasonable range
    bv = max(-0.4, min(2.0, bv))

    # Approximate color temperature from B-V (Ballesteros 2012)
    t = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))

    # Clamp temperature
    t = max(1000.0, min(40000.0, t))

    # Blackbody RGB from temperature (Tanner Helland approximation)
    x = t / 100.0

    # Red
    if x <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((x - 60.0) ** -0.1332047592)

    # Green
    if x <= 66.0:
        g = 99.4708025861 * math.log(x) - 161.1195681661
    else:
        g = 288.1221695283 * ((x - 60.0) ** -0.0755148492)

    # Blue
    if x >= 66.0:
        b = 255.0
    elif x <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(x - 10.0) - 305.0447927307

    r = max(0.0, min(255.0, r)) / 255.0
    g = max(0.0, min(255.0, g)) / 255.0
    b = max(0.0, min(255.0, b)) / 255.0

    return (r, g, b)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_hyg():
    """Download the HYG database CSV if not already cached."""
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(HYG_RAW):
        size = os.path.getsize(HYG_RAW)
        print(f"  Using cached {HYG_RAW} ({size:,} bytes)")
        return
    print(f"  Downloading from {HYG_URL} ...")
    urllib.request.urlretrieve(HYG_URL, HYG_RAW)
    size = os.path.getsize(HYG_RAW)
    print(f"  Saved {HYG_RAW} ({size:,} bytes)")


# ---------------------------------------------------------------------------
# Parse & clean
# ---------------------------------------------------------------------------

def parse_float(val: str, default: float | None = None) -> float | None:
    """Parse a float from a CSV field, returning *default* on failure."""
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def parse_hyg() -> list[dict]:
    """Read the HYG CSV and return a list of cleaned star records."""
    stars = []
    dropped = {"no_pos": 0, "no_mag": 0}

    with open(HYG_RAW, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # --- positions (parsecs -> light-years) ---
            x_pc = parse_float(row.get("x"))
            y_pc = parse_float(row.get("y"))
            z_pc = parse_float(row.get("z"))
            if x_pc is None or y_pc is None or z_pc is None:
                dropped["no_pos"] += 1
                continue

            x_ly = x_pc * PC_TO_LY
            y_ly = y_pc * PC_TO_LY
            z_ly = z_pc * PC_TO_LY

            # --- magnitudes ---
            abs_mag = parse_float(row.get("absmag"))
            app_mag = parse_float(row.get("mag"))
            if abs_mag is None or app_mag is None:
                dropped["no_mag"] += 1
                continue

            # --- color ---
            bv = parse_float(row.get("ci"))
            if bv is not None:
                r, g, b = bv_to_rgb(bv)
            else:
                r, g, b = 1.0, 1.0, 1.0  # default white

            # --- proper motion (mas/yr) ---
            pmra = parse_float(row.get("pmra"), 0.0)
            pmdec = parse_float(row.get("pmdec"), 0.0)

            # --- naming ---
            proper = (row.get("proper") or "").strip()
            bf = (row.get("bf") or "").strip()
            hip = (row.get("hip") or "").strip()
            name = proper or bf or (f"HIP {hip}" if hip else "")

            stars.append({
                "x": x_ly,
                "y": y_ly,
                "z": z_ly,
                "absMag": abs_mag,
                "appMag": app_mag,
                "r": r,
                "g": g,
                "b": b,
                "bv": bv,
                "pmra": pmra,
                "pmdec": pmdec,
                "name": name,
            })

    print(f"  Parsed {len(stars):,} stars")
    print(f"  Dropped: {dropped['no_pos']} (no position), "
          f"{dropped['no_mag']} (no magnitude)")
    return stars


# ---------------------------------------------------------------------------
# Write binary (.bin) — Float32Array compatible
# ---------------------------------------------------------------------------

def write_binary(stars: list[dict]):
    """Write the star data as a flat Float32Array-compatible binary."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(BIN_OUT, "wb") as f:
        for s in stars:
            f.write(struct.pack(
                "<7f",
                s["x"], s["y"], s["z"],
                s["absMag"],
                s["r"], s["g"], s["b"],
            ))
    size = os.path.getsize(BIN_OUT)
    print(f"  Wrote {BIN_OUT} ({size:,} bytes, {len(stars):,} stars)")


# ---------------------------------------------------------------------------
# Write metadata JSON
# ---------------------------------------------------------------------------

def write_metadata(stars: list[dict]):
    """Write metadata JSON: bounds, stats, named stars."""
    xs = [s["x"] for s in stars]
    ys = [s["y"] for s in stars]
    zs = [s["z"] for s in stars]
    mags = [s["appMag"] for s in stars]
    dists = [math.sqrt(s["x"]**2 + s["y"]**2 + s["z"]**2) for s in stars]

    # Named stars: pick the 500 brightest by apparent magnitude (lower = brighter)
    named_candidates = [
        (i, s) for i, s in enumerate(stars) if s["name"]
    ]
    named_candidates.sort(key=lambda t: t[1]["appMag"])
    named_stars = [
        {"name": s["name"], "index": i, "appMag": round(s["appMag"], 2)}
        for i, s in named_candidates[:500]
    ]

    meta = {
        "starCount": len(stars),
        "bytesPerStar": BYTES_PER_STAR,
        "floatsPerStar": FLOATS_PER_STAR,
        "fieldOrder": ["x", "y", "z", "absMag", "r", "g", "b"],
        "bounds": {
            "x": [round(min(xs), 2), round(max(xs), 2)],
            "y": [round(min(ys), 2), round(max(ys), 2)],
            "z": [round(min(zs), 2), round(max(zs), 2)],
        },
        "magnitudeRange": [round(min(mags), 2), round(max(mags), 2)],
        "distanceRange": [round(min(dists), 2), round(max(dists), 2)],
        "namedStars": named_stars,
    }

    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {META_OUT} ({len(named_stars)} named stars)")


# ---------------------------------------------------------------------------
# Write galactic landmarks JSON
# ---------------------------------------------------------------------------

def write_landmarks():
    """Write hardcoded galactic landmarks in the same coordinate system.

    Positions are approximate, in light-years, using the same galactic
    Cartesian frame as HYG (Sun at origin, x toward galactic center,
    y in direction of galactic rotation, z toward north galactic pole).
    """
    landmarks = {
        "points": [
            {
                "name": "Sagittarius A* (Galactic Center)",
                "x": 26000.0, "y": 0.0, "z": -20.0,
                "description": "Supermassive black hole at the Milky Way center"
            },
            {
                "name": "Large Magellanic Cloud",
                "x": -1300.0, "y": -33800.0, "z": -49300.0,
                "description": "Satellite galaxy, ~160,000 ly from Sol"
            },
            {
                "name": "Small Magellanic Cloud",
                "x": 12400.0, "y": -36200.0, "z": -44500.0,
                "description": "Satellite galaxy, ~200,000 ly from Sol"
            },
            {
                "name": "Orion Nebula (M42)",
                "x": -980.0, "y": -520.0, "z": -380.0,
                "description": "Nearby star-forming region, ~1,344 ly"
            },
            {
                "name": "Crab Nebula (M1)",
                "x": -3900.0, "y": 2700.0, "z": -1200.0,
                "description": "Supernova remnant, ~6,500 ly"
            },
            {
                "name": "Eagle Nebula (M16)",
                "x": 5500.0, "y": -1900.0, "z": 80.0,
                "description": "Star-forming region with Pillars of Creation, ~7,000 ly"
            },
        ],
        "spiralArms": [
            {
                "name": "Perseus Arm",
                "description": "Major spiral arm outside Sol's position",
                "polyline": [
                    {"x": 18000, "y": 22000, "z": 0},
                    {"x": 8000, "y": 28000, "z": 0},
                    {"x": -5000, "y": 29000, "z": 0},
                    {"x": -18000, "y": 24000, "z": 0},
                    {"x": -26000, "y": 14000, "z": 0},
                ]
            },
            {
                "name": "Sagittarius Arm",
                "description": "Major spiral arm between Sol and galactic center",
                "polyline": [
                    {"x": 22000, "y": 10000, "z": 0},
                    {"x": 16000, "y": 18000, "z": 0},
                    {"x": 6000, "y": 22000, "z": 0},
                    {"x": -6000, "y": 20000, "z": 0},
                    {"x": -16000, "y": 12000, "z": 0},
                ]
            },
            {
                "name": "Orion-Cygnus Spur (Local Arm)",
                "description": "Minor arm where our Sun resides",
                "polyline": [
                    {"x": 8000, "y": 3000, "z": 0},
                    {"x": 3000, "y": 6000, "z": 0},
                    {"x": -3000, "y": 7000, "z": 0},
                    {"x": -8000, "y": 5000, "z": 0},
                ]
            },
            {
                "name": "Scutum-Centaurus Arm",
                "description": "Major spiral arm inside Sol's orbit",
                "polyline": [
                    {"x": 26000, "y": -4000, "z": 0},
                    {"x": 24000, "y": 8000, "z": 0},
                    {"x": 18000, "y": 16000, "z": 0},
                    {"x": 8000, "y": 20000, "z": 0},
                ]
            },
        ],
        "coordinateSystem": {
            "origin": "Sol",
            "units": "light-years",
            "x": "toward galactic center",
            "y": "direction of galactic rotation",
            "z": "toward north galactic pole",
        }
    }

    with open(LANDMARKS_OUT, "w", encoding="utf-8") as f:
        json.dump(landmarks, f, indent=2)
    print(f"  Wrote {LANDMARKS_OUT}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

KNOWN_STARS = {
    "Sirius": {
        "dist_ly_approx": 8.6,
        "absMag_approx": 1.42,
    },
    "Alpha Centauri": {  # matches "Rigel Kentaurus A" in HYG ProperName
        "alt_names": ["Rigel Kentaurus", "Rigil Kentaurus"],
        "dist_ly_approx": 4.37,
        "absMag_approx": 4.38,
    },
    "Betelgeuse": {
        "dist_ly_approx": 700.0,
        "absMag_approx": -5.85,
    },
}


def verify(stars: list[dict]):
    """Reload the binary and spot-check known stars."""
    print("\n--- Verification ---")

    # Reload binary
    with open(BIN_OUT, "rb") as f:
        data = f.read()
    n = len(data) // BYTES_PER_STAR
    assert n == len(stars), f"Star count mismatch: binary has {n}, expected {len(stars)}"
    print(f"  Binary contains {n:,} stars (matches)")

    # Check known stars by scanning the parsed list
    for target_name, expected in KNOWN_STARS.items():
        alt_names = expected.get("alt_names", [])
        search_names = [target_name] + alt_names

        found = None
        found_idx = None
        for i, s in enumerate(stars):
            for sn in search_names:
                if sn.lower() in s["name"].lower():
                    found = s
                    found_idx = i
                    break
            if found:
                break

        if found is None:
            print(f"  WARNING: {target_name} not found in catalog")
            continue

        dist = math.sqrt(found["x"]**2 + found["y"]**2 + found["z"]**2)

        # Read back from binary to compare
        offset = found_idx * BYTES_PER_STAR
        vals = struct.unpack_from("<7f", data, offset)

        print(f"  {target_name} ('{found['name']}', index {found_idx}):")
        print(f"    Position: ({found['x']:.1f}, {found['y']:.1f}, {found['z']:.1f}) ly")
        print(f"    Distance: {dist:.1f} ly (expected ~{expected['dist_ly_approx']})")
        print(f"    AbsMag:   {found['absMag']:.2f} (expected ~{expected['absMag_approx']})")
        print(f"    Color:    ({found['r']:.3f}, {found['g']:.3f}, {found['b']:.3f})")
        print(f"    Binary readback matches: "
              f"{vals[0]:.1f}=={found['x']:.1f}, "
              f"{vals[3]:.2f}=={found['absMag']:.2f}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(stars: list[dict]):
    """Print summary statistics."""
    dists = [math.sqrt(s["x"]**2 + s["y"]**2 + s["z"]**2) for s in stars]
    mags = [s["appMag"] for s in stars]
    abs_mags = [s["absMag"] for s in stars]
    with_color = sum(1 for s in stars if s["bv"] is not None)
    with_name = sum(1 for s in stars if s["name"])

    print("\n=== Summary ===")
    print(f"  Total stars:        {len(stars):,}")
    print(f"  With B-V color:     {with_color:,}")
    print(f"  With name:          {with_name:,}")
    print(f"  Distance range:     {min(dists):.2f} – {max(dists):.1f} ly")
    print(f"  App magnitude range: {min(mags):.2f} – {max(mags):.2f}")
    print(f"  Abs magnitude range: {min(abs_mags):.2f} – {max(abs_mags):.2f}")
    print(f"  Binary file size:   {os.path.getsize(BIN_OUT):,} bytes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Von Neumann Probe — Star Catalog Pipeline ===\n")

    print("[1/6] Downloading HYG database...")
    download_hyg()

    print("\n[2/6] Parsing and cleaning...")
    stars = parse_hyg()

    print("\n[3/6] Writing binary star data...")
    write_binary(stars)

    print("\n[4/6] Writing metadata JSON...")
    write_metadata(stars)

    print("\n[5/6] Writing galactic landmarks...")
    write_landmarks()

    print("\n[6/6] Verifying...")
    verify(stars)

    print_summary(stars)
    print("\nDone.")


if __name__ == "__main__":
    main()
