"""Download real daily weather for Colombo, Kandy and Galle from the Open-Meteo archive API.

Source: Open-Meteo Historical Weather API (ERA5 / ERA5-Land reanalysis, ECMWF Copernicus).
Docs:   https://open-meteo.com/en/docs/historical-weather-api
No API key is needed, so anyone can re-run this script and get the same data.

GenAI declaration: this script was generated with Claude (Anthropic) and reviewed by the author.

Usage:  python scripts/fetch_data.py
Output: data/sl_weather_2015_2025.csv
"""
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

CITIES = {
    # name: (latitude, longitude, climate zone)
    "Colombo": (6.9271, 79.8612, "Wet zone - west coast"),
    "Kandy": (7.2906, 80.6337, "Wet zone - central hills"),
    "Galle": (6.0535, 80.2210, "Wet zone - south coast"),
}

DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "pressure_msl_mean",
    "cloud_cover_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sl_weather_2015_2025.csv"


def build_url(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Colombo",
    }
    return BASE_URL + "?" + urllib.parse.urlencode(params, safe=",")


def fetch(url, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # network hiccup or rate limit
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retry in {wait}s ({exc})")
            time.sleep(wait)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for city, (lat, lon, zone) in CITIES.items():
        url = build_url(lat, lon)
        print(f"Fetching {city}: {url}")
        payload = fetch(url)
        daily = payload["daily"]
        for i, day in enumerate(daily["time"]):
            row = {
                "date": day,
                "city": city,
                "climate_zone": zone,
                "latitude": payload["latitude"],
                "longitude": payload["longitude"],
                "elevation_m": payload["elevation"],
            }
            for var in DAILY_VARS:
                row[var] = daily[var][i]
            rows.append(row)
        print(f"  {len(daily['time'])} days")

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
