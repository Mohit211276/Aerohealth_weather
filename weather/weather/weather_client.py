"""Open-Meteo client (free, no API key, non-commercial use). Standard library only."""
import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone as dt_timezone

logger = logging.getLogger(__name__)

API_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo hourly variable -> WeatherReading field
FIELD_MAP = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "surface_pressure": "pressure_hpa",
    "precipitation": "precipitation_mm",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_direction_deg",
    "boundary_layer_height": "boundary_layer_m",
}


def _get_json(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "VayuDrishti/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_hourly(lat, lon, past_days=1, forecast_days=3):
    params = {
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "hourly": ",".join(FIELD_MAP),
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "UTC", "wind_speed_unit": "ms",
    }
    return _get_json(f"{API_URL}?{urllib.parse.urlencode(params)}")


def parse_hourly(payload):
    """Open-Meteo response -> list of dicts keyed by WeatherReading field names."""
    hourly = payload.get("hourly") or {}
    rows = []
    for i, stamp in enumerate(hourly.get("time", [])):
        row = {"timestamp": datetime.fromisoformat(stamp).replace(tzinfo=dt_timezone.utc)}
        for api_name, field in FIELD_MAP.items():
            series = hourly.get(api_name)
            row[field] = series[i] if series and i < len(series) else None
        rows.append(row)
    return rows


def fetch_and_store(locations=None, past_days=1, forecast_days=3, pause=0.15, all_locations=False):
    """
    Fetch weather for each airquality.Location and upsert into WeatherReading.
    Default: CPCB stations only. all_locations=True includes satellite-only points too.
    One failing location never stops the others.
    """
    from django.apps import apps

    from .models import WeatherReading

    if locations is None:
        qs = apps.get_model("airquality", "Location").objects.all()
        locations = qs if all_locations else qs.filter(is_cpcb_station=True)

    update_fields = list(FIELD_MAP.values()) + ["fetched_at"]
    summary = {"locations": 0, "rows": 0, "failed": 0}
    for loc in locations:
        try:
            rows = parse_hourly(fetch_hourly(loc.lat, loc.lon, past_days, forecast_days))
        except Exception:
            logger.exception("Open-Meteo fetch failed for %s", loc)
            summary["failed"] += 1
            continue
        objs = [WeatherReading(location=loc, source="open-meteo", **row) for row in rows]
        WeatherReading.objects.bulk_create(
            objs, batch_size=500, update_conflicts=True,
            unique_fields=["location", "timestamp", "source"], update_fields=update_fields)
        summary["locations"] += 1
        summary["rows"] += len(objs)
        time.sleep(pause)
    return summary
