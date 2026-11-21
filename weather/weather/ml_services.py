"""
Model loading, forecast generation and training-data export.
Train and serve share the SAME feature builder, so the column order can never drift.
"""
import csv
import logging
import math
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
from django.apps import apps
from django.conf import settings
from django.contrib.gis.geos import Point
from django.contrib.gis.measure import D
from django.utils import timezone

from .models import AirQualityForecast, WeatherReading

logger = logging.getLogger(__name__)

# Column order the pollutant -> AQI model (POST /predict/) must be trained with.
FEATURES = ["pm25", "pm10", "no", "no2", "nox", "nh3", "co", "so2", "o3",
            "benzene", "toluene", "lat", "lon", "month", "day_of_week"]

# Column order the 24-72h forecast model (LightGBM) must be trained with.
# Weather is taken at the TARGET hour; fire values are the 24h before the forecast time, within 100 km.
FORECAST_FEATURES = ["current_aqi", "wind_speed", "wind_dir_sin", "wind_dir_cos", "temperature",
                     "humidity", "boundary_layer", "fires_100km_24h", "frp_100km_24h",
                     "hour", "month", "day_of_week", "horizon_h"]

FORECAST_HORIZONS = (24, 48, 72)
FIRE_RADIUS_KM = 100


class ModelNotAvailable(Exception):
    pass


# ---- model files -----------------------------------------------------------

def model_path(kind="aqi"):
    base = Path(settings.BASE_DIR) / "datascience" / "models"
    if kind == "forecast":
        return Path(getattr(settings, "AQI_FORECAST_MODEL_PATH", base / "aqi_forecast_model.joblib"))
    return Path(getattr(settings, "AQI_MODEL_PATH", base / "aqi_model.joblib"))


_cache = {}


def get_model(path):
    """Load a joblib model; reload automatically when the file changes. None if the file is missing."""
    path = Path(path)
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    cached = _cache.get(str(path))
    if cached is None or cached[0] != mtime:
        import joblib  # lazy: the API still starts where ML packages are not installed
        _cache[str(path)] = (mtime, joblib.load(path))
    return _cache[str(path)][1]


def predict_aqi(data):
    model = get_model(model_path("aqi"))
    if model is None:
        raise ModelNotAvailable(str(model_path("aqi")))
    row = np.array([[data[f] for f in FEATURES]], dtype=float)
    return min(max(float(model.predict(row)[0]), 0.0), 500.0)


# ---- forecast features -----------------------------------------------------

def _nan_if_none(v):
    return float("nan") if v is None else v


def build_forecast_features(current_aqi, w, fires_n, fires_frp, target, horizon_h):
    """One feature row in FORECAST_FEATURES order. w = WeatherReading at the target hour."""
    rad = math.radians(w.wind_direction_deg)
    return [current_aqi, w.wind_speed_ms, math.sin(rad), math.cos(rad),
            _nan_if_none(w.temperature_c), _nan_if_none(w.humidity_pct),
            _nan_if_none(w.boundary_layer_m), fires_n, fires_frp,
            target.hour, target.month, target.weekday(), horizon_h]


def _has_wind(w):
    return w is not None and w.wind_speed_ms is not None and w.wind_direction_deg is not None


def _floor_hour(ts):
    return ts.replace(minute=0, second=0, microsecond=0)


def _fire_stats(fires, at_time):
    """fires = list of (acq_datetime, frp). Count and total FRP in the 24h up to at_time."""
    start = at_time - timedelta(hours=24)
    sel = [frp or 0.0 for ts, frp in fires if start <= ts <= at_time]
    return len(sel), float(sum(sel))


# ---- forecast generation ---------------------------------------------------

def generate_forecasts(horizons=FORECAST_HORIZONS, allow_baseline=None):
    """
    Forecast AQI at every CPCB station that has a recent reading, for each horizon, and save
    AirQualityForecast rows (one shared run_at per call).
    With no trained model: returns {"status": "no_model"} unless the persistence baseline is allowed
    (settings.WEATHER_ALLOW_BASELINE_FORECAST or allow_baseline=True); baseline rows are labelled
    model_version="persistence" so they are never mistaken for real model output.
    """
    GroundReading = apps.get_model("airquality", "GroundReading")
    FireHotspot = apps.get_model("airquality", "FireHotspot")

    path = model_path("forecast")
    model = get_model(path)
    if allow_baseline is None:
        allow_baseline = getattr(settings, "WEATHER_ALLOW_BASELINE_FORECAST", False)
    if model is None and not allow_baseline:
        return {"status": "no_model", "expected_path": str(path)}

    now = timezone.now()
    hour0 = _floor_hour(now)
    targets = {h: hour0 + timedelta(hours=h) for h in horizons}
    latest = (GroundReading.objects
              .filter(timestamp__gte=now - timedelta(hours=6), aqi__isnull=False,
                      location__is_cpcb_station=True)
              .select_related("location").order_by("location_id", "-timestamp")
              .distinct("location_id"))

    rows, meta = [], []
    for reading in latest:
        loc = reading.location
        weather = {w.timestamp: w for w in WeatherReading.objects.filter(
            location=loc, timestamp__in=list(targets.values()))}
        fires = list(FireHotspot.objects.filter(
            acq_datetime__gte=now - timedelta(hours=24),
            point__dwithin=(Point(loc.lon, loc.lat, srid=4326), D(km=FIRE_RADIUS_KM)),
        ).values_list("acq_datetime", "frp"))
        n_fires, frp = _fire_stats(fires, now)
        for h, target in targets.items():
            if model is not None:
                w = weather.get(target)
                if not _has_wind(w):
                    continue
                rows.append(build_forecast_features(reading.aqi, w, n_fires, frp, target, h))
            meta.append((loc, target, h, reading.aqi))
    if not meta:
        return {"status": "no_features", "created": 0}

    if model is not None:
        preds = model.predict(np.array(rows, dtype=float))
        version = getattr(settings, "AQI_FORECAST_MODEL_VERSION", "lgbm-v1")
    else:
        preds = [aqi for _, _, _, aqi in meta]
        version = "persistence"
    AirQualityForecast.objects.bulk_create([
        AirQualityForecast(location=loc, run_at=now, forecast_for=target, horizon_hours=h,
                           prediction=min(max(float(p), 0.0), 500.0), model_version=version)
        for (loc, target, h, _), p in zip(meta, preds)])
    return {"status": "ok", "created": len(meta), "model_version": version}


# ---- training data export --------------------------------------------------

TRAINING_COLUMNS = ["location_id", "as_of"] + FORECAST_FEATURES + ["target_aqi"]


def training_rows(days=30, horizons=FORECAST_HORIZONS):
    """
    Build supervised rows from history for the forecast model: for each station-hour with an AQI,
    and each horizon, features = (AQI then, weather/fires at the target hour) and target = AQI at the
    target hour. Needs GroundReading (CPCB), WeatherReading (fetch --past-days N) and FireHotspot.
    """
    GroundReading = apps.get_model("airquality", "GroundReading")
    FireHotspot = apps.get_model("airquality", "FireHotspot")
    Location = apps.get_model("airquality", "Location")

    since = timezone.now() - timedelta(days=days)
    out = []
    for loc in Location.objects.filter(is_cpcb_station=True):
        buckets = defaultdict(list)
        for ts, aqi in GroundReading.objects.filter(
                location=loc, timestamp__gte=since, aqi__isnull=False).values_list("timestamp", "aqi"):
            buckets[_floor_hour(ts)].append(aqi)
        aqi_by_hour = {h: sum(v) / len(v) for h, v in buckets.items()}
        if not aqi_by_hour:
            continue
        weather = {w.timestamp: w for w in WeatherReading.objects.filter(location=loc, timestamp__gte=since)}
        fires = list(FireHotspot.objects.filter(
            acq_datetime__gte=since - timedelta(hours=24),
            point__dwithin=(Point(loc.lon, loc.lat, srid=4326), D(km=FIRE_RADIUS_KM)),
        ).values_list("acq_datetime", "frp"))
        for as_of, aqi_now in sorted(aqi_by_hour.items()):
            for h in horizons:
                target = as_of + timedelta(hours=h)
                w = weather.get(target)
                if target not in aqi_by_hour or not _has_wind(w):
                    continue
                n_fires, frp = _fire_stats(fires, target)
                feats = build_forecast_features(aqi_now, w, n_fires, frp, target, h)
                row = {"location_id": loc.id, "as_of": as_of.isoformat(), "target_aqi": round(aqi_by_hour[target], 2)}
                row.update(dict(zip(FORECAST_FEATURES, feats)))
                out.append(row)
    return out


def export_training_csv(path=None, days=30, horizons=FORECAST_HORIZONS):
    path = Path(path or Path(settings.BASE_DIR) / "datascience" / "data" / "processed" / "weather_forecast_training.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = training_rows(days=days, horizons=horizons)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRAINING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "path": str(path)}
