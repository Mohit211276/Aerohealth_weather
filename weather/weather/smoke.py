"""
Stubble-burning smoke drift for Delhi-NCR: advect FIRMS hotspots downwind using Open-Meteo winds.

Every hotspot is a smoke "parcel". Each hour the parcel moves with the wind of the NEAREST weather
station to its CURRENT position (so a plume can pass from one station's wind field into another's).
All parcels are moved together with numpy, so thousands of hotspots are cheap.
"""
import logging
from datetime import timedelta

import numpy as np
from django.apps import apps
from django.conf import settings
from django.contrib.gis.geos import Point
from django.db import transaction
from django.utils import timezone

from .geo import destination_points, downwind_bearing, haversine_km
from .models import WeatherReading, SmokeDriftPrediction

logger = logging.getLogger(__name__)

HORIZONS = (6, 12, 24, 48)
DEFAULT_BBOX = (73.5, 27.5, 78.0, 32.8)  # min_lon, min_lat, max_lon, max_lat: Punjab/Haryana/Delhi


def _fill_time_gaps(a):
    """Forward-fill then back-fill NaNs along axis 0 (time)."""
    a = a.copy()
    for k in range(1, a.shape[0]):
        m = np.isnan(a[k])
        a[k][m] = a[k - 1][m]
    for k in range(a.shape[0] - 2, -1, -1):
        m = np.isnan(a[k])
        a[k][m] = a[k + 1][m]
    return a


def generate_smoke_drift(horizons=HORIZONS, max_hotspot_age_h=12, bbox=None, max_weather_km=300):
    """
    Predict where smoke from recent hotspots will be at each horizon (hours from the current hour).
    bbox = (min_lon, min_lat, max_lon, max_lat); default comes from settings.WEATHER_SMOKE_BBOX.
    """
    FireHotspot = apps.get_model("airquality", "FireHotspot")
    Location = apps.get_model("airquality", "Location")

    bbox = bbox or getattr(settings, "WEATHER_SMOKE_BBOX", DEFAULT_BBOX)
    now = timezone.now()
    t0 = now.replace(minute=0, second=0, microsecond=0)
    n_hours = max(horizons)

    fires = (FireHotspot.objects.filter(acq_datetime__gte=now - timedelta(hours=max_hotspot_age_h))
             .exclude(confidence__in=["l", "L", "low", "Low"]))
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        fires = fires.filter(lon__gte=min_lon, lon__lte=max_lon, lat__gte=min_lat, lat__lte=max_lat)
    fires = list(fires.values("id", "lat", "lon"))
    if not fires:
        return {"hotspots": 0, "predictions": 0, "skipped": 0}

    rows = list(WeatherReading.objects.filter(
        timestamp__gte=t0, timestamp__lt=t0 + timedelta(hours=n_hours),
        wind_speed_ms__isnull=False, wind_direction_deg__isnull=False,
    ).values_list("location_id", "timestamp", "wind_speed_ms", "wind_direction_deg"))
    if not rows:
        return {"hotspots": len(fires), "predictions": 0, "skipped": len(fires),
                "reason": "no weather data; run: python manage.py weather_jobs fetch"}

    station_ids = sorted({r[0] for r in rows})
    col = {sid: i for i, sid in enumerate(station_ids)}
    speed = np.full((n_hours, len(station_ids)), np.nan)
    wdir = np.full((n_hours, len(station_ids)), np.nan)
    for sid, ts, spd, d in rows:
        k = int((ts - t0).total_seconds() // 3600)
        if 0 <= k < n_hours:
            speed[k, col[sid]], wdir[k, col[sid]] = spd, d
    speed, wdir = _fill_time_gaps(speed), _fill_time_gaps(wdir)
    keep = ~np.isnan(speed).any(axis=0)
    if not keep.any():
        return {"hotspots": len(fires), "predictions": 0, "skipped": len(fires),
                "reason": "weather rows have no usable wind"}
    station_ids = [s for s, k in zip(station_ids, keep) if k]
    speed, wdir = speed[:, keep], wdir[:, keep]

    coords = {i: (la, lo) for i, la, lo in
              Location.objects.filter(id__in=station_ids).values_list("id", "lat", "lon")}
    s_lat = np.array([coords[s][0] for s in station_ids])
    s_lon = np.array([coords[s][1] for s in station_ids])

    lat = np.array([f["lat"] for f in fires], dtype=float)
    lon = np.array([f["lon"] for f in fires], dtype=float)
    first = haversine_km(lat[:, None], lon[:, None], s_lat[None, :], s_lon[None, :]).min(axis=1)
    valid = first <= max_weather_km
    total = np.zeros(len(fires))
    speed_sum = np.zeros(len(fires))
    records = {}

    for k in range(n_hours):
        nearest = haversine_km(lat[:, None], lon[:, None], s_lat[None, :], s_lon[None, :]).argmin(axis=1)
        spd, wd = speed[k, nearest], wdir[k, nearest]
        step_km = spd * 3.6  # m/s over one hour -> km
        lat, lon = destination_points(lat, lon, downwind_bearing(wd), step_km)
        total += step_km
        speed_sum += spd
        if (k + 1) in horizons:
            records[k + 1] = (lat.copy(), lon.copy(), total.copy(), speed_sum / (k + 1), wd.copy())

    preds, used_ids = [], []
    for i, fire in enumerate(fires):
        if not valid[i]:
            continue
        used_ids.append(fire["id"])
        for h, (la, lo, dist, mean_speed, wd) in records.items():
            preds.append(SmokeDriftPrediction(
                hotspot_id=fire["id"], hours_ahead=h, predicted_for=t0 + timedelta(hours=h),
                lat=float(la[i]), lon=float(lo[i]), point=Point(float(lo[i]), float(la[i]), srid=4326),
                distance_km=round(float(dist[i]), 2), wind_speed_ms=round(float(mean_speed[i]), 2),
                wind_direction_deg=round(float(wd[i]), 1)))

    with transaction.atomic():
        SmokeDriftPrediction.objects.filter(hotspot_id__in=used_ids).delete()
        SmokeDriftPrediction.objects.filter(predicted_for__lt=now - timedelta(days=1)).delete()
        SmokeDriftPrediction.objects.bulk_create(preds, batch_size=1000)
    return {"hotspots": len(used_ids), "predictions": len(preds), "skipped": int((~valid).sum())}
