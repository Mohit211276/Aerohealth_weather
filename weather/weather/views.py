import logging
from datetime import timedelta

from django.apps import apps
from django.conf import settings
from django.contrib.gis.db.models.functions import Distance
from django.contrib.gis.geos import Point, Polygon
from django.contrib.gis.measure import D
from django.db.models import Count, Max, OuterRef, Subquery, Sum
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
from rest_framework.views import APIView

from . import ml_services
from .ml_services import ModelNotAvailable
from .models import (AirQualityForecast, WeatherReading, SmokeDriftPrediction, aqi_category)
from .serializers import (AirQualityForecastSerializer, WeatherReadingSerializer,
                          PredictionRequestSerializer, SmokeDriftPredictionSerializer)

logger = logging.getLogger(__name__)

DELHI = (28.6139, 77.2090)  # lat, lon of the Delhi-NCR reference point
BAD_POINT = {"error": "valid lat and lon required"}


# ---- helpers ---------------------------------------------------------------

def _location_model():
    return apps.get_model("airquality", "Location")


def _point_params(request, default_radius=50.0):
    """Returns (Point, radius_km); raises ValueError/KeyError on bad input."""
    p = request.query_params
    lat, lon = float(p["lat"]), float(p["lon"])
    radius = float(p.get("radius_km", default_radius))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or radius <= 0:
        raise ValueError
    return Point(lon, lat, srid=4326), radius


def _hours_param(request, default, maximum=168):
    return min(max(int(request.query_params.get("hours", default)), 1), maximum)


def _bbox(request):
    """?bbox=min_lon,min_lat,max_lon,max_lat"""
    raw = request.query_params.get("bbox")
    if not raw:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in raw.split(","))
    except ValueError:
        return None
    return Polygon.from_bbox((x1, y1, x2, y2))


def _nearest_location(pt, location_ids, radius_km):
    return (_location_model().objects
            .filter(id__in=location_ids, point__dwithin=(pt, D(km=radius_km)))
            .annotate(distance=Distance("point", pt)).order_by("distance").first())


def _loc_info(loc):
    return {"id": loc.id, "name": loc.name, "distance_km": round(loc.distance.m / 1000, 1)}


def _newest_run_filter(qs):
    """Keep only rows from each location's most recent forecast run (rows of a run share run_at)."""
    newest = (AirQualityForecast.objects.filter(location=OuterRef("location"))
              .order_by("-run_at").values("run_at")[:1])
    return qs.filter(run_at=Subquery(newest))


class LimitedListMixin:
    """Plain-list responses (same style as the airquality app) with a ?limit= cap."""
    default_limit = 2000
    max_limit = 10000

    def list(self, request, *args, **kwargs):
        try:
            limit = int(request.query_params.get("limit", self.default_limit))
        except ValueError:
            limit = self.default_limit
        limit = max(1, min(limit, self.max_limit))
        qs = self.filter_queryset(self.get_queryset())[:limit]
        return Response(self.get_serializer(qs, many=True).data)


# ---- building blocks shared by the endpoints and /context/ -------------------

def _weather_at(pt, radius_km, hours):
    now = timezone.now()
    ids = (WeatherReading.objects.filter(timestamp__gte=now - timedelta(hours=3))
           .values_list("location_id", flat=True).distinct())
    loc = _nearest_location(pt, ids, radius_km)
    if loc is None:
        return None
    current = WeatherReading.objects.filter(location=loc, timestamp__lte=now).order_by("-timestamp").first()
    hourly = WeatherReading.objects.filter(location=loc, timestamp__gt=now,
                                         timestamp__lte=now + timedelta(hours=hours)).order_by("timestamp")
    return {"location": _loc_info(loc),
            "current": WeatherReadingSerializer(current).data if current else None,
            "hourly": WeatherReadingSerializer(hourly, many=True).data}


def _forecast_at(pt, radius_km, hours):
    now = timezone.now()
    ids = (AirQualityForecast.objects.filter(forecast_for__gte=now)
           .values_list("location_id", flat=True).distinct())
    loc = _nearest_location(pt, ids, radius_km)
    if loc is None:
        return None
    qs = _newest_run_filter(AirQualityForecast.objects.filter(
        location=loc, forecast_for__gte=now, forecast_for__lte=now + timedelta(hours=hours))
    ).order_by("forecast_for")
    return {"location": _loc_info(loc),
            "forecasts": AirQualityForecastSerializer(qs, many=True).data}


def _arrivals(pt, radius_km):
    rows = (SmokeDriftPrediction.objects.filter(point__dwithin=(pt, D(km=radius_km)))
            .values("hours_ahead").annotate(plumes=Count("id"), total_frp=Sum("hotspot__frp"))
            .order_by("hours_ahead"))
    arrivals = [{"hours_ahead": r["hours_ahead"], "plumes": r["plumes"],
                 "total_frp": round(r["total_frp"] or 0, 1)} for r in rows]
    return {"lat": pt.y, "lon": pt.x, "radius_km": radius_km, "smoke_expected": bool(arrivals),
            "first_arrival_hours": arrivals[0]["hours_ahead"] if arrivals else None,
            "arrivals": arrivals}


def _current_aqi(pt):
    """Same rule as the airquality /aqi/ endpoint: CPCB station within 10 km, else satellite grid."""
    GroundReading = apps.get_model("airquality", "GroundReading")
    AQIGridPoint = apps.get_model("airquality", "AQIGridPoint")
    now = timezone.now()
    station = (_location_model().objects.filter(is_cpcb_station=True, point__dwithin=(pt, D(km=10)))
               .annotate(distance=Distance("point", pt)).order_by("distance").first())
    if station:
        reading = (GroundReading.objects.filter(location=station, timestamp__gte=now - timedelta(hours=6),
                                                aqi__isnull=False).order_by("-timestamp").first())
        if reading:
            return {"aqi": round(reading.aqi), "category": aqi_category(reading.aqi),
                    "source": "cpcb_station", "station": station.name,
                    "distance_km": round(station.distance.m / 1000, 1), "timestamp": reading.timestamp}
    near = AQIGridPoint.objects.filter(point__dwithin=(pt, D(km=25)))
    latest_ts = near.order_by("-timestamp").values_list("timestamp", flat=True).first()
    if latest_ts:
        g = near.filter(timestamp=latest_ts).annotate(distance=Distance("point", pt)).order_by("distance").first()
        return {"aqi": round(g.aqi), "category": aqi_category(g.aqi), "source": "satellite_model",
                "distance_km": round(g.distance.m / 1000, 1), "timestamp": g.timestamp}
    return None


# ---- weather -----------------------------------------------------------

class WeatherReadingViewSet(LimitedListMixin, viewsets.ReadOnlyModelViewSet):
    """/readings/?location=1&from=..&to=..   (newest first)"""
    serializer_class = WeatherReadingSerializer

    def get_queryset(self):
        qs = WeatherReading.objects.all()
        p = self.request.query_params
        if p.get("location"):
            qs = qs.filter(location_id=p["location"])
        start = parse_datetime(p["from"]) if p.get("from") else None
        end = parse_datetime(p["to"]) if p.get("to") else None
        if start:
            qs = qs.filter(timestamp__gte=start)
        if end:
            qs = qs.filter(timestamp__lte=end)
        return qs

    @action(detail=False, url_path="at-point")
    def at_point(self, request):
        """/readings/at-point/?lat=..&lon=..&hours=48&radius_km=100 -> current + next hours (+ dispersion)."""
        try:
            pt, radius = _point_params(request, default_radius=100)
            hours = _hours_param(request, 48)
        except (KeyError, ValueError):
            return Response(BAD_POINT, status=400)
        data = _weather_at(pt, radius, hours)
        if data is None:
            return Response({"error": "no weather data near this location"}, status=404)
        return Response(data)


# ---- AQI forecasts ---------------------------------------------------------

class AirQualityForecastViewSet(LimitedListMixin, viewsets.ReadOnlyModelViewSet):
    """
    /forecasts/?location=1&from=..&to=..&model_version=..&latest_run=1
    The list shows future forecasts only, unless `from` is given.
    """
    serializer_class = AirQualityForecastSerializer

    def get_queryset(self):
        qs = AirQualityForecast.objects.select_related("location").order_by("location_id", "forecast_for")
        p = self.request.query_params
        if p.get("location"):
            qs = qs.filter(location_id=p["location"])
        if p.get("model_version"):
            qs = qs.filter(model_version=p["model_version"])
        start = parse_datetime(p["from"]) if p.get("from") else None
        end = parse_datetime(p["to"]) if p.get("to") else None
        if start:
            qs = qs.filter(forecast_for__gte=start)
        elif self.action == "list":
            qs = qs.filter(forecast_for__gte=timezone.now())
        if end:
            qs = qs.filter(forecast_for__lte=end)
        if p.get("latest_run") in ("1", "true"):
            qs = _newest_run_filter(qs)
        return qs

    @action(detail=False, url_path="at-point")
    def at_point(self, request):
        """/forecasts/at-point/?lat=..&lon=..&hours=72&radius_km=50 -> newest run for the nearest location."""
        try:
            pt, radius = _point_params(request, default_radius=50)
            hours = _hours_param(request, 72)
        except (KeyError, ValueError):
            return Response(BAD_POINT, status=400)
        data = _forecast_at(pt, radius, hours)
        if data is None:
            return Response({"error": "no forecast available near this location"}, status=404)
        return Response(data)

    @action(detail=False, methods=["post"], permission_classes=[IsAdminUser])
    def ingest(self, request):
        """Staff only: an external pipeline posts one forecast object or a list (one shared run_at)."""
        many = isinstance(request.data, list)
        serializer = AirQualityForecastSerializer(data=request.data, many=many)
        serializer.is_valid(raise_exception=True)
        serializer.save(run_at=timezone.now())
        return Response({"created": len(request.data) if many else 1}, status=status.HTTP_201_CREATED)


# ---- smoke drift (stubble burning, Delhi-NCR) --------------------------------

class SmokeDriftViewSet(LimitedListMixin, viewsets.ReadOnlyModelViewSet):
    """/smoke-drift/?hours_ahead=24&bbox=min_lon,min_lat,max_lon,max_lat&min_frp=5"""
    serializer_class = SmokeDriftPredictionSerializer

    def get_queryset(self):
        qs = SmokeDriftPrediction.objects.select_related("hotspot").order_by("hours_ahead", "id")
        p = self.request.query_params
        if p.get("hours_ahead"):
            qs = qs.filter(hours_ahead=p["hours_ahead"])
        if p.get("min_frp"):
            qs = qs.filter(hotspot__frp__gte=float(p["min_frp"]))
        box = _bbox(self.request)
        if box:
            qs = qs.filter(point__intersects=box)
        return qs

    @action(detail=False)
    def arrivals(self, request):
        """/smoke-drift/arrivals/?lat=..&lon=..&radius_km=50 -> will smoke reach this point, and when?"""
        try:
            pt, radius = _point_params(request, default_radius=50)
        except (KeyError, ValueError):
            return Response(BAD_POINT, status=400)
        return Response(_arrivals(pt, radius))

    @action(detail=False)
    def ncr(self, request):
        """/smoke-drift/ncr/?radius_km=75 -> smoke heading for Delhi-NCR (reference point: central Delhi)."""
        try:
            radius = float(request.query_params.get("radius_km", 75))
            if radius <= 0:
                raise ValueError
        except ValueError:
            return Response({"error": "radius_km must be a positive number"}, status=400)
        data = _arrivals(Point(DELHI[1], DELHI[0], srid=4326), radius)
        data["region"] = "Delhi-NCR"
        return Response(data)


# ---- one-call summary for the app -------------------------------------------

class ContextView(APIView):
    """
    /context/?lat=..&lon=.. -> everything the app shows for a place in ONE call:
    current AQI, weather + dispersion, 72h AQI forecast and incoming stubble smoke.
    Any part with no data is null (the endpoint still answers 200).
    """

    def get(self, request):
        try:
            pt, _ = _point_params(request)
        except (KeyError, ValueError):
            return Response(BAD_POINT, status=400)
        return Response({
            "lat": pt.y, "lon": pt.x,
            "aqi": _current_aqi(pt),
            "weather": _weather_at(pt, 100, 24),
            "forecast": _forecast_at(pt, 50, 72),
            "smoke": _arrivals(pt, 50),
        })


# ---- prediction + status ---------------------------------------------------

class PredictAQIView(APIView):
    """POST pollutant readings + lat/lon + month/day_of_week -> predicted AQI."""

    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            aqi = ml_services.predict_aqi(serializer.validated_data)
        except ModelNotAvailable as exc:
            return Response({"error": "AQI model is not trained yet", "expected_path": str(exc)}, status=503)
        except Exception:
            logger.exception("Prediction failed")
            return Response({"error": "prediction failed"}, status=500)
        return Response({"predicted_aqi": round(aqi), "category": aqi_category(aqi),
                         "model_version": getattr(settings, "AQI_MODEL_VERSION", "v1")})


class StatusView(APIView):
    """/status/ -> data freshness and model availability (handy for debugging and the demo)."""

    def get(self, request):
        mr = WeatherReading.objects.aggregate(n=Count("id"), latest=Max("timestamp"), fetched=Max("fetched_at"))
        fc = AirQualityForecast.objects.aggregate(n=Count("id"), latest=Max("run_at"))
        sd = SmokeDriftPrediction.objects.aggregate(n=Count("id"), latest=Max("generated_at"))
        aqi_path, fc_path = ml_services.model_path("aqi"), ml_services.model_path("forecast")
        return Response({
            "weather_readings": {"count": mr["n"], "latest_timestamp": mr["latest"], "last_fetched": mr["fetched"]},
            "aqi_forecasts": {"count": fc["n"], "last_run": fc["latest"]},
            "smoke_drift": {"count": sd["n"], "last_generated": sd["latest"]},
            "aqi_model": {"path": str(aqi_path), "exists": aqi_path.exists()},
            "forecast_model": {"path": str(fc_path), "exists": fc_path.exists()},
        })
