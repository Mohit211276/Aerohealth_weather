import csv
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest import mock

from django.apps import apps
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import SimpleTestCase, override_settings
from django.urls import include, path
from django.utils import timezone
from rest_framework.test import APITestCase

from . import ml_services, weather_client, smoke
from . import urls as weather_urls
from .geo import destination_point, downwind_bearing
from .models import AirQualityForecast, WeatherReading, SmokeDriftPrediction

# Tests use their own URLconf, so they do not depend on how the project urls.py includes this app.
urlpatterns = [path("api/weather/", include(weather_urls))]

PREDICT = dict(pm25=180, pm10=300, no=20, no2=60, nox=80, nh3=30, co=1.5, so2=15,
               o3=40, benzene=3, toluene=5, lat=28.65, lon=77.32, month=11, day_of_week=2)


def hour_now():
    return timezone.now().replace(minute=0, second=0, microsecond=0)


def fake_payload(start, n=5):
    times = [(start + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(n)]
    return {"hourly": {
        "time": times, "temperature_2m": [30.0] * n, "relative_humidity_2m": [50] * n,
        "surface_pressure": [1000.0] * n, "precipitation": [0.0] * n,
        "wind_speed_10m": [4.0] * n, "wind_direction_10m": [270] * n,
        "boundary_layer_height": [800.0] * n}}


class FakeModel:
    def __init__(self, value=None):
        self.value = value

    def predict(self, rows):
        return [self.value if self.value is not None else 200.0 + i for i in range(len(rows))]


class GeoTests(SimpleTestCase):
    def test_downwind_bearing(self):
        self.assertEqual(downwind_bearing(270), 90)
        self.assertEqual(downwind_bearing(10), 190)

    def test_destination_east_along_equator(self):
        lat, lon = destination_point(0, 0, 90, 111.195)
        self.assertAlmostEqual(lat, 0, places=3)
        self.assertAlmostEqual(lon, 1.0, places=2)

    def test_api_url_points_at_the_real_open_meteo_service(self):
        self.assertTrue(weather_client.API_URL.startswith("https://api.open-meteo.com/"))

    def test_parse_hourly(self):
        rows = weather_client.parse_hourly(fake_payload(hour_now(), 3))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["wind_speed_ms"], 4.0)
        self.assertIsNotNone(rows[0]["timestamp"].tzinfo)


@override_settings(ROOT_URLCONF=__name__)
class WeatherAPITests(APITestCase):
    def setUp(self):
        Location = apps.get_model("airquality", "Location")
        self.delhi = Location.objects.create(name="Anand Vihar", state="Delhi", lat=28.65, lon=77.32,
                                             is_cpcb_station=True)
        self.ludhiana = Location.objects.create(name="Ludhiana", state="Punjab", lat=30.90, lon=75.85,
                                                is_cpcb_station=True)
        self.t0 = hour_now()
        for loc in (self.delhi, self.ludhiana):
            for h in range(-2, 75):
                WeatherReading.objects.create(
                    location=loc, timestamp=self.t0 + timedelta(hours=h), temperature_c=30,
                    wind_speed_ms=5.0, wind_direction_deg=270.0, boundary_layer_m=1000.0)

    def _ground(self, loc, aqi, hours_ago):
        GroundReading = apps.get_model("airquality", "GroundReading")
        return GroundReading.objects.create(location=loc, timestamp=self.t0 - timedelta(hours=hours_ago),
                                            aqi=aqi, pm25=aqi * 0.6)

    def _fire(self, lat=30.90, lon=75.80, **kw):
        FireHotspot = apps.get_model("airquality", "FireHotspot")
        return FireHotspot.objects.create(lat=lat, lon=lon, acq_datetime=timezone.now(), frp=12.0,
                                          confidence="h", satellite="VIIRS", **kw)

    # --- readings + dispersion
    def test_ventilation_and_dispersion(self):
        r = WeatherReading.objects.filter(location=self.delhi).first()
        self.assertEqual(r.ventilation_index, 5000.0)   # 5 m/s * 1000 m
        self.assertEqual(r.dispersion, "poor")          # below 6000 m2/s

    def test_readings_at_point(self):
        r = self.client.get("/api/weather/readings/at-point/?lat=28.65&lon=77.32&hours=24")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["location"]["name"], "Anand Vihar")
        self.assertEqual(body["current"]["dispersion"], "poor")
        self.assertEqual(len(body["hourly"]), 24)

    def test_readings_far_away_404_and_bad_params_400(self):
        self.assertEqual(self.client.get("/api/weather/readings/at-point/?lat=8.0&lon=77.0").status_code, 404)
        self.assertEqual(self.client.get("/api/weather/readings/at-point/?lat=x").status_code, 400)

    def test_readings_list_limit(self):
        r = self.client.get(f"/api/weather/readings/?location={self.delhi.id}&limit=5")
        self.assertEqual(len(r.json()), 5)

    # --- open-meteo fetch
    def test_fetch_and_store_is_idempotent(self):
        WeatherReading.objects.all().delete()
        with mock.patch.object(weather_client, "_get_json", return_value=fake_payload(self.t0, 5)):
            s1 = weather_client.fetch_and_store(pause=0)
            s2 = weather_client.fetch_and_store(pause=0)
        self.assertEqual(s1["locations"], 2)
        self.assertEqual(s2["failed"], 0)
        self.assertEqual(WeatherReading.objects.count(), 10)

    def test_fetch_failure_does_not_stop_others(self):
        with mock.patch.object(weather_client, "_get_json", side_effect=OSError("network down")):
            self.assertEqual(weather_client.fetch_and_store(pause=0)["failed"], 2)

    def test_fetch_skips_non_cpcb_unless_asked(self):
        apps.get_model("airquality", "Location").objects.create(name="Grid pt", lat=20, lon=78)
        WeatherReading.objects.all().delete()
        with mock.patch.object(weather_client, "_get_json", return_value=fake_payload(self.t0, 2)):
            self.assertEqual(weather_client.fetch_and_store(pause=0)["locations"], 2)
            self.assertEqual(weather_client.fetch_and_store(pause=0, all_locations=True)["locations"], 3)

    # --- smoke drift
    def test_smoke_drifts_downwind(self):
        fire = self._fire()
        self.assertEqual(smoke.generate_smoke_drift()["hotspots"], 1)
        p6 = SmokeDriftPrediction.objects.get(hotspot=fire, hours_ahead=6)
        # wind from the west at 5 m/s for 6 h = 108 km east (~1.1 degrees of longitude here)
        self.assertGreater(p6.lon - fire.lon, 0.9)
        self.assertLess(p6.lon - fire.lon, 1.4)
        self.assertLess(abs(p6.lat - fire.lat), 0.1)
        self.assertAlmostEqual(p6.distance_km, 108, delta=1)
        self.assertEqual(SmokeDriftPrediction.objects.filter(hotspot=fire).count(), 4)
        smoke.generate_smoke_drift()  # re-running replaces rows, never duplicates
        self.assertEqual(SmokeDriftPrediction.objects.filter(hotspot=fire).count(), 4)

    def test_smoke_follows_the_wind_of_the_nearest_station_along_its_path(self):
        Location = apps.get_model("airquality", "Location")
        east = Location.objects.create(name="East", lat=30.90, lon=77.00, is_cpcb_station=True)
        WeatherReading.objects.filter(location=self.ludhiana).delete()
        WeatherReading.objects.filter(location=self.delhi).delete()
        for h in range(0, 49):
            WeatherReading.objects.create(location=self.ludhiana, timestamp=self.t0 + timedelta(hours=h),
                                        wind_speed_ms=5.0, wind_direction_deg=270.0)   # smoke goes east
            WeatherReading.objects.create(location=east, timestamp=self.t0 + timedelta(hours=h),
                                        wind_speed_ms=5.0, wind_direction_deg=180.0)   # smoke goes north
        fire = self._fire()
        smoke.generate_smoke_drift()
        p12 = SmokeDriftPrediction.objects.get(hotspot=fire, hours_ahead=12)
        # ~4 h east inside the first station's wind, then north once nearer the second station
        self.assertGreater(p12.lat, 31.9)
        self.assertGreater(p12.lon, 76.4)
        self.assertLess(p12.lon, 76.7)

    def test_smoke_ignores_hotspots_outside_bbox(self):
        self._fire(lat=12.0, lon=78.0)   # far south of the Punjab/Haryana/Delhi default box
        self.assertEqual(smoke.generate_smoke_drift()["hotspots"], 0)

    def test_smoke_endpoints(self):
        self._fire()
        smoke.generate_smoke_drift()
        self.assertEqual(len(self.client.get("/api/weather/smoke-drift/?hours_ahead=24").json()), 1)
        # 24 h * 5 m/s = 432 km east of 75.8E lands around 80E
        body = self.client.get("/api/weather/smoke-drift/arrivals/?lat=30.9&lon=79.5&radius_km=200").json()
        self.assertTrue(body["smoke_expected"])
        self.assertIn(24, [a["hours_ahead"] for a in body["arrivals"]])
        far = self.client.get("/api/weather/smoke-drift/arrivals/?lat=10&lon=77&radius_km=50").json()
        self.assertFalse(far["smoke_expected"])
        self.assertIsNone(far["first_arrival_hours"])

    def test_smoke_ncr_endpoint(self):
        r = self.client.get("/api/weather/smoke-drift/ncr/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["region"], "Delhi-NCR")
        self.assertFalse(r.json()["smoke_expected"])
        self.assertEqual(self.client.get("/api/weather/smoke-drift/ncr/?radius_km=-3").status_code, 400)

    def test_smoke_without_weather_reports_reason(self):
        WeatherReading.objects.all().delete()
        self._fire()
        self.assertIn("reason", smoke.generate_smoke_drift())

    # --- forecasts
    def _make_forecasts(self):
        now = timezone.now()
        old = AirQualityForecast.objects.create(location=self.delhi, prediction=111, horizon_hours=6,
                                                forecast_for=now + timedelta(hours=6), model_version="v0",
                                                run_at=now - timedelta(hours=3))
        run = now
        for h, v in ((6, 250), (24, 310)):
            AirQualityForecast.objects.create(location=self.delhi, prediction=v, horizon_hours=h, run_at=run,
                                              forecast_for=now + timedelta(hours=h), model_version="v1")
        return old

    def test_forecast_list_and_at_point_use_newest_run(self):
        self._make_forecasts()
        self.assertEqual(len(self.client.get("/api/weather/forecasts/").json()), 3)
        self.assertEqual(len(self.client.get("/api/weather/forecasts/?latest_run=1").json()), 2)
        body = self.client.get("/api/weather/forecasts/at-point/?lat=28.65&lon=77.32").json()
        self.assertEqual([f["prediction"] for f in body["forecasts"]], [250, 310])
        self.assertEqual(body["forecasts"][1]["category"], "Very Poor")
        self.assertEqual(self.client.get("/api/weather/forecasts/at-point/?lat=19&lon=73").status_code, 404)
        self.assertEqual(self.client.get("/api/weather/forecasts/at-point/?lat=x").status_code, 400)

    def test_forecast_ingest_requires_staff_and_shares_run(self):
        base = timezone.now()
        items = [{"location": self.delhi.id, "prediction": 200 + h, "model_version": "v1",
                  "forecast_for": (base + timedelta(hours=h)).isoformat()} for h in (24, 48)]
        self.assertIn(self.client.post("/api/weather/forecasts/ingest/", items, format="json").status_code, (401, 403))
        self.client.force_authenticate(get_user_model().objects.create_user("s", password="x", is_staff=True))
        self.assertEqual(self.client.post("/api/weather/forecasts/ingest/", items, format="json").status_code, 201)
        rows = AirQualityForecast.objects.order_by("forecast_for")
        self.assertEqual(len({r.run_at for r in rows}), 1)           # one run
        self.assertEqual([r.horizon_hours for r in rows], [24, 48])  # derived from forecast_for - run_at
        self.client.force_authenticate(None)
        self.assertEqual(len(self.client.get("/api/weather/forecasts/at-point/?lat=28.65&lon=77.32").json()["forecasts"]), 2)

    # --- forecast generation
    @override_settings(AQI_FORECAST_MODEL_PATH="/nonexistent/forecast.joblib")
    def test_generate_forecasts_without_model(self):
        self._ground(self.delhi, 250, 1)
        self.assertEqual(ml_services.generate_forecasts()["status"], "no_model")
        self.assertEqual(AirQualityForecast.objects.count(), 0)

    def test_generate_forecasts_with_model(self):
        self._ground(self.delhi, 250, 1)
        with mock.patch.object(ml_services, "get_model", return_value=FakeModel()):
            result = ml_services.generate_forecasts()
        self.assertEqual(result, {"status": "ok", "created": 3, "model_version": "lgbm-v1"})
        rows = AirQualityForecast.objects.order_by("horizon_hours")
        self.assertEqual([r.horizon_hours for r in rows], [24, 48, 72])
        self.assertEqual(len({r.run_at for r in rows}), 1)

    def test_model_receives_features_in_documented_order(self):
        self._ground(self.delhi, 250, 1)
        seen = {}

        class Spy:
            def predict(self, rows):
                seen["rows"] = rows
                return [100.0] * len(rows)
        with mock.patch.object(ml_services, "get_model", return_value=Spy()):
            ml_services.generate_forecasts()
        row = dict(zip(ml_services.FORECAST_FEATURES, seen["rows"][0]))
        self.assertEqual(row["current_aqi"], 250)
        self.assertEqual(row["wind_speed"], 5.0)
        self.assertEqual(row["horizon_h"], 24)
        self.assertEqual(seen["rows"].shape[1], len(ml_services.FORECAST_FEATURES))

    @override_settings(AQI_FORECAST_MODEL_PATH="/nonexistent/forecast.joblib", WEATHER_ALLOW_BASELINE_FORECAST=True)
    def test_persistence_baseline_is_labelled(self):
        self._ground(self.delhi, 250, 1)
        self.assertEqual(ml_services.generate_forecasts()["model_version"], "persistence")
        self.assertEqual({r.prediction for r in AirQualityForecast.objects.all()}, {250.0})

    # --- training export
    def test_training_rows_and_csv(self):
        WeatherReading.objects.all().delete()
        for i in range(6):   # hourly AQI and weather for the last 6 hours
            ts = self.t0 - timedelta(hours=i)
            self._ground(self.delhi, 100 + i, i)
            WeatherReading.objects.create(location=self.delhi, timestamp=ts, wind_speed_ms=3.0,
                                        wind_direction_deg=90.0, temperature_c=25, boundary_layer_m=500)
        rows = ml_services.training_rows(days=1, horizons=(1, 2))
        self.assertEqual(len(rows), 9)   # 5 pairs at +1h and 4 at +2h
        self.assertTrue(all(set(ml_services.TRAINING_COLUMNS) == set(r) for r in rows))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train.csv"
            call_command("weather_jobs", "export-training", "--days", "1", "--horizons", "1,2", "--out", str(out))
            with open(out, newline="") as fh:
                self.assertEqual(len(list(csv.DictReader(fh))), 9)

    # --- context, predict, status
    def test_context_combines_everything(self):
        self._ground(self.delhi, 250, 1)
        self._make_forecasts()
        body = self.client.get("/api/weather/context/?lat=28.65&lon=77.32").json()
        self.assertEqual(body["aqi"]["aqi"], 250)
        self.assertEqual(body["aqi"]["source"], "cpcb_station")
        self.assertEqual(body["weather"]["location"]["name"], "Anand Vihar")
        self.assertEqual(len(body["forecast"]["forecasts"]), 2)
        self.assertFalse(body["smoke"]["smoke_expected"])

    def test_context_with_no_data_is_200_with_nulls(self):
        body = self.client.get("/api/weather/context/?lat=8&lon=77").json()
        self.assertIsNone(body["aqi"])
        self.assertIsNone(body["weather"])
        self.assertIsNone(body["forecast"])
        self.assertEqual(self.client.get("/api/weather/context/").status_code, 400)

    @override_settings(AQI_MODEL_PATH="/nonexistent/model.joblib")
    def test_predict_without_model_503(self):
        self.assertEqual(self.client.post("/api/weather/predict/", PREDICT, format="json").status_code, 503)

    def test_predict_validation_400(self):
        self.assertEqual(self.client.post("/api/weather/predict/", dict(PREDICT, month=13),
                                          format="json").status_code, 400)

    def test_predict_with_model(self):
        with mock.patch.object(ml_services, "get_model", return_value=FakeModel(275.4)):
            r = self.client.post("/api/weather/predict/", PREDICT, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.json()["predicted_aqi"], r.json()["category"]), (275, "Poor"))

    def test_status(self):
        r = self.client.get("/api/weather/status/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["weather_readings"]["count"], 154)

    def test_celery_tasks_have_unique_names(self):
        from . import tasks
        names = {tasks.fetch_weather.name, tasks.generate_smoke_drift.name,
                 tasks.generate_forecasts.name, tasks.purge_old.name}
        self.assertTrue(all(n.startswith("weather.") for n in names))
        self.assertEqual(len(names), 4)
