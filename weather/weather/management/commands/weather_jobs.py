from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone


class Command(BaseCommand):
    help = "Run weather jobs without Celery: fetch | drift | forecast | export-training | purge | all"

    def add_arguments(self, parser):
        parser.add_argument("job", choices=["fetch", "drift", "forecast", "export-training", "purge", "all"])
        parser.add_argument("--past-days", type=int, default=1, help="fetch: history to load (max 92)")
        parser.add_argument("--forecast-days", type=int, default=3)
        parser.add_argument("--all-locations", action="store_true", help="fetch: include non-CPCB locations")
        parser.add_argument("--days", type=int, default=30, help="export-training: history window")
        parser.add_argument("--out", default=None, help="export-training: output CSV path")
        parser.add_argument("--horizons", default="24,48,72",
                            help="export-training: forecast horizons in hours, e.g. 24,48,72 (use 1,2,3 while data is short)")

    def handle(self, *args, **opts):
        job = opts["job"]
        if job in ("fetch", "all"):
            from ...weather_client import fetch_and_store
            self.stdout.write("fetch:    %s" % fetch_and_store(
                past_days=opts["past_days"], forecast_days=opts["forecast_days"],
                all_locations=opts["all_locations"]))
        if job in ("drift", "all"):
            from ...smoke import generate_smoke_drift
            self.stdout.write("drift:    %s" % generate_smoke_drift())
        if job in ("forecast", "all"):
            from ...ml_services import generate_forecasts
            self.stdout.write("forecast: %s" % generate_forecasts())
        if job == "export-training":
            from ...ml_services import export_training_csv
            horizons = tuple(int(h) for h in opts["horizons"].split(",") if h.strip())
            self.stdout.write("export:   %s" % export_training_csv(
                path=opts["out"], days=opts["days"], horizons=horizons))
        if job == "purge":
            from ...models import AirQualityForecast, WeatherReading
            now = timezone.now()
            WeatherReading.objects.filter(timestamp__lt=now - timedelta(days=60)).delete()
            AirQualityForecast.objects.filter(forecast_for__lt=now - timedelta(days=30)).delete()
            self.stdout.write("purge:    done")
