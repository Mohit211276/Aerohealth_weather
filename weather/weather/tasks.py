"""Celery tasks. Names are prefixed "weather." so they never clash with other apps' tasks."""
from datetime import timedelta

from celery import shared_task
from django.utils import timezone


@shared_task(name="weather.fetch_weather")
def fetch_weather(past_days=1, forecast_days=3):
    from .weather_client import fetch_and_store
    return fetch_and_store(past_days=past_days, forecast_days=forecast_days)


@shared_task(name="weather.generate_smoke_drift")
def generate_smoke_drift():
    from .smoke import generate_smoke_drift as run
    return run()


@shared_task(name="weather.generate_forecasts")
def generate_forecasts():
    from .ml_services import generate_forecasts as run
    return run()


@shared_task(name="weather.purge_old")
def purge_old(reading_days=60, forecast_days=30):
    from .models import AirQualityForecast, WeatherReading
    now = timezone.now()
    WeatherReading.objects.filter(timestamp__lt=now - timedelta(days=reading_days)).delete()
    AirQualityForecast.objects.filter(forecast_for__lt=now - timedelta(days=forecast_days)).delete()
