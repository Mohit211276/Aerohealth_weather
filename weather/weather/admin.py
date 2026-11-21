from django.contrib import admin

from .models import AirQualityForecast, WeatherReading, SmokeDriftPrediction


@admin.register(WeatherReading)
class WeatherReadingAdmin(admin.ModelAdmin):
    list_display = ("location", "timestamp", "temperature_c", "wind_speed_ms", "wind_direction_deg",
                    "boundary_layer_m")
    date_hierarchy = "timestamp"
    raw_id_fields = ("location",)


@admin.register(AirQualityForecast)
class AirQualityForecastAdmin(admin.ModelAdmin):
    list_display = ("location", "forecast_for", "horizon_hours", "prediction", "model_version", "run_at")
    list_filter = ("model_version", "horizon_hours")
    date_hierarchy = "forecast_for"
    raw_id_fields = ("location",)


@admin.register(SmokeDriftPrediction)
class SmokeDriftPredictionAdmin(admin.ModelAdmin):
    list_display = ("hotspot", "hours_ahead", "predicted_for", "lat", "lon", "distance_km")
    list_filter = ("hours_ahead",)
    raw_id_fields = ("hotspot",)
