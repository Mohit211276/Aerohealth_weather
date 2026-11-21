from django.apps import AppConfig


class WeatherConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    # Works whether it is installed as "backend.weather" or "weather".
    name = __package__
    label = "weather"
    verbose_name = "Weather for air quality"
