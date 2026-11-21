from django.contrib.gis.db import models
from django.contrib.gis.geos import Point
from django.utils import timezone

# Rule of thumb: ventilation (wind speed x boundary-layer height) below ~6000 m2/s means
# pollutants disperse poorly. Adjust here if your team prefers another threshold.
VENTILATION_POOR_M2S = 6000


def aqi_category(aqi):
    """Indian National AQI categories (same bands as the airquality app)."""
    if aqi is None:
        return None
    for limit, name in ((50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
                        (300, "Poor"), (400, "Very Poor")):
        if aqi <= limit:
            return name
    return "Severe"


class WeatherReading(models.Model):
    """Hourly weather at an airquality.Location (past + forecast hours, Open-Meteo)."""
    location = models.ForeignKey("airquality.Location", on_delete=models.CASCADE,
                                 related_name="weather_readings")
    timestamp = models.DateTimeField(db_index=True, help_text="Hour the values apply to (UTC)")
    temperature_c = models.FloatField(null=True, blank=True)
    humidity_pct = models.FloatField(null=True, blank=True)
    pressure_hpa = models.FloatField(null=True, blank=True)
    precipitation_mm = models.FloatField(null=True, blank=True)
    wind_speed_ms = models.FloatField(null=True, blank=True)
    wind_direction_deg = models.FloatField(null=True, blank=True,
                                           help_text="Weatherrological: direction the wind blows FROM")
    boundary_layer_m = models.FloatField(null=True, blank=True)
    source = models.CharField(max_length=30, default="open-meteo")
    fetched_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "weather_reading"
        ordering = ["-timestamp"]
        constraints = [models.UniqueConstraint(fields=["location", "timestamp", "source"],
                                               name="uniq_weather_loc_ts_src")]
        indexes = [models.Index(fields=["location", "-timestamp"], name="weather_reading_loc_ts_idx")]

    @property
    def is_forecast(self):
        return self.timestamp > self.fetched_at

    @property
    def ventilation_index(self):
        """Ventilation coefficient in m2/s: higher = pollutants disperse more easily."""
        if self.wind_speed_ms is None or self.boundary_layer_m is None:
            return None
        return round(self.wind_speed_ms * self.boundary_layer_m, 1)

    @property
    def dispersion(self):
        vi = self.ventilation_index
        if vi is None:
            return None
        return "poor" if vi < VENTILATION_POOR_M2S else "adequate"

    def __str__(self):
        return f"{self.location} @ {self.timestamp:%Y-%m-%d %H:%M}"


class AirQualityForecast(models.Model):
    """AQI forecast (24-72h) for a location. All rows of one pipeline run share run_at."""
    location = models.ForeignKey("airquality.Location", on_delete=models.CASCADE,
                                 related_name="weather_forecasts")
    run_at = models.DateTimeField(default=timezone.now, db_index=True)
    forecast_for = models.DateTimeField()
    horizon_hours = models.PositiveSmallIntegerField(default=0)
    prediction = models.FloatField()
    confidence_lower = models.FloatField(null=True, blank=True)
    confidence_upper = models.FloatField(null=True, blank=True)
    model_version = models.CharField(max_length=30)
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "weather_aqi_forecast"
        ordering = ["location_id", "forecast_for"]
        indexes = [models.Index(fields=["location", "forecast_for"], name="weather_fc_loc_for_idx"),
                   models.Index(fields=["location", "run_at"], name="weather_fc_loc_run_idx")]

    @property
    def category(self):
        return aqi_category(self.prediction)

    def __str__(self):
        return f"{self.location} +{self.horizon_hours}h: {self.prediction:.0f}"


class SmokeDriftPrediction(models.Model):
    """Where smoke from a FIRMS hotspot is expected to be N hours from now (wind advection)."""
    hotspot = models.ForeignKey("airquality.FireHotspot", on_delete=models.CASCADE,
                                related_name="weather_drift_predictions")
    hours_ahead = models.PositiveSmallIntegerField()
    predicted_for = models.DateTimeField(db_index=True)
    lat = models.FloatField()
    lon = models.FloatField()
    point = models.PointField(geography=True, srid=4326, null=True, blank=True, editable=False)
    distance_km = models.FloatField(help_text="Distance travelled from the hotspot")
    wind_speed_ms = models.FloatField(help_text="Mean wind speed over the drift period")
    wind_direction_deg = models.FloatField(help_text="Wind-from direction at the last step")
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "weather_smoke_drift"
        ordering = ["hours_ahead", "id"]
        indexes = [models.Index(fields=["hours_ahead"], name="weather_smoke_hours_idx")]

    def save(self, *args, **kwargs):
        self.point = Point(self.lon, self.lat, srid=4326)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Smoke +{self.hours_ahead}h from hotspot {self.hotspot_id}"
