from rest_framework import serializers

from .models import AirQualityForecast, WeatherReading, SmokeDriftPrediction


class WeatherReadingSerializer(serializers.ModelSerializer):
    is_forecast = serializers.ReadOnlyField()
    ventilation_index = serializers.ReadOnlyField()
    dispersion = serializers.ReadOnlyField()

    class Meta:
        model = WeatherReading
        fields = ["id", "location", "timestamp", "temperature_c", "humidity_pct", "pressure_hpa",
                  "precipitation_mm", "wind_speed_ms", "wind_direction_deg", "boundary_layer_m",
                  "ventilation_index", "dispersion", "is_forecast", "source"]


class AirQualityForecastSerializer(serializers.ModelSerializer):
    category = serializers.ReadOnlyField()
    location_name = serializers.CharField(source="location.name", read_only=True)

    class Meta:
        model = AirQualityForecast
        fields = ["id", "location", "location_name", "run_at", "forecast_for", "horizon_hours",
                  "prediction", "category", "confidence_lower", "confidence_upper",
                  "model_version", "generated_at"]
        read_only_fields = ["id", "run_at", "generated_at"]
        extra_kwargs = {"horizon_hours": {"required": False}}

    def create(self, validated_data):
        if validated_data.get("horizon_hours") is None:
            run_at = validated_data.get("run_at")
            if run_at is not None:
                hours = (validated_data["forecast_for"] - run_at).total_seconds() / 3600
                validated_data["horizon_hours"] = max(0, round(hours))
        return super().create(validated_data)


class SmokeDriftPredictionSerializer(serializers.ModelSerializer):
    source_frp = serializers.FloatField(source="hotspot.frp", read_only=True)

    class Meta:
        model = SmokeDriftPrediction
        fields = ["id", "hotspot", "source_frp", "hours_ahead", "predicted_for", "lat", "lon",
                  "distance_km", "wind_speed_ms", "wind_direction_deg", "generated_at"]


class PredictionRequestSerializer(serializers.Serializer):
    pm25 = serializers.FloatField()
    pm10 = serializers.FloatField()
    no = serializers.FloatField()
    no2 = serializers.FloatField()
    nox = serializers.FloatField()
    nh3 = serializers.FloatField()
    co = serializers.FloatField()
    so2 = serializers.FloatField()
    o3 = serializers.FloatField()
    benzene = serializers.FloatField()
    toluene = serializers.FloatField()
    lat = serializers.FloatField()
    lon = serializers.FloatField()
    month = serializers.IntegerField(min_value=1, max_value=12)
    day_of_week = serializers.IntegerField(min_value=0, max_value=6)
