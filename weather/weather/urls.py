from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

app_name = "weather"  # for include("backend.weather.urls", namespace="weather")

router = DefaultRouter()
router.register("readings", views.WeatherReadingViewSet, basename="reading")
router.register("forecasts", views.AirQualityForecastViewSet, basename="forecast")
router.register("smoke-drift", views.SmokeDriftViewSet, basename="smoke-drift")

urlpatterns = [
    path("context/", views.ContextView.as_view(), name="context"),
    path("predict/", views.PredictAQIView.as_view(), name="predict"),
    path("status/", views.StatusView.as_view(), name="status"),
    path("", include(router.urls)),
]
