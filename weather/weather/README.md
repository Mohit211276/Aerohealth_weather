# weather - weather and smoke-drift backend for the VayuDrishti air-quality platform

A Django REST app that supplies the *weather context* around air-quality readings: hourly weather at
the CPCB monitoring stations, a 24-72 hour AQI forecast, and a stubble-burning smoke-drift estimate for
Delhi-NCR. It is built on top of the `airquality` app and uses its tables (`Location`, `GroundReading`,
`FireHotspot`).

## What it does
- **Weather per station** - hourly temperature, humidity, pressure, rain, wind speed/direction and boundary-layer
  height from Open-Meteo, with a *ventilation index* (wind speed x boundary-layer height) that flags poor dispersion.
- **AQI forecast (24/48/72 h)** - features are the current CPCB AQI, wind and weather at the target hour, and fires
  within 100 km in the last 24 h. Predictions come from a trained model file; training data can be exported
  with one command, using the same feature builder the forecast uses.
- **Stubble-smoke tracker** - recent NASA FIRMS hotspots are moved downwind hour by hour (each smoke parcel follows
  the wind of the nearest station to its current position) to estimate where smoke will be after 6/12/24/48 h,
  and whether it will reach Delhi-NCR.
- **One-call summary** - `context/` returns AQI, weather, dispersion, forecast and incoming smoke for any point.

## Structure
    weather/
      models.py        WeatherReading, AirQualityForecast, SmokeDriftPrediction
      serializers.py   API serializers
      views.py         REST endpoints
      urls.py          routes (app_name = "weather")
      weather_client.py  Open-Meteo client (standard library only)
      smoke.py         vectorised smoke-drift model
      ml_services.py   model loading, forecast generation, training export
      geo.py           pure geometry helpers
      tasks.py         optional Celery tasks
      management/commands/weather_jobs.py   run all jobs without Celery
      migrations/0001_initial.py
      tests.py         31 tests

## Requirements
Python 3.12+, Django, djangorestframework, numpy, joblib, psycopg2, PostgreSQL with PostGIS, GDAL.
(Celery is optional.) The `airquality` app must be installed.

## Install
1. Put this folder at `backend/weather/`.
2. `settings.py`:  add `'backend.weather'` to `INSTALLED_APPS` (after `'backend.airquality'`).
3. `urls.py`:      `path("api/weather/", include("backend.weather.urls", namespace="weather"))`
4. `python manage.py migrate`
5. `python manage.py test backend.weather`   ->  Ran 31 tests ... OK

## Run
    python manage.py weather_jobs fetch                  # Open-Meteo -> WeatherReading (CPCB stations)
    python manage.py weather_jobs fetch --past-days 30   # backfill history for training (max 92)
    python manage.py weather_jobs drift                  # FIRMS hotspots + wind -> smoke drift
    python manage.py weather_jobs forecast               # needs a trained model (below)
    python manage.py weather_jobs export-training --days 30 --horizons 24,48,72
    python manage.py weather_jobs all                    # fetch + drift + forecast

## API (prefix /api/weather/)
| Endpoint | Purpose |
|---|---|
| `context/?lat=&lon=` | AQI + weather + dispersion + 72 h forecast + smoke, in one call |
| `readings/`, `readings/at-point/?lat=&lon=` | hourly weather per station; nearest-station current + next hours |
| `forecasts/`, `forecasts/at-point/?lat=&lon=` | AQI forecasts (newest run), with AQI category |
| `forecasts/ingest/` (POST, staff) | an external pipeline can post forecasts |
| `smoke-drift/`, `smoke-drift/arrivals/?lat=&lon=&radius_km=` | predicted smoke positions; will smoke reach a place |
| `smoke-drift/ncr/` | smoke heading for Delhi-NCR |
| `predict/` (POST) | pollutant readings -> predicted AQI |
| `status/` | data freshness and model-file availability |

## Models
The forecast and AQI models are trained outside the app and loaded from
`datascience/models/aqi_forecast_model.joblib` and `aqi_model.joblib`. Column order is defined once in
`ml_services.py` (`FORECAST_FEATURES`, `FEATURES`). Until a model exists, `weather_jobs forecast` reports
`no_model`; for demos only, `WEATHER_ALLOW_BASELINE_FORECAST = True` saves clearly labelled
`persistence` forecasts (the current AQI repeated).

## Settings (all optional)
`WEATHER_SMOKE_BBOX`, `WEATHER_ALLOW_BASELINE_FORECAST`, `AQI_FORECAST_MODEL_PATH`, `AQI_MODEL_PATH`,
`AQI_FORECAST_MODEL_VERSION`, `AQI_MODEL_VERSION`.

## Limitations
- The smoke model is a wind-advection estimate, not a full dispersion model.
- Weather exists only for stations that were fetched, so a point far from any station has no weather.
- The ventilation threshold (6000 m2/s) is a rule of thumb; change it in `models.py` if needed.

## Data attribution
Weather data by Open-Meteo.com (https://open-meteo.com), licensed CC BY 4.0, free for non-commercial use.
Fire data: NASA FIRMS. AQI ground truth: CPCB.

## Upgrading a development database that still has an OLD `weather` app
Only needed if your database already recorded a different app called `weather` (a fresh database does not).
Back up first, then:

    psql -U postgres -d vayudrishti -c "DELETE FROM django_migrations WHERE app = 'weather';"
    psql -U postgres -d vayudrishti -c "DROP TABLE IF EXISTS weather_weatherobservation, weather_weatherforecast, weather_nasapowerreanalysis CASCADE;"
    python manage.py migrate
