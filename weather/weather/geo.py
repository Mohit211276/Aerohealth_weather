"""Pure geometry helpers (no Django imports, so they are easy to test)."""
import numpy as np

EARTH_RADIUS_KM = 6371.0088


def downwind_bearing(wind_from_deg):
    """Weatherrological wind direction = where the wind comes FROM; smoke travels the opposite way."""
    return (wind_from_deg + 180.0) % 360.0


def destination_points(lat, lon, bearing_deg, distance_km):
    """Great-circle destination. Works on scalars or numpy arrays."""
    phi1, lam1 = np.radians(lat), np.radians(lon)
    theta, delta = np.radians(bearing_deg), np.asarray(distance_km) / EARTH_RADIUS_KM
    phi2 = np.arcsin(np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(theta))
    lam2 = lam1 + np.arctan2(np.sin(theta) * np.sin(delta) * np.cos(phi1),
                             np.cos(delta) - np.sin(phi1) * np.sin(phi2))
    return np.degrees(phi2), (np.degrees(lam2) + 540.0) % 360.0 - 180.0


def destination_point(lat, lon, bearing_deg, distance_km):
    la, lo = destination_points(lat, lon, bearing_deg, distance_km)
    return float(la), float(lo)


def haversine_km(lat1, lon1, lat2, lon2):
    """Scalars or numpy arrays (broadcasting)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
