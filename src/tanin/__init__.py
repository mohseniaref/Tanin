"""Tanin: xarray-native spectral analysis for geodetic time series."""
from .time import to_decimal_days
from .harmonic import fit_harmonics, test_period, fit_geodetic
from .spectrum import spectrum, lssa, dominant_period
from .simulation import simulate
from . import io
from .noise import estimate_noise_vce, estimate_noise, covariance_components, powerlaw_covariance, read_time_series
from .advanced import allssa, lswa, lscsa, lscwa, detect_jumps, just_jumps, just_decompose, just_monitor
from .stpd import stpd, tpdetect, tptr

__all__ = ["to_decimal_days", "fit_harmonics", "fit_geodetic", "test_period", "spectrum", "lssa", "dominant_period", "simulate", "estimate_noise_vce", "estimate_noise", "covariance_components", "powerlaw_covariance", "read_time_series", "allssa", "lswa", "lscsa", "lscwa", "detect_jumps", "just_jumps", "just_decompose", "just_monitor", "stpd", "tpdetect", "tptr", "io"]
