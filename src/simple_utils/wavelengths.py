"""Match CLOUDY C25's air-to-vacuum wavelength convention (Peck & Reeder 1972)."""
import numpy as np


def cloudy_air_to_vacuum(wavelength_angstrom):
    """Convert printed air lines >2000 A with CLOUDY's two-step iteration.

    CLOUDY uses vacuum for shorter printed lines already. Continuum wavelengths
    are always vacuum and must not be passed through this conversion.
    Source: C25 source/cddefines.cpp, t_wavl::p_wlAirVac and p_RefIndex.
    """
    air=np.asarray(wavelength_angstrom,dtype=float)
    if np.any(air<=0) or not np.isfinite(air).all(): raise ValueError('Invalid wavelength')
    vacuum=air.copy()
    for _ in range(2):
        squared=(1e4/vacuum)**2
        index=1+1e-8*(8060.51+2480990/(132.274-squared)+17455.7/(39.32957-squared))
        vacuum=np.where(air>2000,air*index,air)
    return vacuum
