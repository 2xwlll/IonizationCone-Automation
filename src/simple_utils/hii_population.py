"""Render independently placed stellar-ionized gas from a dedicated CLOUDY grid."""
import numpy as np
from scipy.ndimage import gaussian_filter
from src.simple_utils.f187n import line_response
from src.simple_utils.synthetic_photometry import photon_bandpass_weights


def render_hii_population(grid, cfg, east_arcsec, north_arcsec, redshift, tau,
                          dust_exponent, oversample, rng, diagnostics=None):
    """Return relative pre-PSF gas maps. Geometry never takes a cone mask."""
    model=int(cfg['model_index'])
    if not 0<=model<len(grid['line_flux']): raise ValueError('Invalid H II model index')
    wave=grid['filter_wavelength_angstrom'];throughput=grid['throughput']
    pa=np.deg2rad(cfg['position_angle_east_of_north_deg'])
    east=east_arcsec-cfg.get('center_east_arcsec',0.)
    north=north_arcsec-cfg.get('center_north_arcsec',0.)
    major=east*np.sin(pa)+north*np.cos(pa)
    minor=east*np.cos(pa)-north*np.sin(pa)
    radius=np.hypot(major,minor/cfg['axis_ratio'])
    field=gaussian_filter(rng.normal(size=radius.shape),2*oversample)
    field/=max(field.std(),1e-12)
    if cfg['profile']=='disk':
        envelope=np.exp(-radius/cfg['scale_radius_arcsec'])
    elif cfg['profile']=='ring':
        envelope=np.exp(-.5*((radius-cfg['ring_radius_arcsec'])/cfg['ring_width_arcsec'])**2)
    else: raise ValueError('Unknown star-forming spatial profile')
    spatial=envelope*np.exp(.7*field)
    velocity=cfg['rotation_kms']*major/np.sqrt(radius**2+.1**2)
    if diagnostics is not None:
        diagnostics.update(spatial=spatial, velocity=velocity)
    velocities=np.linspace(-400,400,81)
    maps={key:np.zeros_like(radius) for key in ('pa_alpha','other_lines','nebular_continuum')}
    for j,rest in enumerate(grid['rest_line_angstrom']):
        responses=np.array([line_response(rest,redshift,v,cfg['sigma_kms'],wave,throughput) for v in velocities])
        response=np.interp(velocity,velocities,responses)
        attenuation=np.exp(-tau*(rest*(1+velocity/299792.458)/18751.)**(-dust_exponent))
        key='pa_alpha' if grid['line_labels'][j]=='H 1' and min(abs(rest-18751),abs(rest-18756.125))<.1 else 'other_lines'
        maps[key]+=spatial*grid['line_flux'][model,j]*response*attenuation
    weight=photon_bandpass_weights(wave,wave,throughput)
    for j,w in enumerate(wave):
        maps['nebular_continuum']+=spatial*grid['continuum_rest_flambda'][model,j]/(1+redshift)*weight[j]*np.exp(
            -tau*(w/(18751.*(1+redshift)))**(-dust_exponent))
    return maps
