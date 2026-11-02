"""Explicit phenomenological near-IR nuclear continuum, not a fitted torus SED."""
import numpy as np
from src.simple_utils.synthetic_photometry import photon_bandpass_weights


def nuclear_flambda_shape(rest_angstrom, cfg):
    """Rest F_lambda shape normalized to one at reference wavelength.

    Combine a Planck spectrum and F_nu proportional to nu**alpha, attenuate by
    a power-law dust screen, then normalize at the reference. The dust fraction
    is defined before attenuation at that wavelength; luminosity is separate.
    """
    wave=np.asarray(rest_angstrom,dtype=float)
    ref=float(cfg.get('reference_rest_angstrom',18751.))
    temp=float(cfg['hot_dust_temperature_k'])
    fraction=float(cfg['hot_dust_fraction_at_reference'])
    alpha=float(cfg['powerlaw_alpha_nu'])
    tau=float(cfg.get('tau_at_reference',0.))
    exponent=float(cfg.get('extinction_exponent',1.3))
    if np.any(wave<=0) or not np.isfinite(wave).all() or ref<=0 or temp<=0 or not 0<=fraction<=1 or tau<0:
        raise ValueError('Invalid nuclear continuum parameters')
    # hc/k = 1.438776877e8 Angstrom K; ratios avoid arbitrary luminosity units.
    x=1.438776877e8/(wave*temp)
    xref=1.438776877e8/(ref*temp)
    log_expm1=lambda a: a+np.log(-np.expm1(-a))
    log_planck=-5*np.log(wave/ref)+log_expm1(xref)-log_expm1(x)
    planck=np.exp(log_planck)
    powerlaw=(wave/ref)**(-alpha-2)
    attenuation_relative=np.exp(-tau*((wave/ref)**(-exponent)-1))
    return (fraction*planck+(1-fraction)*powerlaw)*attenuation_relative


def nuclear_filter_response(wave, throughput, redshift, cfg):
    """Photon-weighted response for unit rest-reference F_lambda, no distance."""
    if redshift<0:
        raise ValueError('Invalid redshift')
    shape=nuclear_flambda_shape(np.asarray(wave)/(1+redshift),cfg)/(1+redshift)
    return float(photon_bandpass_weights(wave,wave,throughput)@shape)
