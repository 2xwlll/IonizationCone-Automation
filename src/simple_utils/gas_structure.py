"""Illustrative intrinsic gas illumination, separate from segmentation geometry."""
import numpy as np
from scipy.special import expit


def soft_agn_emission_weight(axial, cross, radius, projected_half_angle,
                             label_extent, texture, boundary_field,
                             half_field_arcsec, cfg):
    """Smooth irregular illumination on a continuous cloud distribution.

    The geometric label is deliberately not an input. This is an empirical
    emissivity prescription, not a radiative-transfer solution or a measured
    angular escape fraction. Widths describe intrinsic emission before the PSF.
    """
    width=np.deg2rad(cfg['angular_transition_deg'])
    radial_width=cfg['radial_transition_fraction']
    if width<=0 or radial_width<=0: raise ValueError('Emission transitions must be positive')
    leakage=cfg['off_axis_fraction']
    if not 0<=leakage<1: raise ValueError('Invalid off-axis emissivity floor')
    angle=np.arctan2(np.abs(cross),np.abs(axial))
    perturb=np.deg2rad(cfg['boundary_variation_deg'])*np.tanh(boundary_field)
    illumination=leakage+(1-leakage)*expit((projected_half_angle+perturb-angle)/width)
    radial=expit((label_extent-radius)/radial_width)
    extended=illumination*radial*np.exp(-cfg['radial_decay']*radius)*(.3+texture)
    sigma=cfg['central_gas_sigma_arcsec']
    if sigma<=0: raise ValueError('Central gas width must be positive')
    central=(cfg['central_gas_amplitude']*np.exp(-.5*(radius*half_field_arcsec/sigma)**2)*
             (.7+.3*texture))
    return extended+central


def nuclear_spatial_profile(shape, pixel_scale_arcsec, extended_fraction, sigma_arcsec):
    """Unit-sum point + compact Gaussian; a phenomenological continuum profile."""
    if not 0<=extended_fraction<=1 or sigma_arcsec<=0 or pixel_scale_arcsec<=0:
        raise ValueError('Invalid nuclear structure')
    yy,xx=np.indices(shape,dtype=float)
    radius2=(xx-(shape[1]-1)/2)**2+(yy-(shape[0]-1)/2)**2
    extended=np.exp(-.5*radius2/(sigma_arcsec/pixel_scale_arcsec)**2)
    extended/=extended.sum()
    profile=extended_fraction*extended
    profile[shape[0]//2,shape[1]//2]+=1-extended_fraction
    return profile
