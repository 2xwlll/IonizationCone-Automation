"""Audit the actual archive extraction and sample the unchanged 2D galaxy model."""
import argparse,hashlib,json,os
from pathlib import Path
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
from scipy.special import erf

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.simple_utils.slit_aperture import sky_offsets,gaussian_slit_acceptance
from src.simple_utils.synthetic_photometry import photon_bandpass_weights
from scripts.cloudy.plot_f187n_spectral_reference import reconstruct


def profile_function(s,amplitude,center,sigma,background,slope):
    return amplitude*(erf((s+.105-center)/(2**.5*sigma))-erf((s-.105-center)/(2**.5*sigma)))/2+background+slope*(s+2.5)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source2=ROOT/'data/spectra/ngc1068_slit_audit_v2/slit_2d.fits'
    source1=ROOT/'data/spectra/ngc1068_eso_xshooter_v2/3537083/spectrum.fits'
    with fits.open(source2) as hd:
        h2=hd[0].header;image=np.array(hd[0].data,float);quality=np.array(hd['QUAL'].data)
    with fits.open(source1) as hd:
        h1=hd[0].header;table=hd[1].data[0]
        reduced=np.array(table['FLUX_REDUCED']);valid=table['QUAL']==0;wave=np.array(table['WAVE'])
    params={h1[k]:h1[k.replace(' NAME',' VALUE')] for k in h1 if k.startswith('ESO PRO REC1 PARAM') and k.endswith(' NAME')}
    assert params['localize-method']=='MANUAL' and float(params['localize-slit-position'])==0 and float(params['localize-slit-hheight'])==2
    assert h2['CUNIT2']=='arcsec' and h2['BUNIT']=='ADU'
    spatial=h2['CRVAL2']+(np.arange(len(image))+1-h2['CRPIX2'])*h2['CDELT2']
    assert abs(h2['CDELT2']-.21)<1e-9
    rows=abs(spatial)<=2
    collapsed=image[rows].sum(axis=0)
    check=valid&(wave>1750)&(wave<2015)&(reduced>0)
    difference=abs(collapsed[check]/reduced[check]-1)
    anchor=check&(wave>1985)&(wave<1995)
    anchor_difference=abs(collapsed[anchor]/reduced[anchor]-1)
    # The manual describes extraction before merging orders: the archived merged
    # 2D image need not sum bit-for-bit to the separately merged 1D product.
    # Verify the clean sideband at numerical precision and audit broad residuals.
    if np.max(anchor_difference)>1e-6 or np.percentile(difference,99)>.01 or np.max(difference)>.03:
        raise AssertionError('Archive extraction does not match inferred rows')
    profiles=[];fractions=[];fits_report=[]
    for lo,hi in [(1985,1995),(1979,1984),(1995,2000)]:
        selected=(wave>lo)&(wave<hi)
        profile=np.nanmedian(np.where(quality[:,selected]==0,image[:,selected],np.nan),axis=1)
        good_fraction=np.mean(quality[:,selected]==0,axis=1)
        # The nuclear core is saturation-flagged in most samples. Exclude rows
        # with fewer than half their samples valid; do not fit a selectively
        # depressed median based on only a few unsaturated wavelengths.
        use=(spatial>-3.5)&(spatial<-1.5)&np.isfinite(profile)&(good_fraction>=.5)
        pars,cov=curve_fit(profile_function,spatial[use],profile[use],p0=[150000,-2.5,.22,1500,0],
            bounds=([0,-3,.05,0,-1000],[1e7,-2,1,10000,1000]))
        fits_report.append(dict(air_wavelength_window_nm=[lo,hi],center_arcsec=float(pars[1]),
            effective_compact_fwhm_arcsec=float(pars[2]*2.354820045),parameters=pars.tolist(),
            fitted_row_indices=np.where(use)[0].tolist(),good_sample_fraction_by_row=good_fraction.tolist(),
            fractional_rms=float(np.std((profile_function(spatial[use],*pars)-profile[use])/profile[use]))))
        profiles.append(profile);fractions.append(good_fraction)
    peak=fits_report[0]['center_arcsec'];fwhm=fits_report[0]['effective_compact_fwhm_arcsec']
    lower=float(spatial[rows][0]-.105);upper=float(spatial[rows][-1]+.105)
    length=upper-lower;offset=(upper+lower)/2
    reference=ROOT/'results/cloudy/f187n_reference/nucleus_cutout.npz'
    meta=json.loads(str(np.load(reference)['metadata']))
    origin=SkyCoord(meta['target_ra_deg']*u.deg,meta['target_dec_deg']*u.deg)
    # Nominal target and approximate telescope pointing disagree across slit;
    # retain both rather than silently choosing coordinates to fit line flux.
    nominal=SkyCoord('02h42m40.807s','-00d00m45.7s')
    assert abs(float(h1['ESO TEL TARG ALPHA'])-24240.807)<1e-6 and abs(float(h1['ESO TEL TARG DELTA'])+45.7)<1e-6
    pa=38.;a=np.deg2rad(pa)
    coords={}
    for name,target in [('nominal_target',nominal),('header_pointing',SkyCoord(h1['RA']*u.deg,h1['DEC']*u.deg))]:
        e,n=origin.spherical_offsets_to(target)
        coords[name]=dict(east_arcsec=e.arcsec,north_arcsec=n.arcsec,
            along_arcsec=e.arcsec*np.sin(a)+n.arcsec*np.cos(a),cross_arcsec=e.arcsec*np.cos(a)-n.arcsec*np.sin(a))
    cross=coords['nominal_target']['cross_arcsec'];along=-peak+offset
    base=dict(pa_deg=pa,along_center=along,cross_center=cross,width=.9,length=length,fwhm=fwhm)
    scenarios={'registered':base,
        'nominal_coordinates':{**base,'along_center':coords['nominal_target']['along_arcsec']+offset},
        'along_minus_0p21':{**base,'along_center':along-.21},'along_plus_0p21':{**base,'along_center':along+.21},
        'cross_minus_0p30':{**base,'cross_center':cross-.30},'cross_plus_0p30':{**base,'cross_center':cross+.30},
        'seeing_0p40':{**base,'fwhm':.4},'seeing_0p80':{**base,'fwhm':.8},
        'header_pointing_alternative':{**base,'along_center':coords['header_pointing']['along_arcsec']+offset,'cross_center':coords['header_pointing']['cross_arcsec']}}
    audit=dict(status='Extraction dimensions verified; registration and seeing are approximate and tested with scenarios',
        observation=str(source1.relative_to(ROOT)),slit_product=str(source2.relative_to(ROOT)),
        extraction_parameters=params,selected_row_indices=np.where(rows)[0].tolist(),selected_row_centers_arcsec=spatial[rows].tolist(),
        row_count=int(rows.sum()),actual_extraction_edges_arcsec=[lower,upper],width_arcsec=.9,length_arcsec=length,
        raw_sum_relative_difference_median=float(np.median(difference)),raw_sum_relative_difference_p99=float(np.percentile(difference,99)),
        raw_sum_relative_difference_max=float(np.max(difference)),sideband_sum_max_relative_difference=float(np.max(anchor_difference)),
        extraction_audit_note='Sideband sum matches to numerical precision. Broad residuals are retained, consistent with extraction and merging being ordered differently; this is an inference, not a rerun of the ESO pipeline.',
        raw_sum_checked_wavelength_pixels=int(check.sum()),continuum_peak_fits=fits_report,
        pa_east_of_north_deg=pa,pa_source='https://academic.oup.com/mnras/article/550/3/stag1183/8713495',
        extraction_units_source='https://ftp.eso.org/pub/dfs/pipelines/xshooter/xsh-pipeline-manual-3.0.pdf',
        nominal_and_pointing_offsets=coords,registration_assumption='The compact 1.99-micron continuum peak traces the modeled nucleus; positive slit coordinate points northeast along published PA=38 deg. Cross-slit centering follows nominal target coordinates.',
        seeing_assumption='Pixel-integrated Gaussian fit to compact continuum wings, rejecting rows with <50 percent valid samples because the core is saturation-flagged. Effective source width, not an independent atmospheric PSF measurement.',
        saturation_flag_source='https://ftp.eso.org/pub/dfs/pipelines/instruments/xshooter/xshoo-pipeline-manual-3.6.3.pdf',
        nuclear_core_flag=1048576,nuclear_core_note='Most core samples in rows 13 and 14 carry the NIR saturation flag. They are outside the extracted rows; neither their fluxes nor a fitted core reconstruction are used to correct the 1D spectrum.',
        registration_sensitivity_note='Offset and seeing scenarios are explicit sensitivity choices, not confidence limits; header-coordinate alternative is separate from the small-offset envelope',
        scenarios=scenarios)
    (out/'aperture_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    np.savez_compressed(out/'slit_audit_data.npz',spatial_arcsec=spatial,selected_rows=rows,continuum_profiles=np.array(profiles),
        good_sample_fractions=np.array(fractions),
        fitted_profile=profile_function(spatial,*fits_report[0]['parameters']),wavelength_air_nm=wave,
        collapsed_reduced_flux=collapsed,archived_reduced_flux=reduced,checked=check)
    state_path=ROOT/'results/cloudy/ngc1068_f187n_spectral_export_v1/spectral_state.npz'
    state=dict(np.load(state_path));cfg=json.loads(str(state['configuration']))
    grid=dict(np.load(ROOT/cfg['grid']));hii=dict(np.load(ROOT/cfg['star_forming']['grid']))
    fw=grid['filter_wavelength_angstrom'];throughput=grid['throughput'];wave=np.arange(fw[0],fw[-1]+.05,.1)
    fine=meta['pixel_scale_arcsec']/cfg['oversample'];east,north=sky_offsets(state['tau'].shape,fine,meta['pixel_to_sky_matrix'])
    wts=photon_bandpass_weights(wave,fw,throughput);coarse=photon_bandpass_weights(fw,fw,throughput)
    reports={}
    for name,geometry in scenarios.items():
        print('Extracting',name,geometry,flush=True)
        acceptance=gaussian_slit_acceptance(east,north,**geometry)
        spectra,continua=reconstruct(state,cfg,grid,hii,wave,intrinsic_acceptance=acceptance)
        total=sum(spectra.values());continuum=sum(spectra[k] for k in continua)
        np.savez_compressed(out/(name+'.npz'),observed_vacuum_angstrom=wave,synthetic_relative_flambda=total,
            synthetic_continuum=continuum,filter_wavelength_angstrom=fw,throughput=throughput,
            intrinsic_acceptance=acceptance,**spectra)
        hard=gaussian_slit_acceptance(east,north,**{**geometry,'fwhm':0})
        reports[name]=dict(geometry=geometry,geometric_area_fraction_in_existing_field=float(hard.sum()*fine**2/(geometry['width']*geometry['length'])),
            seeing_acceptance_area_fraction_in_existing_field=float(acceptance.sum()*fine**2/(geometry['width']*geometry['length'])),
            component_band_integrals={k:float(coarse@continua[k] if k in continua else wts@v) for k,v in spectra.items()})
        (out/'extraction_summary.json').write_text(json.dumps(reports,indent=2)+'\n')
    audit['hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source1,source2,state_path,reference,ROOT/cfg['grid'],ROOT/cfg['star_forming']['grid'],Path(__file__),ROOT/'src/simple_utils/slit_aperture.py',ROOT/'scripts/cloudy/plot_f187n_spectral_reference.py']}
    (out/'aperture_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Completed all slit scenarios.',flush=True)

if __name__=='__main__':main()
