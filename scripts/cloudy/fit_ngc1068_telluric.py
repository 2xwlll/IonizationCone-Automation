"""Atmospheric fit and conservative error assessment independent of galaxy model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from astropy.io import fits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.simple_utils.wavelengths import cloudy_air_to_vacuum
from src.simple_utils.telluric_correction import AtmosphereGrid,calibration_masks,fit_transmission,corrected_with_errors,velocity_frame_factor,line_injection_sensitivity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/cloudy/ngc1068_telluric_v1')
    parser.add_argument('--spectrum',type=Path,default=ROOT/'data/spectra/ngc1068_eso_xshooter_v1/3273695/spectrum.fits')
    parser.add_argument('--atmosphere-grid',type=Path,default=ROOT/'data/spectra/ngc1068_skycalc_v2')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    obs=args.spectrum.resolve()
    models=args.atmosphere_grid.resolve()
    with fits.open(obs) as hd:
        h=hd[0].header;data=hd[1].data[0]
        if hd[1].header['TUCD1']!='em.wl;obs.atmos' or h['SPECSYS']!='TOPOCENT':raise ValueError('Unverified wavelength frame')
        air=np.array(data['WAVE'],float);wave=cloudy_air_to_vacuum(air*10)/10
        jac=np.gradient(wave,air)
        flux=np.array(data['FLUX'],float)/jac;error=np.array(data['ERR'],float)/jac;quality=np.array(data['QUAL'])
        valid=(quality==0)&np.isfinite(flux)&np.isfinite(error)&(error>0)
    use=(wave>1750)&(wave<2015)
    wave,flux,error,quality,valid=[a[use] for a in (wave,flux,error,quality,valid)]
    grid=AtmosphereGrid(models)
    meteo_pwv=float(np.mean([h['ESO TEL AMBI IWV START'],h['ESO TEL AMBI IWV END']]))
    initial=np.array([meteo_pwv,np.log(h['SPEC_RES']),0.,0.,0.,0.])
    _,_,t0=grid.evaluate(initial,wave)
    training,validation=calibration_masks(wave,valid,t0)
    if training.sum()<100 or validation.sum()<20:raise ValueError('Insufficient independent calibration/validation samples')
    variants=[dict(name='primary',degree=2,floor_fraction=.02,airmass_scale=1.),
              dict(name='linear_continuum',degree=1,floor_fraction=.02,airmass_scale=1.),
              dict(name='larger_error_floor',degree=2,floor_fraction=.05,airmass_scale=1.),
              dict(name='airmass_start',degree=2,floor_fraction=.02,airmass_scale=float(h['ESO TEL AIRM START'])/float(h['ESO QC AIRM MEAN'])),
              dict(name='airmass_end',degree=2,floor_fraction=.02,airmass_scale=float(h['ESO TEL AIRM END'])/float(h['ESO QC AIRM MEAN']))]
    results=[]
    for spec in variants:
        spec=spec.copy();name=spec.pop('name');print('Fitting',name,'training',training.sum(),'validation',validation.sum(),flush=True)
        fit=fit_transmission(grid,wave,flux,error,training,pwv_prior=meteo_pwv,resolving_power=h['SPEC_RES'],initial=results[0]['params'] if results else None,**spec)
        fit['name']=name;results.append(fit)
        serial={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in fit.items() if k not in ('predicted','continuum','transmission')}
        (out/(name+'_fit.json')).write_text(json.dumps(serial,indent=2)+'\n')
        print(name,serial,flush=True)
    primary=results[0]
    if not primary['success']:raise ValueError('Primary optimizer did not converge')
    trans=primary['transmission'];pred=primary['predicted']
    residual=(flux-pred)/np.maximum(pred,1e-30)
    held=validation&(trans>.25)
    median=float(np.median(residual[held]));scatter=float(1.4826*np.median(abs(residual[held]-median)))
    relative_validation_floor=max(.05,float(np.sqrt(scatter**2+median**2)))
    variations=[r['transmission'] for r in results]
    for j,step in enumerate(np.sqrt(np.maximum(0,np.diag(primary['approximate_conditional_covariance'])))):
        for sign in [-1,1]:
            trial=primary['params'].copy();trial[j]+=sign*step
            trial[0]=np.clip(trial[0],grid.pwv[0],grid.pwv[-1])
            trial[1]=np.clip(trial[1],np.log(3000),np.log(12000));trial[2]=np.clip(trial[2],-.25,.25)
            variations.append(grid.evaluate(trial,wave)[2])
    sensitivity=np.max(np.abs(np.array(variations)/trans-1),axis=0)
    relative_model=np.hypot(sensitivity,relative_validation_floor)
    helio=velocity_frame_factor(h['ESO QC VRAD HELICOR'])
    injection,cases=line_injection_sensitivity(grid,primary['params'],wave,helio)
    (out/'line_injection_tests.json').write_text(json.dumps(cases,indent=2)+'\n')
    relative_model=np.hypot(relative_model,injection)
    corrected,stat,total=corrected_with_errors(flux,error,trans,relative_model)
    usable=valid&(trans>=.25)&(relative_model<=.20)&(np.abs(flux)/error>=5)
    # Retain the raw data; usable points are a separate conservative quality layer.
    reasons=np.zeros(len(wave),dtype=np.uint16)
    reasons[~valid]|=1;reasons[trans<.25]|=2;reasons[relative_model>.20]|=4;reasons[np.abs(flux)/error<5]|=8
    output_wave=(wave-primary['params'][2])*helio*10
    np.savez_compressed(out/'corrected_spectrum.npz',topocentric_vacuum_nm=wave,
        heliocentric_vacuum_angstrom=output_wave,raw_flux=flux/helio,raw_error=error/helio,
        corrected_flux=corrected/helio,corrected_stat_error=stat/helio,corrected_error_envelope=total/helio,
        transmission=trans,transmission_sensitivity=sensitivity,relative_atmosphere_error=relative_model,
        line_injection_bias_envelope=injection,
        observed_quality=quality,observed_valid=valid,usable=usable,rejection_bits=reasons,
        fit_training=training,fit_validation=validation,forward_flux=pred/helio,calibration_continuum=primary['continuum']/helio)
    zoom=(output_wave>18580)&(output_wave<19050)
    summary=dict(status='Preliminary model-based telluric correction; not Molecfit or a validated intrinsic line profile',
        synthetic_galaxy_used_in_correction=False,atmosphere_source='ESO SkyCalc grid at measured mean airmass',
        observation=str(obs.relative_to(ROOT)),observation_date=h['DATE-OBS'],target=h['OBJECT'],
        atmosphere_grid=str(models.relative_to(ROOT)),grid_pwv_mm=grid.pwv.tolist(),airmass=float(h['ESO QC AIRM MEAN']),
        pwv_prior_mm=meteo_pwv,pwv_prior_effective_sigma_mm=.8,
        fitted_pwv_mm=float(primary['params'][0]),fitted_resolving_power=primary['resolving_power'],fitted_wavelength_shift_nm=float(primary['params'][2]),
        primary_success=primary['success'],primary_bounds_touched=primary['bounds_touched'],
        excluded_galaxy_lines_nm=[[1790,1800],[1819,1830],[1837,1849],[1858,1905],[1910,1935],[1947,1979]],
        training_pixels=int(training.sum()),validation_pixels=int(validation.sum()),validation_transmission_ge_0p25_pixels=int(held.sum()),
        heldout_fractional_median_residual=median,heldout_fractional_robust_scatter=scatter,relative_model_error_floor=relative_validation_floor,
        usable_zoom_pixels=int(np.sum(usable&zoom)),total_zoom_pixels=int(zoom.sum()),
        usable_definition='original QUAL==0, finite with positive ERR; effective transmission >=0.25; relative atmospheric sensitivity envelope <=0.20; absolute raw SNR>=5',
        rejection_bits={'1':'invalid/flagged input','2':'transmission <0.25','4':'atmospheric sensitivity >20 percent','8':'absolute raw SNR <5'},
        error_envelope='Quadrature of pipeline statistical error and flux times sensitivity. Fractional sensitivity combines max change among five fits and conditional parameter +/-1 sigma perturbations, >=5 percent held-out residual floor, and maximum generic line-injection division bias in quadrature. Not a calibrated confidence interval.',
        limitations=['No source-model or CLOUDY template was fitted to repair the observation',
            'Transmission multiplication and instrumental convolution do not commute; 126 generic line injections assess limited scenarios, not all intrinsic profiles',
            'Absolute calibration, slit losses, aperture/seeing mismatch and spectral covariance are not covered by this error envelope',
            'Atmospheric profile/interpolation errors may exceed limited fit-family sensitivity',
            'Saturated and unsupported wavelengths stay rejected; no interpolation or synthetic filling is performed',
            'This is corrected ground-based data, not a JWST spectrum'],
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [obs,models/'manifest.json',Path(__file__),ROOT/'src/simple_utils/telluric_correction.py']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
