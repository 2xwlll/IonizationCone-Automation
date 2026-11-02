"""Conservative transmission fitting using external ESO SkyCalc model grids.

This is a local, limited-range model-based correction, not ESO Molecfit. The
galaxy simulation is deliberately absent from this module.
"""
from pathlib import Path
import json
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares


def velocity_frame_factor(correction_kms):
    return 1+float(correction_kms)/299792.458


class AtmosphereGrid:
    def __init__(self,folder):
        manifest=json.loads((Path(folder)/'manifest.json').read_text())
        self.pwv=np.array(sorted(m['pwv_mm'] for m in manifest['models']))
        if len(self.pwv)<2 or np.any(np.diff(self.pwv)<=0):raise ValueError('Need distinct atmospheric grid nodes')
        tables=[]
        for pwv in self.pwv:
            with fits.open(Path(folder)/f'pwv_{pwv:.1f}.fits') as hd:
                wave=np.array(hd[1].data['lam'],float)
                transmission=np.array(hd[1].data['trans'],float)
                if hd[1].columns['lam'].unit!='nm':raise ValueError('SkyCalc wavelength units require inspection')
            if np.any(np.diff(wave)<=0) or not np.isfinite(transmission).all():raise ValueError('Invalid model grid')
            tables.append((wave,transmission))
        self.logwave=np.linspace(np.log(tables[0][0][0]),np.log(tables[0][0][-1]),len(tables[0][0]))
        self.wave=np.exp(self.logwave);self.dlog=np.diff(self.logwave).mean()
        self.tau=np.array([-np.log(np.clip(np.interp(self.wave,w,t),1e-12,1.)) for w,t in tables])

    def transmission(self,pwv,airmass_scale=1.):
        j=np.clip(np.searchsorted(self.pwv,pwv)-1,0,len(self.pwv)-2)
        f=(pwv-self.pwv[j])/(self.pwv[j+1]-self.pwv[j])
        return np.exp(-((1-f)*self.tau[j]+f*self.tau[j+1])*airmass_scale)

    def evaluate(self,params,observed_wave,airmass_scale=1.):
        """Continuum*transmission convolved before integration over detector bins.

        params = PWV mm, ln(R), wavelength shift nm, ln(C0), slope, curvature.
        Wavelengths are topocentric vacuum nm; C0 is in the caller's flux unit.
        """
        pwv,log_r,shift,c0,c1,c2=params
        x=(self.wave-1880)/100
        continuum=np.exp(c0+c1*x+c2*x*x)
        transmission=self.transmission(pwv,airmass_scale)
        sigma=1/(2.354820045*np.exp(log_r)*self.dlog)
        forward=gaussian_filter1d(continuum*transmission,sigma,mode='nearest')
        reference=gaussian_filter1d(continuum,sigma,mode='nearest')
        width=np.median(np.diff(observed_wave))
        # Three-node Gauss quadrature averages across native detector wavelength bins.
        offsets=width/2*np.sqrt(3/5)*np.array([-1.,0.,1.])
        quadrature=np.array([5.,8.,5.])/18
        prediction=np.zeros_like(observed_wave);source=np.zeros_like(prediction)
        for delta,q in zip(offsets,quadrature):
            points=observed_wave-shift+delta
            prediction+=q*np.interp(points,self.wave,forward)
            source+=q*np.interp(points,self.wave,reference)
        effective=np.clip(prediction/source,1e-12,1.)
        return prediction,source,effective

    def sample_source(self,source,params,observed_wave,*,atmosphere=True):
        """Independent line-injection forward operator on the high-resolution grid."""
        value=np.asarray(source,float)
        if atmosphere:value=value*self.transmission(params[0])
        sigma=1/(2.354820045*np.exp(params[1])*self.dlog)
        blurred=gaussian_filter1d(value,sigma,mode='nearest')
        width=np.median(np.diff(observed_wave));output=np.zeros_like(observed_wave)
        for node,q in zip([-1.,0.,1.],np.array([5.,8.,5.])/18):
            output+=q*np.interp(observed_wave-params[2]+node*width/2*np.sqrt(3/5),self.wave,blurred)
        return output


def line_injection_sensitivity(grid,params,wave,helio_factor):
    """Test division bias with generic Gaussian lines, never the galaxy template.

    Envelope covers the specified widths, positions and amplitudes only; it is
    not an uncertainty distribution or a guarantee for an unknown line shape.
    """
    x=(grid.wave-1880)/100
    continuum=np.exp(params[3]+params[4]*x+params[5]*x*x)
    teff=grid.evaluate(params,wave)[2]
    envelope=np.zeros_like(wave)
    centers=np.array([1871.2296,1876.1395,1877.3243,1881.933,1882.72618,1883.3286,1896.6827])/helio_factor
    cases=[]
    for center in centers:
        for velocity in [-300.,0.,300.]:
            position=center*(1+velocity/299792.458)
            for dispersion in [40.,100.,250.]:
                for amplitude in [1.,5.]:
                    source=continuum*(1+amplitude*np.exp(-.5*((grid.wave-position)/(position*dispersion/299792.458))**2))
                    measured=grid.sample_source(source,params,wave,atmosphere=True)
                    truth=grid.sample_source(source,params,wave,atmosphere=False)
                    bias=np.abs(measured/teff/truth-1)
                    envelope=np.maximum(envelope,bias)
                    reliable=teff>=.25
                    cases.append(dict(center_topocentric_nm=float(center),velocity_kms=velocity,
                        sigma_kms=dispersion,peak_over_continuum=amplitude,
                        max_fractional_bias_at_transmission_ge_0p25=float(np.max(bias[reliable]))))
    return envelope,cases


def calibration_masks(wave,quality_mask,reference_transmission):
    """Exclude intrinsic source lines and reserve whole wavelength chunks.

    Includes guards for H2 S(7)/S(6)/S(5), hydrogen, helium and coronal features.
    Entire F187N zoom is withheld: its line strengths cannot tune correction.
    """
    eligible=quality_mask&(wave>1758)&(wave<2010)&(reference_transmission>.10)
    for lo,hi in [(1790,1800),(1819,1830),(1837,1849),(1858,1905),(1910,1935),(1947,1979)]:
        eligible &= ~((wave>=lo)&(wave<=hi))
    chunks=np.floor((wave-1758)/4).astype(int)
    return eligible&(chunks%5!=0),eligible&(chunks%5==0)


def fit_transmission(grid,wave,flux,error,fit_mask,*,pwv_prior=5.185,
                     resolving_power=5573.,floor_fraction=.02,degree=2,airmass_scale=1.,initial=None):
    unit=1e-14
    y=flux/unit;err=error/unit
    # Fixed error floor prevents tiny formal errors in deep troughs dominating.
    continuum_scale=float(np.median(y[fit_mask]))
    sigma=np.hypot(err,floor_fraction*max(continuum_scale,.1))
    p0=np.array([pwv_prior,np.log(resolving_power),0.,np.log(max(continuum_scale,.1)),0.,0.]) if initial is None else np.array(initial,copy=True)
    lower=np.array([grid.pwv[0],np.log(3000.),-.25,-5.,-2.,-2.])
    upper=np.array([grid.pwv[-1],np.log(12000.),.25,5.,2.,2.])
    active=np.array([0,1,2,3,4,5] if degree==2 else [0,1,2,3,4])
    if degree==1:p0[5]=0
    def residual(v):
        p=p0.copy();p[active]=v
        predicted,_,_=grid.evaluate(p,wave,airmass_scale)
        data=(predicted[fit_mask]-y[fit_mask])/sigma[fit_mask]
        # Conservative profile/LSF priors, not the much smaller IWV sensor error.
        return np.r_[data,(p[0]-pwv_prior)/.8,(p[1]-np.log(resolving_power))/.4,p[2]/.1]
    result=least_squares(residual,p0[active],bounds=(lower[active],upper[active]),loss='soft_l1',f_scale=1.,max_nfev=180,x_scale='jac')
    params=p0.copy();params[active]=result.x
    predicted,source,trans=grid.evaluate(params,wave,airmass_scale)
    covariance=np.zeros((6,6))
    covariance[np.ix_(active,active)]=np.linalg.pinv(result.jac.T@result.jac)*max(1.,2*result.cost/max(1,fit_mask.sum()-len(active)))
    return dict(params=params,predicted=predicted*unit,continuum=source*unit,transmission=trans,
        approximate_conditional_covariance=covariance,
        success=bool(result.success),message=result.message,nfev=int(result.nfev),cost=float(result.cost),
        resolving_power=float(np.exp(params[1])),airmass_scale=airmass_scale,degree=degree,
        floor_fraction=floor_fraction,fit_pixels=int(fit_mask.sum()),
        bounds_touched=bool(np.any((params[active]-lower[active])<.001)|np.any((upper[active]-params[active])<.001)))


def corrected_with_errors(flux,error,transmission,relative_model_error):
    """Return corrected data, statistical error and a model-sensitivity envelope.

    This envelope is not a calibrated confidence interval. It excludes absolute
    flux/slit-loss calibration and wavelength-correlated covariance.
    """
    t=np.asarray(transmission)
    if np.any(t<=0):raise ValueError('Nonpositive transmission must be masked, not divided')
    corrected=flux/t;stat=error/t
    total=np.hypot(stat,np.abs(corrected)*relative_model_error)
    return corrected,stat,total
