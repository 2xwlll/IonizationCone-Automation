"""Two-dimensional slit extraction with Gaussian seeing and sky coordinates."""
import numpy as np
from scipy.special import ndtr


def sky_offsets(shape,pixel_scale,matrix,center=None):
    """East/north offsets using the renderer's normalized local WCS matrix."""
    yy,xx=np.indices(shape,dtype=float)
    cx,cy=((shape[1]-1)/2,(shape[0]-1)/2) if center is None else center
    m=np.asarray(matrix,float);m=m/np.linalg.norm(m[:,0])
    return (m[0,0]*(xx-cx)+m[0,1]*(yy-cy))*pixel_scale,(m[1,0]*(xx-cx)+m[1,1]*(yy-cy))*pixel_scale


def gaussian_slit_acceptance(east,north,*,pa_deg,along_center,cross_center,width,length,fwhm):
    """Fraction of each point source's Gaussian PSF admitted by a rectangle.

    Position angle is east of north, distances arcsec. Evaluate on the intrinsic
    source grid; this replaces rather than compounds the imaging PSF.
    """
    if width<=0 or length<=0 or fwhm<0:raise ValueError('Invalid slit or seeing size')
    a=np.deg2rad(pa_deg)
    along=np.asarray(east)*np.sin(a)+np.asarray(north)*np.cos(a)-along_center
    cross=np.asarray(east)*np.cos(a)-np.asarray(north)*np.sin(a)-cross_center
    if fwhm==0:return ((abs(along)<=length/2)&(abs(cross)<=width/2)).astype(float)
    sigma=fwhm/2.354820045
    return ((ndtr((length/2-along)/sigma)-ndtr((-length/2-along)/sigma))*
            (ndtr((width/2-cross)/sigma)-ndtr((-width/2-cross)/sigma)))
