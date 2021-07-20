import numpy as np
from scipy.ndimage import gaussian_filter
from utils import supergauss_hw
from radial_profile import radial_data

"""
summary of included functions: 
    image_preprocessing - centers images and subtracts background noise
    fourier_transform   - takes the Fourier transform of each image in array
    power_spectrum      - generates power spectra for image array, given FTs
    generate_ACF        - generates an array of autocorrelation functions given power spectra 
    ACF_cc              - modifies ACF array for contrast curves
    R_U                 - returns the differential polarimetric visibility, R,  for Stokes U ; given FTs
    R_Q                 - returns the differential polarimetric visibility, R,  for Stokes Q ; given FTs
    IFT                 - returns Stokes Q or U vector, given R
"""


def image_preprocessing(ims, gsigma, subframe_size):
    """
    A function that subtracts background noise and centers each image in the input array.

    Parameters
    ----------
    ims             - the array of images to be processed
    gsigma          - standard deviation for Gaussian kernel
    subframe_size   - the desired size of final image in pixels ; (e.g. 600 -yields-> image of 600x600 pixels)
                      (note: subframe_size must be less pixels than the input image)

    Returns
    -------
    ims_out - a centered, background subtracted array of images
    """

    npix = len(ims[0])
    bpix = npix - int(0.05 * npix)
    ims_out = []

    for i in range(len(ims)):
        im = ims[i]

        im_ft_b = im.copy()
        im_ft_b -= np.median(im[(bpix):, (bpix):])

        im_g = gaussian_filter(im_ft_b, sigma=gsigma)
        maximum = np.where(im_g == np.max(im_g))

        y1 = int(maximum[0][0] - (subframe_size / 2))
        y2 = int(maximum[0][0] + (subframe_size / 2))

        x1 = int(maximum[1][0] - (subframe_size / 2))
        x2 = int(maximum[1][0] + (subframe_size / 2))

        center_image = im_ft_b[y1:y2, x1:x2]

        ims_out.append(center_image)

    return np.array(ims_out)


def fourier_transform(ims, HWHM, m):
    """
    A function that takes the Fourier transform of each image in the given array,
    and also applies a supergaussian window.

    Parameters
    ----------
    ims        - input array of images
    HWHM       - half-width at half maximum for supergaussian window
    m          - desired order of supergaussian window

    Returns
    -------
    ims_ft_out - A fourier transformed array of images with a supergaussian window applied.

    """
    npix = len(ims[0])
    sg = supergauss_hw(HWHM, m, npix)
    ims_ft_out = []

    for i in range(len(ims)):
        im = ims[i]

        # make a copy
        im_ft = im.copy()
        FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im_ft * sg)))
        ims_ft_out.append(FT / FT[int(npix / 2), int(npix / 2)])

    return np.array(ims_ft_out, dtype=complex)


def power_spectrum(ims_ft, wavelength, pupil_diameter, scaling):
    """
    A function that returns a bias-subtracted, filtered array of power spectra, calculated using the input
    FTs and utilizing radial data.

    Parameters
    ----------
    ims_ft          - input array of FTs
    wavelength      - wavelength of the wavefront in meters
    pupil_diameter  - pupil diameter in meters
    scaling         - scalar that determines radial cutoff (fcut)

    Returns
    -------
    ims_ps_out      - an array of the generated power spectra
    """
    ims_ps_out = []
    npix = len(ims_ft[0])


    plate_scale = 0.25 * wavelength / pupil_diameter * 206265.
    ps_mpp = 1. / (npix * plate_scale) * 206265. * wavelength
    fcut = pupil_diameter / ps_mpp * scaling

    for i in range(len(ims_ft)):
        im_ft = ims_ft[i]
        im_ft_in = im_ft.copy()
        PS = np.abs(im_ft_in / im_ft_in[int(npix / 2), int(npix / 2)]) ** 2

        rad_stats = radial_data(PS)

        PS_bsub = PS.copy()
        PS_bsub -= np.mean(PS[int(0.9 * npix), int(0.9 * npix):])

        PS_filt = PS_bsub.copy()

        for xx in range(len(PS_filt)):
            for yy in range(len(PS_filt)):
                rad = np.sqrt((xx - npix / 2) ** 2 + (yy - npix / 2) ** 2)
                if rad < fcut:
                    drad = rad_stats.r - rad
                    val = rad_stats.mean[np.where(np.abs(drad) == np.min(np.abs(drad)))][0]
                    PS_filt[yy, xx] /= val



        ims_ps_out.append(PS_filt)

        return np.array(ims_ps_out)


def generate_ACF(ims_ps):
    """
    A function that, upon receiving an array of power spectra, returns the corresponding normalized
    autocorrelation function.

    Parameters
    ----------
    ims_ps      - input array of power spectra

    Returns
    -------
    ims_acf_out - an array of the generated autocorrelation functions
    """

    npix = len(ims_ps[0])
    ims_acf_out = []

    for i in range(len(ims_ps)):
        ps = ims_ps[i]
        im_ps = ps.copy()

        ACF_filt = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im_ps)))
        ACF_norm = ACF_filt / ACF_filt[int(npix / 2), int(npix / 2)]
        ims_acf_out.append(ACF_norm)

    return np.array(ims_acf_out)

def ACF_cc(ACF):
    """
    A function that generates an array of suitable ACFs to generate contrast curves.

    Parameters
    ----------
    ACF - input array of ACFs

    Returns
    -------
    ACF_ccs_out - an array of ACFs to generate contrast curves
    """
    ACF_ccs_out = -2.5*np.log10((1.-np.sqrt(1.-(2*ACF)**2))/(2*ACF))
    return ACF_ccs_out


def R_U(f_L, f_R, N_e, theta, h):
    """
    A function that generates the differential polarimetric visibility, R, for Stokes U,
    given the array of Fourier transforms of both the extraordinary and ordinary beams.

    Parameters
    ----------
    f_L   -  left side of FTs (numpy array of images)
    f_r   -  right side of FTs (numpy array of images)
    N_e   -  average number of photons in a single frame
    theta -  half-wave plate angle (measured in degrees)
    h     -  number of harmonics

    Returns
    -------
    R_U   -  the differential polarimetric visibility, R, for Stokes U

    """
    for i in range(len(f_L)):
        thetas = np.ones(f_L.shape)
        for x in range(len(thetas)):
            thetas[x] = thetas[x] * theta[x]
        num = np.mean((f_L - f_R) * (f_L + f_R).conj() * np.sin(h * np.radians(thetas)), axis=0)
        den = np.mean((f_L + f_R) * (f_L + f_R).conj(), axis=0) - 1.0 / N_e

        R_U = 1 + num / den
    return R_U


def R_Q(f_L, f_R, N_e, theta, h):
    """
    A function that generates the differential polarimetric visibility, R, for Stokes Q,
    given the array of Fourier transforms of both the extraordinary and ordinary beams.

    Parameters
    ----------
    f_L   -  left side of FTs (numpy array of images)
    f_r   -  right side of FTs (numpy array of images)
    N_e   -  average number of photons in a single frame
    theta -  half-wave plate angle (measured in degrees)
    h     -  number of harmonics

    Returns
    -------
    R_Q   -  the differential polarimetric visibility, R, for Stokes Q
    """
    thetas = np.ones(f_L.shape)
    for x in range(len(thetas)):
        thetas[x] = thetas[x]*theta[x]
    num = np.mean((f_L - f_R) * (f_L + f_R).conj() * np.cos(h*np.radians(thetas)),axis=0)
    den = np.mean((f_L + f_R) * (f_L + f_R).conj(),axis=0) - 1.0/N_e

    R_Q = 1 + num/den
    return R_Q

def IFT(R_in):
    """
    A function that will shift and take the inverse FT of R in order to obtain the corresponding Stokes vector
    (i.e. inputting R_U will yield Stokes U)

    Parameters
    ----------
    R_in  - R_U or R_Q

    Returns
    -------
    R_out - the corresponding Stokes vector

    """
    R_out = []
    R = R_in.copy()

    IFT_R = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(R - 1.0)))
    R_out.append(IFT_R)

    return R_out
