import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from utils import supergauss_hw
from radial_profile import radial_data

"""
summary of included functions: 
    image_preprocessing - centers images and subtracts background noise
    fourier_transform   - takes the Fourier transform of each image in array
    power_spectrum      - generates power spectra for image array, given FTs
    generate_ACF        - generates an array of autocorrelation functions given power spectra 
    ACF_contrast        - returns speckle contrast curve for the given autocorrelation function
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
    bpix = npix - int(0.1 * npix)
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
        im_ft = im.copy()

        FT = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(im_ft * sg)))
        ims_ft_out.append(FT)

    return np.array(ims_ft_out, dtype=complex)


def power_spectrum(ims_ft, q, wavelength, pupil_diameter, scaling=1.):
    """
    A function that returns two outputs: a bias-subtracted, filtered array of power spectra, calculated using the input
    FTs and utilizing radial data, and the average power spectrum.

    Parameters
    ----------
    ims_ft          - input array of FTs
    q               - the number of pixels per resolution element (= lambda f / D)
    wavelength      - wavelength of the wavefront in meters
    pupil_diameter  - pupil diameter in meters
    scaling         - scalar that determines radial cutoff (fcut) ; (default: 1.)

    Returns
    -------
    ps_out      - a three dimensional array of the generated power spectra
    avg_ps_out  - a two dimensional array of the average power spectrum
    """

    npix = len(ims_ft[0])
    bpix = npix - int(0.1 * npix)

    ps_out = []

    plate_scale = wavelength / (pupil_diameter * q) * 206265.
    ps_mpp = 1. / (npix * plate_scale) * 206265. * wavelength # (meters per pixel)
    fcut = pupil_diameter / ps_mpp * scaling
    for i in range(len(ims_ft)):
        im_ft = ims_ft[i]
        im_ft_in = im_ft.copy()

        PS = np.abs(im_ft_in / im_ft_in[int(npix / 2), int(npix / 2)]) ** 2

        PS_bsub = PS.copy()
        PS_bsub -= np.mean(PS[(bpix):, (bpix):])

        rad_stats = radial_data(PS_bsub)
        PS_filt = PS_bsub.copy()

        for xx in range(len(PS_filt)):
            for yy in range(len(PS_filt)):
                rad = np.sqrt((xx - npix / 2) ** 2 + (yy - npix / 2) ** 2)
                if rad < fcut:
                    drad = rad_stats.r - rad
                    val = rad_stats.mean[np.where(np.abs(drad) == np.min(np.abs(drad)))][0]
                    PS_filt[yy, xx] /= val
        ps_out.append(PS_filt)

    avg_ps = np.sum(ps_out, axis=0)

    return np.array(ps_out), avg_ps


def generate_ACF(ims_ps):
    """
    A function that, upon receiving the average power spectrum, returns the corresponding normalized average
    autocorrelation function.

    Parameters
    ----------
    ims_ps      - a two dimensional input array of the average power spectrum

    Returns
    -------
    ims_acf_out - a two dimensional array of the average autocorrelation function
    """

    npix = len(ims_ps[0])

    im_ps = ims_ps.copy()

    ACF_filt = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im_ps)))
    ACF_norm = ACF_filt / ACF_filt[int(npix / 2), int(npix / 2)]

    return ACF_norm


def ACF_contrast(ACF, q, wavelength, pupil_diameter, magnitude, sigma=5, figure_size=(5, 4)):
    """
    A function that generates an array of suitable ACFs to generate contrast curves.

    Parameters
    ----------
    ACF            - input array of ACFs
    q              - the number of pixels per resolution element (= lambda f / D)
    wavelength     - wavelength of the wavefront in meters
    pupil_diameter - pupil diameter in meters
    magnitude      - object magnitude
    figure_size    - size of returned plot ; (optional, default:(5,4) )
    sigma          - determines contrast level, e.g. sigma=5 --> 5-sigma contrast curve
                     (optional, default: 5)
    Returns
    -------

    """
    plate_scale = wavelength / (pupil_diameter * q) * 206265. # (arcsec/pixel)
    rad_ACF = radial_data(np.abs(ACF), annulus_width=2)
    ACF_cc = -2.5 * np.log10((1. - np.sqrt(1. - (2 * (sigma * rad_ACF.std)) ** 2)) / (2 * (sigma * rad_ACF.std)))
    ACF_xax = np.array(range(len(rad_ACF.mean))) * plate_scale # arcsec 

    f = plt.figure(figsize=figure_size)
    plt.plot(ACF_xax, ACF_cc, label='V = ' + str(magnitude) + ' mag', lw=3)
    #plt.xlim(0.0, 2.0)
    plt.gca().invert_yaxis()
    plt.legend(loc='lower left')
    plt.ylabel(r'' + str(sigma) + ' $\sigma$ Contrast (mag)')
    plt.xlabel(r'Separation (arcsec)')
    plt.title('VIPER Conventional Speckle')
    plt.show()



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
        thetas[x] = thetas[x] * theta[x]
    num = np.mean((f_L - f_R) * (f_L + f_R).conj() * np.cos(h * np.radians(thetas)), axis=0)
    den = np.mean((f_L + f_R) * (f_L + f_R).conj(), axis=0) - 1.0 / N_e

    R_Q = 1 + num / den
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
    R = R_in.copy()

    IFT_R = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(R - 1.0)))
    R_out = np.real(IFT_R)

    return R_out
