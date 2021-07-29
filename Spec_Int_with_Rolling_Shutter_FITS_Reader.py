from scipy.ndimage import gaussian_filter
import numpy  as np
import matplotlib.pyplot as plt
import time
from utils import *
import copy
from hcipy import *
import os.path
import csv
from astropy.io import fits
from processing import *

### INPUT PARAMETERS BEGIN HERE ###

## Metadata
name = "Test_Prime_Wind_Test" # Name you want to use to save images
fits_name = "Test_Prime_BSI_Fast_Wind.fits" # Type name of rolling shutter FITS file you want to read

second_shutter_test = False # TRUE if you wish to load a second shutter FITS simultaneously and overlay those results
second_fits_name = "Test_Prime_BSI_Slow_Wind.fits" # Name of Second Shutter Data. Is not used if second_shutter_test = False
sigma = 5 # determines contrast level, e.g. sigma=5 --> 5-sigma contrast curve
### INPUT PARAMETERS END HERE ###

## Generate name strings
radial_name = name + '_Radial_Profile.jpg'
speckle_image_name = name + "_Speckle_Image.jpg"
contrast_curve_name = name + "_Contrast_Curve.jpg"

## If Second True, Generate name strings
if second_shutter_test == True:
    second_radial_name = name + '_Radial_Profile_with_Second_Overlay.jpg'
    second_speckle_image_name = name + "_Speckle_Images_with_Second.jpg"
    second_contrast_curve_name =  name + "_Contrast_Curve_with_Second_Overlay.jpg"


## Load the FITS fileprint("Loading FITS File")
fits_file = fits.open(fits_name)

## If Second True, Load Second FITS File
if second_shutter_test == True:
   second_fits_file = fits.open(second_fits_name) 

## Loading Necessary Values from Header
wavelength = fits_file[0].header['Wavelnth']
wavelength = wavelength.split(' ')
wavelength = float(wavelength[0])
pupil_diameter = fits_file[0].header['PUPDIAMT']
pupil_diameter = pupil_diameter.split(' ')
pupil_diameter = float(pupil_diameter[0])
mag = fits_file[0].header['PRIMAG']
mag = mag.split(' ')
mag = float(mag[0])
q = fits_file[0].header['Q']
q = q.split(' ')
q = float(q[0])

## Generating Sample Image
im1 = fits_file[1].data
if second_shutter_test == True:
    im2 = second_fits_file[1].data
plt.imshow(im1, vmax = 3000)
plt.xlim(200,600)
plt.ylim(200,600)
plt.colorbar()
plt.savefig(name + "_Unprocessed_Image_Comparison.png")

input("Check image")


## Combine Image Data
print("Combining Image Data")
npix = int(np.sqrt(np.prod(fits_file[1].data.shape)))
ims_out = []
for i in range(1, len(fits_file)):
    im = fits_file[i].data
    im_out = im.copy().reshape([npix,npix])
    ims_out.append(np.array(im_out))
ims_out_a = np.array(ims_out)

## If Second, Combine Second Image Data
if second_shutter_test == True:    
    print("Combining Second Image Data")
    second_npix = int(np.sqrt(np.prod(second_fits_file[1].data.shape)))
    ims_out = []
    for i in range(1, len(second_fits_file)):
        im = second_fits_file[i].data
        im_out = im.copy().reshape([second_npix,second_npix])
        ims_out.append(np.array(im_out))
    second_ims_out_a = np.array(ims_out)

## Image Processing
# includes preprocessing, taking FTs, power spectra, and ACFs

# Function Parameters - see processing.py for more detail
# ims           - input image array
# ims_ft        - input FT array
# gsigma        - std deviation for the Gaussian kernel
# subframe_size - final image size in pixels
# HWHM          - half-wavelength at half maximum for supergaussian window
# m             - order of supergaussian window
# scaling       - determines radial cutoff (fcut) for PS

print("Performing Image Preprocessing")
ims_p = image_preprocessing(ims_out_a, 10, 550)
if second_shutter_test == True:
    print("Performing Second Image Preprocessing")
    second_ims_p = image_preprocessing(second_ims_out_a, 10, 550)
#              parameters: (ims, gsigma, subframe_size)
print("Taking Fourier Transform")
FT = fourier_transform(ims_p, 100, 4)
if second_shutter_test == True:
    print("Taking Second Fourier Transform")
    second_FT = fourier_transform(second_ims_p, 100, 4)
#              parameters: (ims, HWHM, m)
print("Taking Power Spectrum")
PS, Av_PS = power_spectrum(FT, q, wavelength, pupil_diameter, 1.)
if second_shutter_test == True:
    print("Taking Second Power Spectrum")
    second_PS, second_Av_PS = power_spectrum(second_FT, q, wavelength, pupil_diameter, 1.)
#              parameters: (ims_ft, wavelength, pupil_diameter, scaling, HWHM, m)
print("Performing Auto Correlation Function")
ACF = generate_ACF(Av_PS)
if second_shutter_test == True:
   print("Performing Second Auto Correlation Function")
   second_ACF = generate_ACF(second_Av_PS) 
#              parameters: (ims_ps)

## Generating Radial Data
print("Generating Radial Data")
from radial_profile import radial_data
rad_stats = radial_data(PS[0])
if second_shutter_test == True:
    second_rad_stats = radial_data(second_PS[0])
f = plt.figure(figsize = (10,8))
plt.plot(rad_stats.r, rad_stats.mean, label = name + "_First_Shutter")
if second_shutter_test == True:
    plt.plot(second_rad_stats.r, second_rad_stats.mean, label = name + "_Second_Shutter")
plt.legend(loc='upper right')
plt.xlabel('Radial coordinate')
plt.ylabel('Mean')
if second_shutter_test == True: 
    plt.savefig(second_radial_name)
else:
    plt.savefig(radial_name)

## Final Image, PS, and ACF
# Image, PS, and ACF plots
print("Generating Image, PS, and ACF Plots")
if second_shutter_test == True:
    f = plt.figure(figsize=(30,20))
else:
    f = plt.figure(figsize=(15,5))
if second_shutter_test == True:
    ax=f.add_subplot(231)
    plt.imshow(ims_p[0])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('Rolling_Shutter_Image')

    ax = f.add_subplot(232)
    plt.title('Rolling_Shutter_Power Spectrum')
    plt.imshow(np.abs(PS[0]))
    ax.set_yticks([])
    ax.set_xticks([])

    fsub = 30
    ax=f.add_subplot(233)
    plt.imshow(np.abs(ACF[0])[int(550/2)-fsub:int(550/2)+fsub,int(550/2)-fsub:int(550/2)+fsub])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('Rolling_Shutter_Autocorrelation')

    ax=f.add_subplot(234)
    plt.imshow(second_ims_p[0])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('Second_Shutter_Image')

    ax = f.add_subplot(235)
    plt.title('Second_Shutter_Power Spectrum')
    plt.imshow(np.abs(second_PS[0]))
    ax.set_yticks([])
    ax.set_xticks([])

    fsub = 30
    ax=f.add_subplot(236)
    plt.imshow(np.abs(second_ACF[0])[int(550/2)-fsub:int(550/2)+fsub,int(550/2)-fsub:int(550/2)+fsub])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('Second_Shutter_Autocorrelation')
    plt.savefig(second_speckle_image_name)
else:
    ax=f.add_subplot(131)
    plt.imshow(ims_p[0])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('Image')

    ax = f.add_subplot(132)
    plt.title('Power Spectrum')
    plt.imshow(np.abs(PS[0]))
    ax.set_yticks([])
    ax.set_xticks([])

    fsub = 30
    ax=f.add_subplot(133)
    plt.imshow(np.abs(ACF)[int(550/2)-fsub:int(550/2)+fsub,int(550/2)-fsub:int(550/2)+fsub])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('Autocorrelation')
    plt.savefig(speckle_image_name)


## Speckle Contrast Curve
print("Generating Contrast Curves")
plate_scale = wavelength / (pupil_diameter * q) * 206265. # (arcsec/pixel)
rad_ACF = radial_data(np.abs(ACF), annulus_width=2)
ACF_cc = -2.5 * np.log10((1. - np.sqrt(1. - (2 * (sigma * rad_ACF.std)) ** 2)) / (2 * (sigma * rad_ACF.std)))
ACF_xax = np.array(range(len(rad_ACF.mean))) * plate_scale # arcsec 

if second_shutter_test == True:
    plate_scale = wavelength / (pupil_diameter * q) * 206265. # (arcsec/pixel)
    second_rad_ACF = radial_data(np.abs(second_ACF), annulus_width=2)
    second_ACF_cc = -2.5 * np.log10((1. - np.sqrt(1. - (2 * (sigma * second_rad_ACF.std)) ** 2)) / (2 * (sigma * second_rad_ACF.std)))
    second_ACF_xax = np.array(range(len(rad_ACF.mean))) * plate_scale # arcsec 

f = plt.figure(figsize=(10,8))
plt.plot(ACF_xax, ACF_cc, label='First Shutter, V = ' + str(mag) + ' mag', lw=3)
if second_shutter_test == True:
    plt.plot(ACF_xax, ACF_cc, label='Second Shutter, V = ' + str(mag) + ' mag', lw=3)
plt.xlim(0.0, 2.0)
plt.gca().invert_yaxis()
plt.legend(loc='lower left')
plt.ylabel(r'' + str(sigma) + ' $\sigma$ Contrast (mag)')
plt.xlabel(r'Separation (arcsec)')
plt.title('VIPER Conventional Speckle')
plt.show()

## Writing Contrast Curve Data into CSV file
print("Writing Contrast Curve Data")
if second_shutter_test == False:
    csv_name = name + '_Contrast_Curve_Data.csv'
else:
    csv_name = name + '_Contrast_Curve_Data_with_Second.csv'
with open(csv_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(["rolling xax", ACF_xax])
    writer.writerow(["rolling cc",ACF_cc])
    if second_shutter_test == True:
        writer.writerow(["second xax", second_ACF_xax])
        writer.writerow(["second cc", second_ACF_cc])

## Saving Contrast Curve
if second_shutter_test == True:
    plt.savefig(second_contrast_curve_name, dpi=300)
else:
    plt.savefig(contrast_curve_name, dpi=300)



