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
name = "Noisy_Simulation_Prime_BSI" # Name you want to use to save images
fits_name = "Noisy_Rolling_Shutter_Simulation_Prime_BSI.fits" # Type name of rolling shutter FITS file you want to read

global_shutter_test = True # TRUE if you wish to load global shutter FITS simultaneously and overlay those results
global_fits_name = "Noisy_Global_Shutter_Simulation_Prime_BSI.fits" # Name of Global Shutter Data. Is not used if global_shutter_test = False

### INPUT PARAMETERS END HERE ###

## Generate name strings
radial_name = name + '_Radial_Profile.jpg'
speckle_image_name = name + "_Speckle_Image.jpg"
contrast_curve_name = name + "_Contrast_Curve.jpg"

## If Global, Generate name strings
if global_shutter_test == True:
    global_radial_name = name + '_Radial_Profile_with_Global_overlay.jpg'
    global_speckle_image_name = name + "_Speckle_Images_with_Global.jpg"
    global_contrast_curve_name =  name + "_Contrast_Curve_with_Global_Overlay.jpg"


## Load the FITS fileprint("Loading FITS File")
fits_file = fits.open(fits_name)

## If Global True, Load Global FITS File
if global_shutter_test == True:
   global_fits_file = fits.open(global_fits_name) 

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

## Combine Image Data
print("Combining Image Data")
npix = int(np.sqrt(np.prod(fits_file[1].data.shape)))
ims_out = []
for i in range(1, len(fits_file)):
    im = fits_file[i].data
    im_out = im.copy().reshape([npix,npix])
    ims_out.append(np.array(im_out))
ims_out_a = np.array(ims_out)

## If Global, Combine Global Image Data
if global_shutter_test == True:    
    print("Combining Global Image Data")
    global_npix = int(np.sqrt(np.prod(global_fits_file[1].data.shape)))
    ims_out = []
    for i in range(1, len(global_fits_file)):
        im = global_fits_file[i].data
        im_out = im.copy().reshape([global_npix,global_npix])
        ims_out.append(np.array(im_out))
    global_ims_out_a = np.array(ims_out)

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
if global_shutter_test == True:
    print("Performing Global Image Preprocessing")
    global_ims_p = image_preprocessing(global_ims_out_a, 10, 550)
#              parameters: (ims, gsigma, subframe_size)
print("Taking Fourier Transform")
FT = fourier_transform(ims_p, 100, 4)
if global_shutter_test == True:
    print("Taking Global Fourier Transform")
    global_FT = fourier_transform(global_ims_p, 100, 4)
#              parameters: (ims, HWHM, m)
print("Taking Power Spectrum")
PS = power_spectrum(FT, wavelength, pupil_diameter, 1.)
if global_shutter_test == True:
    print("Taking Global Power Spectrum")
    global_PS = power_spectrum(global_FT, wavelength, pupil_diameter, 1.)
#              parameters: (ims_ft, wavelength, pupil_diameter, scaling, HWHM, m)
print("Performing Auto Correlation Function")
ACF = generate_ACF(PS)
if global_shutter_test == True:
   print("Performing Global Auto Correlation Function")
   global_ACF = generate_ACF(global_PS) 
#              parameters: (ims_ps)

## Generating Radial Data
print("Generating Radial Data")
from radial_profile import radial_data
rad_stats = radial_data(PS[0])
if global_shutter_test == True:
    global_rad_stats = radial_data(global_PS[0])
f = plt.figure(figsize = (10,8))
plt.plot(rad_stats.r, rad_stats.mean, label = name + "_Rolling_Shutter")
if global_shutter_test == True:
    plt.plot(global_rad_stats.r, global_rad_stats.mean, label = name + "_Global_Shutter")
plt.legend(loc='upper right')
plt.xlabel('Radial coordinate')
plt.ylabel('Mean')
if global_shutter_test == True: 
    plt.savefig(global_radial_name)
else:
    plt.savefig(radial_name)

## Final Image, PS, and ACF
# Image, PS, and ACF plots
print("Generating Image, PS, and ACF Plots")
if global_shutter_test == True:
    f = plt.figure(figsize=(30,20))
else:
    f = plt.figure(figsize=(15,5))
if global_shutter_test == True:
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
    plt.imshow(global_ims_p[0])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('Global_Shutter_Image')

    ax = f.add_subplot(235)
    plt.title('Global_Shutter_Power Spectrum')
    plt.imshow(np.abs(global_PS[0]))
    ax.set_yticks([])
    ax.set_xticks([])

    fsub = 30
    ax=f.add_subplot(236)
    plt.imshow(np.abs(global_ACF[0])[int(550/2)-fsub:int(550/2)+fsub,int(550/2)-fsub:int(550/2)+fsub])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('Global_Shutter_Autocorrelation')
    plt.savefig(global_speckle_image_name)
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
    plt.imshow(np.abs(ACF[0])[int(550/2)-fsub:int(550/2)+fsub,int(550/2)-fsub:int(550/2)+fsub])
    ax.set_yticks([])
    ax.set_xticks([])
    plt.title('Autocorrelation')
    plt.savefig(speckle_image_name)


## Speckle Contrast Curve
# Grid: 800 pixels across, equalling 200 lambda / D, plate scale is then 0.25*lambda / D per pixel
plate_scale = 0.25 * wavelength / pupil_diameter * 206265.   #of image in (arcsec / pixel)

# Plate scale in meters per pixel
ps_mpp = 1. / (npix * plate_scale) * 206265. * wavelength 
scaling = 0.5
fcut = pupil_diameter / ps_mpp * scaling
#Generating contrast curves
print("Generating Contrast Curves")
ACF_m = ACF_cc(ACF[0])
rad_ACF = radial_data(np.abs(ACF[0]), annulus_width=2)
cc = ACF_cc(5*rad_ACF.std)
xax = np.array(range(len(rad_ACF.mean))) * plate_scale / ((wavelength) / pupil_diameter * 206265) #in lambda/D units
if global_shutter_test == True:
    global_ACF_m = ACF_cc(global_ACF[0])
    global_rad_ACF = radial_data(np.abs(global_ACF[0]), annulus_width=2)
    global_cc = ACF_cc(5*global_rad_ACF.std)
    global_xax = np.array(range(len(global_rad_ACF.mean))) * plate_scale / ((wavelength) / pupil_diameter * 206265) #in lambda/D units
f = plt.figure(figsize=(10,8))
plt.plot(xax,cc,label=name + '_Rolling_Shutter, V = '+str(mag)+' mag',lw=3)
if global_shutter_test == True:
    plt.plot(global_xax,global_cc,label=name + '_Global_Shutter, V = '+str(mag)+' mag',lw=3)
plt.xlim(0.0,20.0)
plt.gca().invert_yaxis()
plt.legend(loc='lower left')
plt.ylabel(r'5$\sigma$ Contrast (mag)')
plt.xlabel(r'Separation ($\lambda$ / D)')
plt.title('VIPER Conventional Speckle: 100 x 10ms')

## Writing Contrast Curve Data into CSV file
print("Writing Contrast Curve Data")
if global_shutter_test == False:
    csv_name = name + '_Contrast_Curve_Data.csv'
else:
    csv_name = name + '_Contrast_Curve_Data_with_Global.csv'
with open(csv_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(["rolling xax", xax])
    writer.writerow(["rolling cc", cc])
    if global_shutter_test == True:
        writer.writerow(["global xax", global_xax])
        writer.writerow(["global cc", global_cc])

## Saving Contrast Curve
if global_shutter_test == True:
    plt.savefig(global_contrast_curve_name, dpi=300)
else:
    plt.savefig(contrast_curve_name, dpi=300)



