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
import viper_detector

### INPUT PARAMETERS BEGIN HERE ###

## Metadata
name = "Test_Viper_Detectors_iXon" # filename you want to write, no extension
overwrite = True # True if you wish to overwrite files with the same name

## Initial Values for Shutter
detector_name = "iXon887" 
number_of_subdivisions = 32 # Number of Subdivisions to divide up rolling shutter. Leave blank for Global Shutter

## Properties of the Focal Grid
q=2 # Number of pixels per resolution element
nairy = 200 #The spatial extent of the grid radius in resolution elements (=lambda f/D)

## Exposure time and total number of exposures
exposure_time = 0.02 # Exposure time in seconds. Make sure this is greater than 1/FPS
exposure_total = 100 # Total number of exposures

## Setting Up the Atmosphere
seeing = 1.75
outer_scale = 40. # (meter) 
velocity = 30. # (m/s) 
                                    
## Setting up the telescope
pupil_diameter = 3.048 # (meter)
f_number = 13 # effective focal ratio
grid_size = 256 # Number of pixels per dimension
filter_name = 'V' # Name of filter
telescope_pupil_generator = make_lick_aperture()


## Add a Primary and Companion
# Primary parameters
mag = 5 # Magnitude in the band of interest
stokes_vector= [1.,0.,0.,0.] # I, Q, U, V
# Companion parameters
contrast = 0.
stokes_ps = [1.,0.,0.,0.] # I, Q, U, V
angular_separation= 2 # Lambda/D

### INPUT PARAMETERS END HERE ###

## Some math to define additional parameters
collecting_area = np.pi * (3.048**2 - 0.9779**2)
effective_focal_length = pupil_diameter * f_number # (meter)
wavelength = filters[filter_name]['lambda_eff'] * 1e-6 # (meter)

## Generate name strings
fits_name = name + ".fits"

## Checking to see if filenames exists
if os.path.isfile(fits_name):
    if overwrite:
        print("File name ", fits_name, " already exists. Preparing to overwrite.") 
        os.remove(fits_name)
    else:
        print("Error, file name ",fits_name," already exists. Overwrite was not allowed.")
        print("Exiting program...")
        quit()

## Generating the pupil grid
print("Generating the pupil grid")
pupil_grid = make_pupil_grid(grid_size, diameter=pupil_diameter)

## Adjust spiders to represent Shane pupil
print("Generating the telescope pupil")
telescope_pupil = telescope_pupil_generator(pupil_grid)

## Generating the atmosphere
print("Generating the atmosphere")
fried_parameter = seeing_to_fried_parameter(seeing, wavelength)                             
Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, wavelength)
tau0 = 0.314 * fried_parameter/velocity
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)

##Generating the focal grid
print("Generating the focal grid")
focal_grid = make_focal_grid(q=q, 
                             num_airy=nairy,
                             pupil_diameter=pupil_diameter,
                             focal_length = effective_focal_length,
                             reference_wavelength=wavelength)

## Define the Detector
call_detector = "viper_detector." + detector_name + "(focal_grid," + filter_name + ")"
try:
    detector = eval(call_detector)
except:
    print("Error, invalid detector name.")
    print("Exiting...")
    quit()                            

## Generating the propagator
print("Generating the propagator")
prop = FraunhoferPropagator(pupil_grid, focal_grid, 
                            focal_length=effective_focal_length)

## Generating wavefront of primary and companion
print("Generating wavefront of primary and companion")
pupil_wavefront = Wavefront(telescope_pupil, wavelength,
                            input_stokes_vector=stokes_vector)
pupil_wavefront.total_power = number_of_photons(mag,filter_name,collecting_area,)#In photons/s
wf_planet = Wavefront(telescope_pupil*np.exp(4j*np.pi*pupil_grid.x*angular_separation/pupil_diameter),
                      wavelength,
                      input_stokes_vector=stokes_ps)
wf_planet.total_power = contrast * number_of_photons(mag,filter_name,collecting_area,)# (photons/s)

## Create the FITS file
print("Creating the Rolling FITS file")
hdr = fits.Header()
hdr['Title'] = name
hdr['Author'] = "Written by Kyle Lam."
hdr['FPS'] = str(detector.output_fps()) + " # Readout speed of the detector in fps"
hdr['DetSize'] = str(detector.output_detector_size()) + " # width of shortest side of detector."
hdr['q'] = str(q) + " # Number of pixels per resolution element"
hdr['NAiry'] = str(nairy) + " # The spatial extent of the grid radius in resolution elements"
hdr['ExpoTime'] = str(exposure_time) + " # Exposure time in seconds"
hdr['NumExpos'] = str(exposure_total) + " # Total Number of Exposures"
hdr['Seeing'] = str(seeing) + " # Seeing"
hdr['OutScale'] = str(outer_scale) + " # Outer Scale"
hdr['Velocity'] = str(velocity) + " # Wind Velocity"
hdr['PupDiamt'] = str(pupil_diameter) + " # Pupil Diameter"
hdr['FNum'] = str(f_number) + " # F Number"
hdr['GridSize'] = str(grid_size) + " # Grid Size"
hdr['FiltName'] = str(filter_name) + " # Filter Name"
hdr['QE'] = str(detector.output_QE) + " # Quantum Efficiency"
hdr['DarkCurr'] = str(detector.output_dark_current()) + " # Dark Current"
hdr['RdNoise'] = str(detector.output_read_noise()) + " # Read Noise"
hdr['FltField'] = str(detector.output_flat_field()) + "# Flat Field"
hdr['PhtNoise'] = str(detector.output_photon_noise()) + " # Photon Noise"
hdr['PriMag'] = str(mag) + " # Magnitude of Primary"
hdr['PriStoke'] = str(stokes_vector) + " # Stokes Vector of Primary"
hdr['Contrast'] = str(contrast) + " # Companion Contrast"
hdr['ComStoke'] = str(stokes_ps) + " # Stokes Vector of Companion"
hdr['AngSep'] = str(angular_separation) + " # Angular Separation"
hdr['Wavelnth'] = str(wavelength) + " # Wavelength"
hdr['Shutter'] = str(detector.output_shutter_type()) + " # Shutter Type"
empty_primary = fits.PrimaryHDU(header=hdr)
hdul = fits.HDUList([empty_primary])

## Create the shutter image
print("Generating the Shutter Image")
if detector.output_shutter_type() == "Rolling":
    for i in range(exposure_total):
        print("Percent complete = ", round((i+1)/exposure_total * 100, 3), end = '\r')
        rolling_image = detector.roll_shutter([pupil_wavefront, wf_planet], layer, prop, exposure_time, number_of_subdivisions)
        image_hdu = fits.ImageHDU(rolling_image)
        hdul.append(image_hdu)
elif detector.output_shutter_type() == "Global":
    for i in range(exposure_total):
        print("Percent complete = ", round((i+1)/exposure_total * 100, 3), end = '\r')
        layer.t += exposure_time
        detector.integrate(prop((layer(pupil_wavefront))),exposure_time)
        detector.integrate(prop((layer(wf_planet))),exposure_time) 
        image_comb = detector.read_out()
        image_hdu = fits.ImageHDU(image_comb)
        hdul.append(image_hdu)
else:
    print("Error, Check Shutter Type in viper_detectors")
    quit()


## Write the FITS file
print()
print("Writing the FITS file")
hdul.writeto(fits_name)
header_name = name + "_Header.txt"
hdul[0].header.totextfile(header_name, endcard = False, overwrite = True)
print("FITS file", fits_name,  "generated.")
    