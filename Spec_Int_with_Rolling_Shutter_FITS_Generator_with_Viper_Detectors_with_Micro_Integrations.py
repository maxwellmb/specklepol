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
#import winsound

### INPUT PARAMETERS BEGIN HERE ###
## Metadata
name = "Marana_Mag_5_Band_V" # filename you want to write, no extension
overwrite = False # True if you wish to overwrite files with the same name
detector_name = "Marana" 

## Properties for EMCCD
EM_gain = None # Set EM Gain for EMCCDs. If running a detector with no EM Gain, set = None
EM_saturation = None # Set behavior when full well depth is reached. None means saturated pixels will be automatically set to full well depth. np.nan means saturated pixels will be set to np.nan

## Properties of the Focal Grid
q=2 # Number of pixels per resolution element
nairy = 200 #The spatial extent of the grid radius in resolution elements (=lambda f/D)

## Exposure time and total number of exposures
exposure_time = 0.01 # Exposure time in seconds. Make sure this is greater than 1/FPS
exposure_total = 1000 # Total number of exposures
micro_integration_time = 0.001

## Setting Up the Atmosphere
seeing = 0.6
outer_scale = 40. # (meter) 
velocity = 20. # (m/s) 
                                    
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
call_detector = "viper_detector." + detector_name + "(focal_grid, " + f'"{filter_name}"'
if EM_gain == None:
    call_detector += ")"
else:
    call_detector += ", " + str(EM_gain)+ ", " + str(EM_saturation) +")"
print(call_detector)
detector = eval(call_detector)                        

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
print("Creating the FITS file")
hdr = fits.Header()
hdr['Title'] = name
hdr['Author'] = "Written by Kyle Lam."
hdr['DetName'] = detector_name + " # Name of detector"
hdr['DetType'] = str(detector.detector_type) + " # Type of detector"
hdr['FPS'] = str(detector.max_fps) + " # Readout speed of the detector in fps"
hdr['DetSize'] = str(detector.detector_size) + " # width of shortest side of detector."
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
hdr['QE'] = str(detector.QE) + " # Quantum Efficiency"
hdr['DarkCurr'] = str(detector.dark_current_rate) + " # Dark Current"
hdr['RdNoise'] = str(detector.read_noise) + " # Read Noise"
hdr['FltField'] = str(detector.flat_field) + "# Flat Field"
hdr['PhtNoise'] = str(detector.include_photon_noise) + " # Photon Noise"
hdr['PriMag'] = str(mag) + " # Magnitude of Primary"
hdr['PriStoke'] = str(stokes_vector) + " # Stokes Vector of Primary"
hdr['Contrast'] = str(contrast) + " # Companion Contrast"
hdr['ComStoke'] = str(stokes_ps) + " # Stokes Vector of Companion"
hdr['AngSep'] = str(angular_separation) + " # Angular Separation"
hdr['Wavelnth'] = str(wavelength) + " # Wavelength"
hdr['Shutter'] = str(detector.shutter_type) + " # Shutter Type"
try:
	hdr['EMGain'] = str(detector.EM_gain) + " # EM Gain Multiplier"
except AttributeError:
	pass
try:
	hdr['EMSat'] = str(detector.EM_saturate) + " # EM Saturate Behavior"
except AttributeError:
	pass
try:
	hdr['FullWell'] = str(detector.full_well_depth) + " # Full Well Depth"
except AttributeError:
	pass
empty_primary = fits.PrimaryHDU(header=hdr)
hdul = fits.HDUList([empty_primary])

## Create the shutter image
print("Generating the Shutter Image")
for i in range(exposure_total):
    print("Percent complete = ", round((i+1)/exposure_total * 100, 3), end = '\r')
    for j in range(int(exposure_time/micro_integration_time)):
        layer.t += micro_integration_time
        detector.integrate(prop((layer(pupil_wavefront))),micro_integration_time)
        detector.integrate(prop((layer(wf_planet))),micro_integration_time) 
    image_comb = detector.read_out()
    image_hdu = fits.ImageHDU(image_comb.shaped)
    hdul.append(image_hdu)

## Write the FITS file
print()
print("Writing the FITS file")
hdul.writeto(fits_name)
header_name = name + "_Header.txt"
hdul[0].header.totextfile(header_name, endcard = False, overwrite = True)
print("FITS file", fits_name,  "generated.")
#winsound.Beep(1000, 300)    