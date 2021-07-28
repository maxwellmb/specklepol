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
name = "Test2_Prime_BSI" # filename, no extension
overwrite = True # True if you wish to overwrite files with the same name
global_shutter_test = True # True if you wish to test a global shutter simultaneously and generate a fits file

## Initial Values for Rolling Shutter
fps = 173 # Readout speed of the detector in frames per second
detector_size = 512 # width of shortest side of detector. Number of rows in the detector
number_of_subdivisions = 24 # Number of rows that subdivide the detector to simulate a rolling shutter.

## Properties of the Focal Grid
q=2 # Number of pixels per resolution element
nairy = 200 #The spatial extent of the grid radius in resolution elements (=lambda f/D)

## Exposure time and total number of exposures
exposure_time = 0.01 # Exposure time in seconds
exposure_total = 5 # Total number of exposures

## Setting Up the Atmosphere
seeing = 1.75
outer_scale = 40. # (meter) 
velocity = 20. # (m/s) 
                                    
## Setting up the telescope
pupil_diameter = 3.048 # (meter)
f_number = 13 # effective focal ratio
grid_size = 256 # Number of pixels per dimension
filter_name = 'V' # Name of filter
telescope_pupil_generator = make_lick_aperture()

## Setting up the noisy detector
dark_current = 0.12 # Dark Current
read_noise = 1.1 # Read Noise
flat_field = 0 # Flat Field
photon_noise = True # Photon Noise

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

## If Global, Generate name strings
if global_shutter_test == True:
    global_fits_name = name + "_Global.fits"
    
## Checking to see if filenames exists
if os.path.isfile(fits_name):
    if overwrite:
        print("File name ", fits_name, " already exists. Preparing to overwrite.") 
        os.remove(fits_name)
    else:
        print("Error, file name ",fits_name," already exists. Overwrite was not allowed.")
        print("Exiting program...")
        quit()
## Checking to see if Global filenames exist
if global_shutter_test == True:
    if os.path.isfile(global_fits_name) and global_shutter_test == True:
        if overwrite:
            print("File name ", global_fits_name, " already exists. Preparing to overwrite.") 
            os.remove(global_fits_name)
        else:
            print("Error, file name ",global_fits_name," already exists. Overwrite was not allowed.")
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

## Generating the propagator
print("Generating the propagator")
prop = FraunhoferPropagator(pupil_grid, focal_grid, 
                            focal_length=effective_focal_length)

## Generating wavefront of primary and companion
print("Generating wavefront of primary and companion")
pupil_wavefront = Wavefront(telescope_pupil, wavelength,
                            input_stokes_vector=stokes_vector)
pupil_wavefront.total_power = number_of_photons(mag,filter_name,collecting_area,) #In photons/s
wf_planet = Wavefront(telescope_pupil*np.exp(4j*np.pi*pupil_grid.x*angular_separation/pupil_diameter),
                      wavelength,
                      input_stokes_vector=stokes_ps)
wf_planet.total_power = contrast * number_of_photons(mag,filter_name,collecting_area,) # (photons/s)

## Create the FITS file
print("Creating the FITS file")
hdr = fits.Header()
hdr['Title'] = name
hdr['Author'] = "Written by Kyle Lam."
hdr['FPS'] = str(fps) + " # Readout speed of the detector in fps"
hdr['DetSize'] = str(detector_size) + " # width of shortest side of detector."
hdr['NSubdivi'] = str(number_of_subdivisions) + " # Number of Subdivisions"
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
hdr['DarkCurr'] = str(dark_current) + " # Dark Current"
hdr['RdNoise'] = str(read_noise) + " # Read Noise"
hdr['FltField'] = str(flat_field) + "# Flat Field"
hdr['PhtNoise'] = str(photon_noise) + " # Photon Noise"
hdr['PriMag'] = str(mag) + " # Magnitude of Primary"
hdr['PriStoke'] = str(stokes_vector) + " # Stokes Vector of Primary"
hdr['Contrast'] = str(contrast) + " # Companion Contrast"
hdr['ComStoke'] = str(stokes_ps) + " # Stokes Vector of Companion"
hdr['AngSep'] = str(angular_separation) + " # Angular Separation"
hdr['Wavelnth'] = str(wavelength) + " # Wavelength"
empty_primary = fits.PrimaryHDU(header=hdr)
hdul = fits.HDUList([empty_primary])

## Create the rolling shutter image
print("Generating the rolling shutter image")
detector = NoisyDetector(focal_grid, dark_current_rate= dark_current, 
                        read_noise=read_noise, flat_field=flat_field, 
                        include_photon_noise=photon_noise)
number_of_rows = int(np.sqrt(focal_grid.size))# Width of the focal plane. Number of in rows the FOCAL PLANE.
row_layout = np.linspace(0, number_of_rows, number_of_subdivisions, dtype = "int")
row_readout_rate = 1/fps/detector_size
row_exposure_time = row_readout_rate*number_of_rows/number_of_subdivisions
start_time = time.perf_counter()
layer.t = exposure_time
for i in range(exposure_total):
    layer.t+=exposure_time-(row_exposure_time*number_of_subdivisions)
    detector.integrate(prop((layer(pupil_wavefront))),exposure_time-(row_exposure_time*number_of_subdivisions))
    detector.integrate(prop((layer(wf_planet))),exposure_time-(row_exposure_time*number_of_subdivisions)) 
    image_comb = detector.read_out()
    for j in row_layout:
        layer.evolve_until(row_exposure_time)
        detector.integrate(prop((layer(pupil_wavefront))),row_exposure_time)
        detector.integrate(prop((layer(wf_planet))),row_exposure_time)   
        image_row = detector.read_out()
        start = int(focal_grid.size*(j))
        end = int(focal_grid.size*((j+1)))
        image_comb[start:end]+=image_row[start:end]        
        print("Percent complete = ", round((j+i*int(number_of_rows))/(number_of_rows*exposure_total) * 100, 3), end = '\r')
    image_hdu = fits.ImageHDU(image_comb.shaped)
    hdul.append(image_hdu)

## Write the FITS file
print()
print("Writing the FITS file")
hdul.writeto(fits_name)
header_name = name + "_header.txt"
hdul[0].header.totextfile(header_name, endcard = False, overwrite = True)
print("FITS file", fits_name,  "generated.")
    
## If Include Global Test, Generate Global Image
if global_shutter_test == True:
    print("Generating Global Shutter Image")
    number_of_subdivisions = 1
    detector = NoisyDetector(focal_grid, dark_current_rate= dark_current, 
                        read_noise=read_noise, flat_field=flat_field, 
                        include_photon_noise=photon_noise)
    number_of_rows = int(np.sqrt(focal_grid.size))# Width of the focal plane. Number of in rows the FOCAL PLANE.
    row_layout = np.linspace(0, number_of_rows, number_of_subdivisions, dtype = "int")
    row_readout_rate = 1/fps/detector_size
    row_exposure_time = row_readout_rate*number_of_rows/number_of_subdivisions
    start_time = time.perf_counter()
    layer.t = exposure_time
    for i in range(exposure_total):
        layer.t+=exposure_time-(row_exposure_time*number_of_subdivisions)
        detector.integrate(prop((layer(pupil_wavefront))),exposure_time-(row_exposure_time*number_of_subdivisions))
        detector.integrate(prop((layer(wf_planet))),exposure_time-(row_exposure_time*number_of_subdivisions)) 
        image_comb = detector.read_out()
        for j in row_layout:
            layer.evolve_until(row_exposure_time)
            detector.integrate(prop((layer(pupil_wavefront))),row_exposure_time)
            detector.integrate(prop((layer(wf_planet))),row_exposure_time)   
            image_row = detector.read_out()
            start = int(focal_grid.size*(j))
            end = int(focal_grid.size*((j+1)))
            image_comb[start:end]+=image_row[start:end]        
            print("Percent complete = ", round((j+i*int(number_of_rows))/(number_of_rows*exposure_total) * 100, 3), end = '\r')
        image_hdu = fits.ImageHDU(image_comb.shaped)
        hdul.append(image_hdu)
    
    # Adjust the header of the FITS file
    hdr['NSubdivi'] = str(number_of_subdivisions) + " # Number of Subdivisions"
    hdr['Title'] = global_fits_name
    header_name = name + "_Global_Header.txt"
    hdul[0].header.totextfile(header_name, endcard = False, overwrite = overwrite)
    # Write the Global FITS file
    print()
    print("Writing the Global FITS file")
    hdul.writeto(global_fits_name)
    print("FITS file", global_fits_name,  "generated.")
