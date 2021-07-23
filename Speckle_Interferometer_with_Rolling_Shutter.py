#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy  as np
import matplotlib.pyplot as plt
import time
from utils import *
import copy
from hcipy import *
import os.path
from astropy.io import fits


# ## Initial Values for Rolling Shutter

# In[2]:

#Readout speed of the detector in frames per second
fps = 522
# width of shortest side of detector. Number of rows in the detector
detector_size = 512 
#Number of rows that subdivide the detector to simulate a rolling shutter. 1 = global shutter. Detector size = most realistic and computationally expensive
number_of_subdivisions = 4

#Properties of the Focal Grid
q=2
nairy = 200

# Exposure time and total number of exposures
exposure_time = 0.01 # (seconds)
exposure_total = 3

# ## Setting Up the Atmosphere
# seeing estimated from the following source: 
# https://mthamilton.ucolick.org/techdocs/MH_weather/obstats/avg_seeing.html

# In[5]:
seeing = 1.75
outer_scale = 40. # (meter) --> GUESS, NEEDS REFINING
velocity = 20. # (m/s) --> GUESS, NEEDS REFINING
                                    
# ## Telescope Setup
# Starting with the Magellan pupil (scaled to 3.048 m) till we get a description of the Shane pupil 

# In[3]:


pupil_diameter = 3.048 # (meter)
collecting_area = np.pi * (3.048**2 - 0.9779**2)

f_number = 13 # effective focal ratio
effective_focal_length = pupil_diameter * f_number # (meter)

filter_name = 'V'
wavelength = filters[filter_name]['lambda_eff'] * 1e-6 # (meter)


# In[4]:


# Generating the pupil grid
grid_size = 256

pupil_grid = make_pupil_grid(grid_size, diameter=pupil_diameter)

# Adjust spiders to represent Shane pupil
telescope_pupil_generator = make_lick_aperture()
telescope_pupil = telescope_pupil_generator(pupil_grid)

# In[6]:
fried_parameter = seeing_to_fried_parameter(seeing, wavelength)
                             
Cn_squared = Cn_squared_from_fried_parameter(fried_parameter, wavelength)

tau0 = 0.314 * fried_parameter/velocity

# Generating phase screens
layer = InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)

phase_screen_phase = layer.phase_for(wavelength) # (radian)
phase_screen_opd = phase_screen_phase * (wavelength / (2 * np.pi)) * 1e6



# In[7]:
# Propagating to focal plane

focal_grid = make_focal_grid(q=q, 
                             num_airy=nairy,
                             pupil_diameter=pupil_diameter,
                             focal_length = effective_focal_length,
                             reference_wavelength=wavelength)

prop = FraunhoferPropagator(pupil_grid, focal_grid, 
                            focal_length=effective_focal_length)


# ## Add a Primary and Companion

# In[8]:


# Primary parameters
mag = 5 #Vega magnitude in the band of interest
stokes_vector= [1.,0.,0.,0.] #I, Q, U, V

# Companion parameters
contrast = 0.
stokes_ps = [1.,0.,0.,0.] #I, Q, U, V
angular_separation= 2 #Lambda/D


# ### Simulating Image Data

# In[9]:


from scipy.ndimage import gaussian_filter

def center_image(img, gsigma, cpix):
#   [img] - image array
#   [gsigma] - standard deviation for Gaussian kernel
#   [cpix] - number of pixels in output image 
    im_o_g = gaussian_filter(img, sigma=gsigma)
    maximum = np.where(im_o_g == np.max(im_o_g))
    #print(maximum)
    #print(type(img))
    #print(type(im_o_g))
    #imshow_field(img)
    #plt.imshow(im_o_g)
    x1 = int(maximum[0] - (cpix/2))
    x2 = int(maximum[0] + (cpix/2))
    
    y1 = int(maximum[1] - (cpix/2))
    y2 = int(maximum[1] + (cpix/2))
    
    center_image = img[x1:x2, y1:y2]
    return(center_image)


# In[10]:


pupil_wavefront = Wavefront(telescope_pupil, wavelength,
                            input_stokes_vector=stokes_vector)
pupil_wavefront.total_power = number_of_photons(mag,filter_name,collecting_area,) #In photons/s


wf_planet = Wavefront(telescope_pupil*np.exp(4j*np.pi*pupil_grid.x*angular_separation/pupil_diameter),
                      wavelength,
                      input_stokes_vector=stokes_ps)
wf_planet.total_power = contrast * number_of_photons(mag,filter_name,collecting_area,) # (photons/s)


# In[11]:
##Create the FITS file
hdr = fits.Header()
hdr['Title'] = 'Speckle Rolling Shutter Test Images'
hdr['COMMENT'] = "This fits contains image data for testing a rolling shutter for speckle imaging. Written by Kyle Lam."
empty_primary = fits.PrimaryHDU(header=hdr)
hdul = fits.HDUList([empty_primary])

##Create the rolling shutter image
detector = NoisyDetector(focal_grid, dark_current_rate=0, read_noise=0, flat_field=0, include_photon_noise=True)
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


##Write the FITS file\
while True:
    print("Type a name in for this fits file.")
    name = input('Name:')
    name = name + '.fits'
    command = 0
    if os.path.isfile(name):
        print("File name already exists")
    else:
        break
hdul.writeto(name)
   

