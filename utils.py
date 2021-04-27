import numpy as np
from hcipy import *

### UBVRIJHK Filter Info
## Taken from http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
## Flux0 - erg/cm^2/s/A
## photons0 - photons/cm^2/s/A
## lambda_eff - um
## delta_lambda - um 

filters = {
                "U": {"Flux0": 4.175e-9, "lambda_eff":0.36 , "delta_lambda":0.06 , "photons0":756.1},
                "B": {"Flux0": 6.32e-9 , "lambda_eff":0.438, "delta_lambda":0.09 , "photons0":1392.6},
                "V": {"Flux0": 3.631e-9, "lambda_eff":0.545, "delta_lambda":0.085, "photons0":995.5},
                "R": {"Flux0": 2.177e-9, "lambda_eff":0.641, "delta_lambda":0.15 , "photons0":702.9},
                "I": {"Flux0": 1.126e-9, "lambda_eff":0.798, "delta_lambda":0.15 , "photons0":452},
                "J": {"Flux0": 0.315e-9, "lambda_eff":1.22 , "delta_lambda":0.26 , "photons0":193.1},
                "H": {"Flux0": 0.114e-9, "lambda_eff":1.63 , "delta_lambda":0.29 , "photons0":93.3},
                "K": {"Flux0": 0.004e-9, "lambda_eff":2.19 , "delta_lambda":0.41 , "photons0":43.6},
}


def make_lick_aperture(normalized=False, with_spiders=True):
    '''
    This is almost a lick aperture, based on the hcipy make_magellan_aperture

    Make the Magellan aperture.

    Parameters
    ----------
    normalized : boolean
        If this is True, the outer diameter will be scaled to 1. Otherwise, the
        diameter of the pupil will be 6.5 meters.
    with_spiders: boolean
        If this is False, the spiders will be left out.

    Returns
    -------
    Field generator
        The Magellan aperture.
    '''
    pupil_diameter = 3.048 #m
    spider_width1 = 0.75 * 0.0254 #m
    spider_width2 = 1.5 * 0.0254 #m
    secondary_diameter = 0.9779 # 
    central_obscuration_ratio = secondary_diameter/pupil_diameter #
    spider_offset = [0,0.34] #m

    if normalized:
        spider_width1 /= pupil_diameter
        spider_width2 /= pupil_diameter
        spider_offset = [x / pupil_diameter for x in spider_offset]
        pupil_diameter = 1.0

    spider_offset = np.array(spider_offset)

    mirror_edge1 = (pupil_diameter / (2 * np.sqrt(2)), pupil_diameter / (2 * np.sqrt(2)))
    mirror_edge2 = (-pupil_diameter / (2 * np.sqrt(2)), pupil_diameter / (2 * np.sqrt(2)))
    mirror_edge3 = (pupil_diameter / (2 * np.sqrt(2)), -pupil_diameter / (2 * np.sqrt(2)))
    mirror_edge4 = (-pupil_diameter / (2 * np.sqrt(2)), -pupil_diameter / (2 * np.sqrt(2)))

    obstructed_aperture = make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio)

    if not with_spiders:
        return obstructed_aperture

    spider1 = make_spider(spider_offset, mirror_edge1, spider_width1)
    spider2 = make_spider(spider_offset, mirror_edge2, spider_width1)
    spider3 = make_spider(-spider_offset, mirror_edge3, spider_width2)
    spider4 = make_spider(-spider_offset, mirror_edge4, spider_width2)

    def func(grid):
        return obstructed_aperture(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
    return func


def number_of_photons(mag, filter_name, collecting_area):
    '''
    Return the number of photons/s captured by the collecting
    area for the given magnitude and filter
    '''
    
    photons0 = filters[filter_name]['photons0'] #units: photons/cm^2/s/A
    
    #Convert to the correct magnitude
    photons = photons0*10**(mag/(-2.5))
    
    #Multiply by collecting area 
    photons = photons*collecting_area*1e4 #Convert to cm^2
    
    #Multiply by filter width
    delta_lambda = filters[filter_name]['delta_lambda']
    delta_lambda *=1e4 #Convert to of Angstrom
    
    photons = photons*delta_lambda
    
    return photons 