#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:58:08 2021

@author: maaikevankooten
"""
from hcipy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def make_keck_aperture(normalized=False, with_spiders=False, with_segment_gaps=False, gap_padding=0, segment_transmissions=1, return_header=False, return_segments=False):
    """
    

    Parameters
    ----------
    normalized : TYPE, optional
        DESCRIPTION. The default is True.
    with_spiders : TYPE, optional
        DESCRIPTION. The default is False.
    with_segment_gaps : TYPE, optional
        DESCRIPTION. The default is False.
    gap_padding : TYPE, optional
        DESCRIPTION. The default is 0.
    segment_transmissions : TYPE, optional
        DESCRIPTION. The default is 1.
    return_header : TYPE, optional
        DESCRIPTION. The default is False.
    return_segments : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    pupil_diameter = 10.9 #m actual circumscribed diameter
    actual_segment_flat_diameter = np.sqrt(3)/2 * 1.8 #m actual segment flat-to-flat diameter
    # iris_ao_segment = np.sqrt(3)/2 * .7 mm (~.606 mm)
    actual_segment_gap = 0.003 #m actual gap size between segments
    # (3.5 - (3 D + 4 S)/6 = iris_ao segment gap (~7.4e-17)
    spider_width = 1*0.02450 #m actual strut size
    if normalized: 
        actual_segment_flat_diameter/=pupil_diameter
        actual_segment_gap/=pupil_diameter
        spider_width/=pupil_diameter
        pupil_diameter/=pupil_diameter
    gap_padding = 10.
    segment_gap = actual_segment_gap * gap_padding #padding out the segmentation gaps so they are visible and not sub-pixel
    segment_transmissions = 1.

    segment_flat_diameter = actual_segment_flat_diameter - (segment_gap - actual_segment_gap)
    segment_circum_diameter = 2 / np.sqrt(3) * segment_flat_diameter #segment circumscribed diameter

    num_rings = 3 #number of full rings of hexagons around central segment

    segment_positions = make_hexagonal_grid(actual_segment_flat_diameter + actual_segment_gap, num_rings)
    segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

    segment = hexagonal_aperture(segment_circum_diameter, np.pi / 2)

    spider1 = make_spider_infinite([0, 0], 0, spider_width)
    spider2 = make_spider_infinite([0, 0], 60, spider_width)
    spider3 = make_spider_infinite([0, 0], 120, spider_width)
    spider4 = make_spider_infinite([0, 0], 180, spider_width)
    spider5 = make_spider_infinite([0, 0], 240, spider_width)
    spider6 = make_spider_infinite([0, 0], 300, spider_width)

    segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments=True)

    segmentation, segments = segmented_aperture
    def segment_with_spider(segment):
        return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
    segments = [segment_with_spider(s) for s in segments]
    contour = make_segmented_aperture(segment, segment_positions)

    def func(grid):
        res = contour(grid) * spider1(grid) * spider2(grid) * spider3(grid)* spider4(grid) * spider3(grid)* spider5(grid) * spider6(grid) # * coro(grid)
        return Field(res, grid)
    if return_segments:
        return func, segments
    else:
        return func
    
def make_keck_aperture_shift(D, center = None, normalized=True, with_spiders=False, with_segment_gaps=False, gap_padding=0, segment_transmissions=1, return_header=False, return_segments=False):
    """

    This is the same as make_keck_aperture, except that you can scale the diameter, and you can shift the grid.

    Parameters
    ----------
    normalized : TYPE, optional
        DESCRIPTION. The default is True.
    with_spiders : TYPE, optional
        DESCRIPTION. The default is False.
    with_segment_gaps : TYPE, optional
        DESCRIPTION. The default is False.
    gap_padding : TYPE, optional
        DESCRIPTION. The default is 0.
    segment_transmissions : TYPE, optional
        DESCRIPTION. The default is 1.
    return_header : TYPE, optional
        DESCRIPTION. The default is False.
    return_segments : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if center is None:
        shift = np.zeros(2)
    else:
        shift = center * np.ones(2)
#    pupil_diameter = 10.9 #m actual circumscribed diameter
    pupil_diameter = D #m actual circumscribed diameter
    actual_segment_flat_diameter = np.sqrt(3)/2 * 1.8 * D/10.9#m actual segment flat-to-flat diameter
    # iris_ao_segment = np.sqrt(3)/2 * .7 mm (~.606 mm)
    actual_segment_gap = 0.003 * D/10.9 #m actual gap size between segments
    # (3.5 - (3 D + 4 S)/6 = iris_ao segment gap (~7.4e-17)
    spider_width = 1*0.02450 * D/10.9 #m actual strut size
    if normalized:
        actual_segment_flat_diameter/=pupil_diameter
        actual_segment_gap/=pupil_diameter
        spider_width/=pupil_diameter
        pupil_diameter/=pupil_diameter
    gap_padding = 10.
    segment_gap = actual_segment_gap * gap_padding #padding out the segmentation gaps so they are visible and not sub-pixel
    segment_transmissions = 1.

    segment_flat_diameter = actual_segment_flat_diameter - (segment_gap - actual_segment_gap)
    segment_circum_diameter = 2 / np.sqrt(3) * segment_flat_diameter #segment circumscribed diameter

    num_rings = 3 #number of full rings of hexagons around central segment

    segment_positions = make_hexagonal_grid(actual_segment_flat_diameter + actual_segment_gap, num_rings)
    segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

    segment = hexagonal_aperture(segment_circum_diameter, np.pi / 2)

    spider1 = make_spider_infinite([0, 0], 0, spider_width)
    spider2 = make_spider_infinite([0, 0], 60, spider_width)
    spider3 = make_spider_infinite([0, 0], 120, spider_width)
    spider4 = make_spider_infinite([0, 0], 180, spider_width)
    spider5 = make_spider_infinite([0, 0], 240, spider_width)
    spider6 = make_spider_infinite([0, 0], 300, spider_width)

    segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments=True)

    segmentation, segments = segmented_aperture
    def segment_with_spider(segment):
        return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
    segments = [segment_with_spider(s) for s in segments]
    contour = make_segmented_aperture(segment, segment_positions)

    def func(grid):
        grid_shift = grid.shifted(-shift)
        res = contour(grid_shift) * spider1(grid_shift) * spider2(grid_shift) * spider3(grid_shift)* spider4(grid_shift) * spider3(grid_shift)* spider5(grid_shift) * spider6(grid_shift) # * coro(grid)
        return Field(res, grid)
    if return_segments:
        return func, segments
    else:
        return func

def make_keck_aperture_half(**kwargs):
    return make_keck_aperture_shift(1.003*10.9/2., normalized=False, **kwargs)

def make_keck_aperture_half_topleft(grid):
    return make_keck_aperture_half(center=[-0.25*10.9,0.25*10.9])(grid)

def make_keck_aperture_half_topright(grid):
    return make_keck_aperture_half(center=[0.25*10.9,0.25*10.9])(grid)

def make_keck_aperture_half_bottomleft(grid):
    return make_keck_aperture_half(center=[-0.25*10.9,-0.25*10.9])(grid)

def make_keck_aperture_half_bottomright(grid):
    return make_keck_aperture_half(center=[0.25*10.9,-0.25*10.9])(grid)

# grid_size=512
# telescope_diameter=10.9
# pupil_grid = make_pupil_grid(grid_size, telescope_diameter)

# #not all modes are supported yet and the exact values for spider, segment gaps etc are being confirmed with Sam Ragland from Keck. 
# #this command returns the segments (if you want to make a segmented DM to move the primary mirror) and doesnt normalize the spatial units by D. 
keck_aperture, segments = make_keck_aperture(return_segments=True,normalized=False)
# telescope_pupil = evaluate_supersampled(keck_aperture, pupil_grid, 2)
# segments = evaluate_supersampled(segments, pupil_grid, 2)
# plt.figure()
# imshow_field(telescope_pupil)
