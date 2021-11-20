#@title
 #helper functions 
import numpy as np
from hcipy import *
from scipy.ndimage import gaussian_filter


def bin(imin,fbin): 

    ''' Parameters
    ----------
    imin : 2D numpy array
         The 2D image that you want to bin
    fbin : int
         
    
    Returns
    -------
    out : 2D numpy array
        the 2D binned image
        '''
    out=np.zeros((int(imin.shape[0]/fbin),int(imin.shape[1]/fbin)))
   #  begin binning
    for i in np.arange(fbin-1,imin.shape[0]-fbin,fbin):
        for j in np.arange(fbin-1,imin.shape[1]-fbin,fbin):
            out[int((i+1)/fbin)-1,int((j+1)/fbin)-1]=np.sum(imin[i-int((fbin-1)/2):i+int((fbin-1)/2),j-int((fbin-1)/2):j+int((fbin-1)/2)])
    return out


def pyramid_slopes(image,pixels_pyramid_pupils, diameter):

    ''' Parameters
    ----------
    image : 1D numpy array
         The flatted image of the pyramid wfs pupils
           
    Returns
    -------
    slopes : 1D numpy array
        x- and y- slopes inside the pupil stacked onto of eachother for 1D array
        '''
    D = diameter
    pyramid_plot_grid = make_pupil_grid(pixels_pyramid_pupils*2, D)
    

#     pyr1=circular_aperture(0.5*D,[-0.25*D,0.25*D])(pyramid_plot_grid)
#     pyr2=circular_aperture(0.5*D,[0.25*D,0.25*D])(pyramid_plot_grid)
#     pyr3=circular_aperture(0.5*D,[-0.25*D,-0.25*D])(pyramid_plot_grid)
#     pyr4=circular_aperture(0.5*D,[0.25*D,-0.25*D])(pyramid_plot_grid) 

#     pyr1=hexagonal_aperture(0.5*D,center = [-0.25*D,0.25*D], angle = 0)(pyramid_plot_grid)
#     pyr2=hexagonal_aperture(0.5*D,center = [0.25*D,0.25*D], angle = 0)(pyramid_plot_grid)
#     pyr3=hexagonal_aperture(0.5*D,center = [-0.25*D,-0.25*D], angle = 0)(pyramid_plot_grid)
#     pyr4=hexagonal_aperture(0.5*D,center = [0.25*D,-0.25*D], angle = 0)(pyramid_plot_grid)
    
    pyr1=hexagonal_aperture(0.5*D,center = [-0.25*D,0.25*D], angle = 0)(pyramid_plot_grid)
    pyr2=hexagonal_aperture(0.5*D,center = [0.25*D,0.25*D], angle = 0)(pyramid_plot_grid)
    pyr3=hexagonal_aperture(0.5*D,center = [-0.25*D,-0.25*D], angle = 0)(pyramid_plot_grid)
    pyr4=hexagonal_aperture(0.5*D,center = [0.25*D,-0.25*D], angle = 0)(pyramid_plot_grid)
        
    N=4*np.sum(pyr1[pyr1>0])
    norm=(image[pyr1>0]+image[pyr2>0]+image[pyr3>0]+image[pyr4>0])/N
    sx=(image[pyr1>0]-image[pyr2>0]+image[pyr3>0]-image[pyr4>0])
    sy=(image[pyr1>0]+image[pyr2>0]-image[pyr3>0]-image[pyr4>0])
    return np.array([sx,sy]).flatten()

def plot_slopes(slopes,pixels_pyramid_pupils, diameter):
    ''' 
    Only want if we decide to plot the slopes. 

    Parameters
    ----------
    slopes : 1D numpy array
         The flatted slopes produced by pyramid_slopes(). 
           
    Returns
    -------
    slopes : 1D numpy array
        x- and y- slopes mapped within their pupils for easy plotting
    '''
    D=diameter
    mid=int(slopes.shape[0]/2)
    pyramid_plot_grid = make_pupil_grid(pixels_pyramid_pupils, D)
    pyr_mask=circular_aperture(D)(pyramid_plot_grid)
    sx=pyr_mask.copy()
    sy=pyr_mask.copy()
    sx[sx>0]=slopes[0:mid]
    sy[sy>0]=slopes[mid::]
    return [sx,sy]


def make_command_matrix(deformable_mirror, mpwfs,modsteps,wfs_camera,wf,pixels_pyramid_pupils, diameter, pyr_ref):

    probe_amp = 0.02 * wf.wavelength
    response_matrix = []
    num_modes=deformable_mirror.num_actuators

    for i in range(int(num_modes)):
        slope = 0

        for s in [1, -1]:
            amp = np.zeros((num_modes,))
            amp[i] = s * probe_amp
            deformable_mirror.flatten()
            deformable_mirror.actuators = amp

            dm_wf = deformable_mirror.forward(wf)
            wfs_wf = mpwfs.forward(dm_wf)

            for m in range (modsteps) :
                wfs_camera.integrate(wfs_wf[m], 1)

            image_nophot = wfs_camera.read_out()
            image_nophot/=image_nophot.sum()
            D = diameter  
            sxy=pyramid_slopes(image_nophot,pixels_pyramid_pupils, D)

            slope += s * (sxy-pyr_ref)/(2*probe_amp)#indent  #these are not really slopes; this is just a normalized differential image

        response_matrix.append(slope.ravel())#indet

    response_mtx= ModeBasis(response_matrix)
    rcond = 1e-3

    reconstruction_matrix = inverse_tikhonov(response_mtx.transformation_matrix, rcond=rcond)

    return reconstruction_matrix