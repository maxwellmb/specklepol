import numpy as np
import sys
from hcipy import *

def emgain_large_poisson(lam, thresh=1e6):
    """
    Draw samples from a Poisson distribution while taking into account EM multiplicative noise and taking care of large values of `lam`.

    At large values of `lam` the distribution automatically switches to the corresponding normal distribution.
    This switch is independently decided for each expectation value in the `lam` array.

    Parameters
    ----------
    lam : array_like
        Expectation value for the Poisson distribution. Must be >= 0.
    thresh : float
        The threshold at which the distribution switched from a Poisson to a normal distribution.

    Returns
    -------
    array_like
        The drawn samples from the Poisson or normal distribution, depending on the expectation value.
    """
    large = lam > thresh
    small = ~large

    # Use normal approximation if the number of photons is large
    n = np.zeros(lam.shape)
    n[large] = np.round(lam[large] + np.random.normal(size=np.sum(large)) * np.sqrt(2*lam[large]))
    n[small] = np.random.poisson(2*lam[small], size=np.sum(small))-lam[small]

    if hasattr(lam, 'grid'):
        n = Field(n, lam.grid)

    return n

class ProEM512(NoisyDetector):
    '''A subclass of NoisyDetector class based on the ProEM®-HS:512BX3.\n

    Details can be found at: \n
    https://www.princetoninstruments.com/wp-content/uploads/2020/10/ProEM-HS_512BX3_datasheet.pdf \n

    The parameters of this detector have been hardcoded into the subclass based on values drawn from 
    brochures and user manuals. These parameters include:\n
    dark current rate - found in documentation \n
    read noise - found in documentation (Note that when a range of read noises was given in the documentation,
        the MAXIMUM read noise was always choosen. Read Noise generally increases as a function of frame rate
        and given the speckle applications of this project, we presumed that the maximum frame rate was desired.
        Presuming the maximum frame rate lead to the choice of the maximum read noise.)
    flat field - presumed to be zero \n
    include photon noise - presumed to be true \n
    max fps - The maximum FPS at the size most relevant to the VIPER Project. \n
    detector size - The size of the short length of the detector. Many detectors can have their regions of 
        interest reduced in order to increase frame rate. Some detectors reduce ROI into smaller sqaures 
        while others reduce ROI into rows with max length but reduced height. In the case of square ROIs
        the detector size corresponds to the side length of the square. In the case of rectangular ROIs
        the detector size corresponds the the length of the short side, as the long side remains unchanged. \n
    shutter type - The electronic shutter that determines the way in which photons are captured and then read out
        of the detector. Can be either Rolling or Global. \n
    detector type - The classification of the detector. Either EMCCD or some form of CMOS camera. \n
    quantum efficiency - The quantum efficiency of the detector calculated based on taking a weighted average
        of the QE curves given in documentation with the weights being the transmission ratios of the filters
        U, B, V, R, and I. \n

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector samples.
    filter : string
        A letter indicating the filter from UBVRI. Relevant for computing the quantum efficiency.
    EM_gain: Scalar or None
        The EM multiplication gain of the EMCCD. Used to determine the read noise and if the full well depth is exceeded.
        None means EM Multiplicative noise will not be considered.
    EM_saturate: None or np.nan
        Choose the behavior of the detector if the full well depth is exceeded. If None, every instance that the full
        well depth is exceeded is replaced with the maximum full well depth. If np.nan, it is replaced with np.nan.
    subsampling : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''
    def __init__(self, detector_grid, filter, EM_gain = None, EM_saturate = None , subsampling=1):
        NoisyDetector.__init__(self, detector_grid, subsampling)

        # Setting the start charge level.
        self.accumulated_charge = 0

        # The parameters.
        self.dark_current_rate = 0.001
        self.read_noise = 125
        self.flat_field = 0
        self.include_photon_noise = True
        self.max_fps = 61
        self.detector_size = 512
        self.shutter_type = 'Global'
        self.detector_type = "EMCCD"
        self.EM_gain = EM_gain
        self.EM_saturate = EM_saturate
        self.full_well_depth = 200000

        # Set Quantum Efficiency based on filter
        if filter == 'U' or filter == 'u':
            self.QE = 0.494
        elif filter == 'B' or filter == 'b':
            self.QE = 0.795
        elif filter == 'V' or filter == 'v':
            self.QE = 0.902
        elif filter == 'R' or filter == 'r':
            self.QE = 0.894
        elif filter == 'I' or filter == 'i':
            self.QE = 0.499
        else:
            raise ValueError("Error, invalid filter name.")

        # Determine the read noise based on the EM Gain
        if EM_gain is not None:
            self.read_noise = self.read_noise/EM_gain

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Identical to the integrate funcion of NoisyDetector except includes loss due to Quantum Efficiency.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        
        # The power that the detector detects during the integration.
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

        # Adding the generated dark current.
        self.accumulated_charge += self.dark_current_rate * dt * weight

    def read_out(self):
        '''Reads out the detector.

        Identical to the read_out function of NoisyDetector except EM multiplicative noise and EM gain has been included.

        Returns
        ----------
        Field
            The final detector image.
        '''
        # Make sure not to overwrite output
        output_field = self.accumulated_charge.copy()

        # Adding photon noise.
        if self.EM_gain is not None: 
            if self.include_photon_noise:
                output_field = emgain_large_poisson(output_field, thresh=1e6)
        else:
            if self.include_photon_noise:
                output_field = large_poisson(output_field, thresh=1e6)

        # Adding flat field errors.
        output_field *= self.flat_field

        # Adding read-out noise.
        output_field += np.random.normal(loc=0, scale=self.read_noise, size=output_field.size)

        # If EM Gain is active, replace saturated pixels with either full well depth or nan. 
        if self.EM_gain is not None:
            
            # Notify user that full well depth has been exceeded
            for i in range(output_field.shape[0]):
               if output_field[i]>self.full_well_depth/self.EM_gain:
                    print("Warning: Full Well Depth Exceeded")
                    break

            # Set Saturated pixels to nan
            if self.EM_saturate is np.nan:
                output_field[output_field>self.full_well_depth/self.EM_gain] = np.nan
           
            # Set Saturated pixels to full well depth
            if self.EM_saturate is None:
                output_field[output_field>self.full_well_depth/self.EM_gain] = self.full_well_depth/self.EM_gain
        
        # Multiply Signal by EM Gain
        output_field = self.EM_gain*output_field
        # Reset detector
        self.accumulated_charge = 0

        return output_field   

class iXon897(NoisyDetector):
    '''A subclass of NoisyDetector class based on the iXon Ultra 897.\n

    Details can be found at: \n
    https://andor.oxinst.com/assets/uploads/products/andor/documents/andor-ixon-ultra-emccd-specifications.pdf \n

    The parameters of this detector have been hardcoded into the subclass based on values drawn from 
    brochures and user manuals. These parameters include:\n
    dark current rate - found in documentation \n
    read noise - found in documentation (Note that when a range of read noises was given in the documentation,
        the MAXIMUM read noise was always choosen. Read Noise generally increases as a function of frame rate
        and given the speckle applications of this project, we presumed that the maximum frame rate was desired.
        Presuming the maximum frame rate lead to the choice of the maximum read noise.)
    flat field - presumed to be zero \n
    include photon noise - presumed to be true \n
    max fps - The maximum FPS at the size most relevant to the VIPER Project. \n
    detector size - The size of the short length of the detector. Many detectors can have their regions of 
        interest reduced in order to increase frame rate. Some detectors reduce ROI into smaller sqaures 
        while others reduce ROI into rows with max length but reduced height. In the case of square ROIs
        the detector size corresponds to the side length of the square. In the case of rectangular ROIs
        the detector size corresponds the the length of the short side, as the long side remains unchanged. \n
    shutter type - The electronic shutter that determines the way in which photons are captured and then read out
        of the detector. Can be either Rolling or Global. \n
    detector type - The classification of the detector. Either EMCCD or some form of CMOS camera. \n
    quantum efficiency - The quantum efficiency of the detector calculated based on taking a weighted average
        of the QE curves given in documentation with the weights being the transmission ratios of the filters
        U, B, V, R, and I. \n

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector samples.
    filter : string
        A letter indicating the filter from UBVRI. Relevant for computing the quantum efficiency.
    EM_gain: Scalar or None
        The EM multiplication gain of the EMCCD. Used to determine the read noise and if the full well depth is exceeded.
        None means EM Multiplicative noise will not be considered.
    EM_saturate: None or np.nan
        Choose the behavior of the detector if the full well depth is exceeded. If None, every instance that the full
        well depth is exceeded is replaced with the maximum full well depth. If np.nan, it is replaced with np.nan.
    subsampling : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''
    def __init__(self, detector_grid, filter, EM_gain = None, EM_saturate= None , subsampling=1):
        NoisyDetector.__init__(self, detector_grid, subsampling)

        # Setting the start charge level.
        self.accumulated_charge = 0

        # The parameters.
        self.dark_current_rate = 0.003
        self.read_noise = 89
        self.flat_field = 0
        self.include_photon_noise = True
        self.max_fps = 56
        self.detector_size = 512
        self.shutter_type = 'Global'
        self.detector_type = "EMCCD"
        self.EM_gain = EM_gain
        self.EM_saturate = EM_saturate
        self.full_well_depth = 180000

        # Set Quantum Efficiency based on filter
        if filter == 'U' or filter == 'u':
            self.QE = 0.258
        elif filter == 'B' or filter == 'b':
            self.QE = 0.735
        elif filter == 'V' or filter == 'v':
            self.QE = 0.957
        elif filter == 'R' or filter == 'r':
            self.QE = 0.869
        elif filter == 'I' or filter == 'i':
            self.QE = 0.479
        else:
            raise ValueError("Error, invalid filter name.")

        # Determine the read noise based on the EM Gain
        if EM_gain is not None:
            self.read_noise = self.read_noise/EM_gain

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Identical to the integrate funcion of NoisyDetector except includes loss due to Quantum Efficiency.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        
        # The power that the detector detects during the integration.
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

        # Adding the generated dark current.
        self.accumulated_charge += self.dark_current_rate * dt * weight

    def read_out(self):
        '''Reads out the detector.

        Identical to the read_out function of NoisyDetector except EM multiplicative noise and EM gain has been included.

        Returns
        ----------
        Field
            The final detector image.
        '''
        # Make sure not to overwrite output
        output_field = self.accumulated_charge.copy()

        # Adding photon noise.
        if self.EM_gain is not None: 
            if self.include_photon_noise:
                output_field = emgain_large_poisson(output_field, thresh=1e6)
        else:
            if self.include_photon_noise:
                output_field = large_poisson(output_field, thresh=1e6)

        # Adding flat field errors.
        output_field *= self.flat_field

        # Adding read-out noise.
        output_field += np.random.normal(loc=0, scale=self.read_noise, size=output_field.size)

        # If EM Gain is active, replace saturated pixels with either full well depth or nan. 
        if self.EM_gain is not None:
            
            # Notify user that full well depth has been exceeded
            for i in range(output_field.shape[0]):
                if output_field[i]>self.full_well_depth/self.EM_gain:
                    print("Warning: Full Well Depth Exceeded")
                    break

            # Set Saturated pixels to nan
            if self.EM_saturate is np.nan:
                output_field[output_field>self.full_well_depth/self.EM_gain] = np.nan
           
            # Set Saturated pixels to full well depth
            if self.EM_saturate is None:
                output_field[output_field>self.full_well_depth/self.EM_gain] = self.full_well_depth/self.EM_gain

        # Multiply Signal by EM Gain
        output_field = self.EM_gain*output_field
        
        # Reset detector
        self.accumulated_charge = 0

        return output_field   

class ORCA_Quest(NoisyDetector):
    '''A subclass of NoisyDetector class based on the ORCA Quest.\n

    Details can be found at: \n
    https://www.hamamatsu.com/resources/pdf/sys/SCAS0154E_C15550-20UP_tec.pdf \n

    The parameters of this detector have been hardcoded into the subclass based on values drawn from 
    brochures and user manuals. These parameters include:\n
    dark current rate - found in documentation \n
    read noise - found in documentation (Note that when a range of read noises was given in the documentation,
        the MAXIMUM read noise was always choosen. Read Noise generally increases as a function of frame rate
        and given the speckle applications of this project, we presumed that the maximum frame rate was desired.
        Presuming the maximum frame rate lead to the choice of the maximum read noise.)
    flat field - presumed to be zero \n
    include photon noise - presumed to be true \n
    max fps - The maximum FPS at the size most relevant to the VIPER Project. \n
    detector size - The size of the short length of the detector. Many detectors can have their regions of 
        interest reduced in order to increase frame rate. Some detectors reduce ROI into smaller sqaures 
        while others reduce ROI into rows with max length but reduced height. In the case of square ROIs
        the detector size corresponds to the side length of the square. In the case of rectangular ROIs
        the detector size corresponds the the length of the short side, as the long side remains unchanged. \n
    shutter type - The electronic shutter that determines the way in which photons are captured and then read out
        of the detector. Can be either Rolling or Global. \n
    detector type - The classification of the detector. Either EMCCD or some form of CMOS camera. \n
    quantum efficiency - The quantum efficiency of the detector calculated based on taking a weighted average
        of the QE curves given in documentation with the weights being the transmission ratios of the filters
        U, B, V, R, and I. \n

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector samples.
    filter : string
        A letter indicating the filter from UBVRI. Relevant for computing the quantum efficiency.
    subsampling : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''	
    def __init__(self, detector_grid, filter, subsampling=1):
        NoisyDetector.__init__(self, detector_grid, subsampling)

        # Setting the start charge level.
        self.accumulated_charge = 0

                # The parameters.
        self.dark_current_rate = 0.006
        self.read_noise = 0.43
        self.flat_field = 0
        self.include_photon_noise = True
        self.max_fps = 532
        self.detector_size = 512
        self.shutter_type = 'Rolling'
        self.detector_type = "sCMOS"
        self.number_of_subdivisions = 32
                
        # Set Quantum Efficiency based on filter
        if filter == 'U' or filter == 'u':
            self.QE = 0.459
        elif filter == 'B' or filter == 'b':
            self.QE = 0.865
        elif filter == 'V' or filter == 'v':
            self.QE = 0.835
        elif filter == 'R' or filter == 'r':
            self.QE = 0.648
        elif filter == 'I' or filter == 'i':
            self.QE = 0.362
        else:
            raise ValueError("Error, invalid filter name.")

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Identical to the integrate funcion of NoisyDetector except includes loss due to Quantum Efficiency.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        
        # The power that the detector detects during the integration.
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

        # Adding the generated dark current.
        self.accumulated_charge += self.dark_current_rate * dt * weight

    def roll_shutter(self, wavefronts, layer, prop, exposure_time):
        '''Simulates a rolling shutter.

        A combination of integrate and read_out that simulates the effects of a rolling shutter.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        layer: Atmospheric layer
            The atmospheric layer 
        prop: Propagator
            The propagator for the shutter
        exposure_time: Scalar
            The total exposure time in seconds

        Returns
        ----------
        Field.shaped
            The final shaped detector image.
        '''
        number_of_rows = int(np.sqrt(self.detector_grid.size))
        row_readout_time = 1/(self.max_fps*number_of_rows)
        row_differential_time = row_readout_time*number_of_rows/self.number_of_subdivisions
        layer.t += exposure_time-(row_differential_time*self.number_of_subdivisions)
        for k in wavefronts:
            self.integrate(prop((layer(k))),exposure_time-(row_differential_time*self.number_of_subdivisions)) 
        read_noise_temp = self.read_noise
        self.read_noise = 0 # Prevent double counting of the read noise when reading out the detector twice. 
        image_comb = self.read_out()
        for j in range(self.number_of_subdivisions):
            layer.t += row_differential_time
            for k in wavefronts:
                self.integrate(prop((layer(k))),row_differential_time)       
            self.read_noise = read_noise_temp
            image_row = self.read_out()
            start = int(self.detector_grid.size*(j)/self.number_of_subdivisions)
            end = int(self.detector_grid.size*((j+1)/self.number_of_subdivisions))
            image_comb[start:end]+=image_row[start:end]        
        return image_comb.shaped

class Marana(NoisyDetector):
    '''A subclass of NoisyDetector class based on the Andor Marana.\n

    Details can be found at: \n
    https://andor.oxinst.com/assets/uploads/products/andor/documents/andor-marana-scmos-specifications.pdf \n

    The parameters of this detector have been hardcoded into the subclass based on values drawn from 
    brochures and user manuals. These parameters include:\n
    dark current rate - found in documentation \n
    read noise - found in documentation (Note that when a range of read noises was given in the documentation,
        the MAXIMUM read noise was always choosen. Read Noise generally increases as a function of frame rate
        and given the speckle applications of this project, we presumed that the maximum frame rate was desired.
        Presuming the maximum frame rate lead to the choice of the maximum read noise.)
    flat field - presumed to be zero \n
    include photon noise - presumed to be true \n
    max fps - The maximum FPS at the size most relevant to the VIPER Project. \n
    detector size - The size of the short length of the detector. Many detectors can have their regions of 
        interest reduced in order to increase frame rate. Some detectors reduce ROI into smaller sqaures 
        while others reduce ROI into rows with max length but reduced height. In the case of square ROIs
        the detector size corresponds to the side length of the square. In the case of rectangular ROIs
        the detector size corresponds the the length of the short side, as the long side remains unchanged. \n
    shutter type - The electronic shutter that determines the way in which photons are captured and then read out
        of the detector. Can be either Rolling or Global. \n
    detector type - The classification of the detector. Either EMCCD or some form of CMOS camera. \n
    quantum efficiency - The quantum efficiency of the detector calculated based on taking a weighted average
        of the QE curves given in documentation with the weights being the transmission ratios of the filters
        U, B, V, R, and I. \n

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector samples.
    filter : string
        A letter indicating the filter from UBVRI. Relevant for computing the quantum efficiency.
    subsampling : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''	
    def __init__(self, detector_grid, filter, subsampling=1):
        NoisyDetector.__init__(self, detector_grid, subsampling)

        # Setting the start charge level.
        self.accumulated_charge = 0

                # The parameters.
        self.dark_current_rate = 0.7
        self.read_noise = 1.6
        self.flat_field = 0
        self.include_photon_noise = True
        self.max_fps = 24
        self.detector_size = 2048
        self.shutter_type = 'Rolling'
        self.detector_type = "sCMOS"
        self.number_of_subdivisions = 32
                
        # Set Quantum Efficiency based on filter
        if filter == 'U' or filter == 'u':
            self.QE = 0.439
        elif filter == 'B' or filter == 'b':
            self.QE = 0.782
        elif filter == 'V' or filter == 'v':
            self.QE = 0.941
        elif filter == 'R' or filter == 'r':
            self.QE = 0.813
        elif filter == 'I' or filter == 'i':
            self.QE = 0.424
        else:
            raise ValueError("Error, invalid filter name.")

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Identical to the integrate funcion of NoisyDetector except includes loss due to Quantum Efficiency.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        
        # The power that the detector detects during the integration.
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

        # Adding the generated dark current.
        self.accumulated_charge += self.dark_current_rate * dt * weight

    def roll_shutter(self, wavefronts, layer, prop, exposure_time):
        '''Simulates a rolling shutter.

        A combination of integrate and read_out that simulates the effects of a rolling shutter.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        layer: Atmospheric layer
            The atmospheric layer 
        prop: Propagator
            The propagator for the shutter
        exposure_time: Scalar
            The total exposure time in seconds

        Returns
        ----------
        Field.shaped
            The final shaped detector image.
        '''
        number_of_rows = int(np.sqrt(self.detector_grid.size))
        row_readout_time = 1/(self.max_fps*number_of_rows)
        row_differential_time = row_readout_time*number_of_rows/self.number_of_subdivisions
        layer.t += exposure_time-(row_differential_time*self.number_of_subdivisions)
        for k in wavefronts:
            self.integrate(prop((layer(k))),exposure_time-(row_differential_time*self.number_of_subdivisions)) 
        read_noise_temp = self.read_noise
        self.read_noise = 0 # Prevent double counting of the read noise when reading out the detector twice. 
        image_comb = self.read_out()
        for j in range(self.number_of_subdivisions):
            layer.t += row_differential_time
            for k in wavefronts:
                self.integrate(prop((layer(k))),row_differential_time)       
            self.read_noise = read_noise_temp
            image_row = self.read_out()
            start = int(self.detector_grid.size*(j)/self.number_of_subdivisions)
            end = int(self.detector_grid.size*((j+1)/self.number_of_subdivisions))
            image_comb[start:end]+=image_row[start:end]        
        return image_comb.shaped


class Kinetix(NoisyDetector):
    '''A subclass of NoisyDetector class based on the Kinetix.\n

    Details can be found at: \n
    https://www.photometrics.com/wp-content/uploads/2019/10/Kinetix-Datasheet-Rev-A2-060082021.pdf \n

    The parameters of this detector have been hardcoded into the subclass based on values drawn from 
    brochures and user manuals. These parameters include:\n
    dark current rate - found in documentation \n
    read noise - found in documentation (Note that when a range of read noises was given in the documentation,
        the MAXIMUM read noise was always choosen. Read Noise generally increases as a function of frame rate
        and given the speckle applications of this project, we presumed that the maximum frame rate was desired.
        Presuming the maximum frame rate lead to the choice of the maximum read noise.)
    flat field - presumed to be zero \n
    include photon noise - presumed to be true \n
    max fps - The maximum FPS at the size most relevant to the VIPER Project. \n
    detector size - The size of the short length of the detector. Many detectors can have their regions of 
        interest reduced in order to increase frame rate. Some detectors reduce ROI into smaller sqaures 
        while others reduce ROI into rows with max length but reduced height. In the case of square ROIs
        the detector size corresponds to the side length of the square. In the case of rectangular ROIs
        the detector size corresponds the the length of the short side, as the long side remains unchanged. \n
    shutter type - The electronic shutter that determines the way in which photons are captured and then read out
        of the detector. Can be either Rolling or Global. \n
    detector type - The classification of the detector. Either EMCCD or some form of CMOS camera. \n
    quantum efficiency - The quantum efficiency of the detector calculated based on taking a weighted average
        of the QE curves given in documentation with the weights being the transmission ratios of the filters
        U, B, V, R, and I. \n

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector samples.
    filter : string
        A letter indicating the filter from UBVRI. Relevant for computing the quantum efficiency.
    subsampling : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''	
    def __init__(self, detector_grid, filter, subsampling=1):
        NoisyDetector.__init__(self, detector_grid, subsampling)

        # Setting the start charge level.
        self.accumulated_charge = 0

                # The parameters.
        self.dark_current_rate = 1.27
        self.read_noise = 1.2
        self.flat_field = 0
        self.include_photon_noise = True
        self.max_fps = 166
        self.detector_size = 1500
        self.shutter_type = 'Rolling'
        self.detector_type = "sCMOS"
        self.number_of_subdivisions = 32
                
        # Set Quantum Efficiency based on filter
        if filter == 'U' or filter == 'u':
            self.QE = 0.427
        elif filter == 'B' or filter == 'b':
            self.QE = 0.797
        elif filter == 'V' or filter == 'v':
            self.QE = 0.946
        elif filter == 'R' or filter == 'r':
            self.QE = 0.859
        elif filter == 'I' or filter == 'i':
            self.QE = 0.503
        else:
            raise ValueError("Error, invalid filter name.")

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Identical to the integrate funcion of NoisyDetector except includes loss due to Quantum Efficiency.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        
        # The power that the detector detects during the integration.
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

        # Adding the generated dark current.
        self.accumulated_charge += self.dark_current_rate * dt * weight

    def roll_shutter(self, wavefronts, layer, prop, exposure_time):
        '''Simulates a rolling shutter.

        A combination of integrate and read_out that simulates the effects of a rolling shutter.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        layer: Atmospheric layer
            The atmospheric layer 
        prop: Propagator
            The propagator for the shutter
        exposure_time: Scalar
            The total exposure time in seconds

        Returns
        ----------
        Field.shaped
            The final shaped detector image.
        '''
        number_of_rows = int(np.sqrt(self.detector_grid.size))
        row_readout_time = 1/(self.max_fps*number_of_rows)
        row_differential_time = row_readout_time*number_of_rows/self.number_of_subdivisions
        layer.t += exposure_time-(row_differential_time*self.number_of_subdivisions)
        for k in wavefronts:
            self.integrate(prop((layer(k))),exposure_time-(row_differential_time*self.number_of_subdivisions)) 
        read_noise_temp = self.read_noise
        self.read_noise = 0 # Prevent double counting of the read noise when reading out the detector twice. 
        image_comb = self.read_out()
        for j in range(self.number_of_subdivisions):
            layer.t += row_differential_time
            for k in wavefronts:
                self.integrate(prop((layer(k))),row_differential_time)       
            self.read_noise = read_noise_temp
            image_row = self.read_out()
            start = int(self.detector_grid.size*(j)/self.number_of_subdivisions)
            end = int(self.detector_grid.size*((j+1)/self.number_of_subdivisions))
            image_comb[start:end]+=image_row[start:end]        
        return image_comb.shaped

class Prime_BSI(NoisyDetector):
    '''A subclass of NoisyDetector class based on the Prime BSI.\n

    Details can be found at: \n
    https://www.photometrics.com/wp-content/uploads/2019/10/PrimeBSI-Datasheet_Rev_A4_-07312020.pdf \n

    The parameters of this detector have been hardcoded into the subclass based on values drawn from 
    brochures and user manuals. These parameters include:\n
    dark current rate - found in documentation \n
    read noise - found in documentation (Note that when a range of read noises was given in the documentation,
        the MAXIMUM read noise was always choosen. Read Noise generally increases as a function of frame rate
        and given the speckle applications of this project, we presumed that the maximum frame rate was desired.
        Presuming the maximum frame rate lead to the choice of the maximum read noise.)
    flat field - presumed to be zero \n
    include photon noise - presumed to be true \n
    max fps - The maximum FPS at the size most relevant to the VIPER Project. \n
    detector size - The size of the short length of the detector. Many detectors can have their regions of 
        interest reduced in order to increase frame rate. Some detectors reduce ROI into smaller sqaures 
        while others reduce ROI into rows with max length but reduced height. In the case of square ROIs
        the detector size corresponds to the side length of the square. In the case of rectangular ROIs
        the detector size corresponds the the length of the short side, as the long side remains unchanged. \n
    shutter type - The electronic shutter that determines the way in which photons are captured and then read out
        of the detector. Can be either Rolling or Global. \n
    detector type - The classification of the detector. Either EMCCD or some form of CMOS camera. \n
    quantum efficiency - The quantum efficiency of the detector calculated based on taking a weighted average
        of the QE curves given in documentation with the weights being the transmission ratios of the filters
        U, B, V, R, and I. \n

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector samples.
    filter : string
        A letter indicating the filter from UBVRI. Relevant for computing the quantum efficiency.
    subsampling : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''	
    def __init__(self, detector_grid, filter, subsampling=1):
        NoisyDetector.__init__(self, detector_grid, subsampling)

        # Setting the start charge level.
        self.accumulated_charge = 0

                # The parameters.
        self.dark_current_rate = 0.12
        self.read_noise = 1.1
        self.flat_field = 0
        self.include_photon_noise = True
        self.max_fps = 173
        self.detector_size = 512
        self.shutter_type = 'Rolling'
        self.detector_type = "sCMOS"
        self.number_of_subdivisions = 32
                
        # Set Quantum Efficiency based on filter
        if filter == 'U' or filter == 'u':
            self.QE = 0.436
        elif filter == 'B' or filter == 'b':
            self.QE = 0.781
        elif filter == 'V' or filter == 'v':
            self.QE = 0.936
        elif filter == 'R' or filter == 'r':
            self.QE = 0.814
        elif filter == 'I' or filter == 'i':
            self.QE = 0.437
        else:
            raise ValueError("Error, invalid filter name.")

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Identical to the integrate funcion of NoisyDetector except includes loss due to Quantum Efficiency.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        
        # The power that the detector detects during the integration.
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

        # Adding the generated dark current.
        self.accumulated_charge += self.dark_current_rate * dt * weight

    def roll_shutter(self, wavefronts, layer, prop, exposure_time):
        '''Simulates a rolling shutter.

        A combination of integrate and read_out that simulates the effects of a rolling shutter.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        layer: Atmospheric layer
            The atmospheric layer 
        prop: Propagator
            The propagator for the shutter
        exposure_time: Scalar
            The total exposure time in seconds

        Returns
        ----------
        Field.shaped
            The final shaped detector image.
        '''
        number_of_rows = int(np.sqrt(self.detector_grid.size))
        row_readout_time = 1/(self.max_fps*number_of_rows)
        row_differential_time = row_readout_time*number_of_rows/self.number_of_subdivisions
        layer.t += exposure_time-(row_differential_time*self.number_of_subdivisions)
        for k in wavefronts:
            self.integrate(prop((layer(k))),exposure_time-(row_differential_time*self.number_of_subdivisions)) 
        read_noise_temp = self.read_noise
        self.read_noise = 0 # Prevent double counting of the read noise when reading out the detector twice. 
        image_comb = self.read_out()
        for j in range(self.number_of_subdivisions):
            layer.t += row_differential_time
            for k in wavefronts:
                self.integrate(prop((layer(k))),row_differential_time)       
            self.read_noise = read_noise_temp
            image_row = self.read_out()
            start = int(self.detector_grid.size*(j)/self.number_of_subdivisions)
            end = int(self.detector_grid.size*((j+1)/self.number_of_subdivisions))
            image_comb[start:end]+=image_row[start:end]        
        return image_comb.shaped

class Test_Global_50FPS(NoisyDetector):
    '''A subclass of NoisyDetector. Used to test the rolling vs global shutter question.\n

    Parameters based off the ProEM®-HS:512BX3\n

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector samples.
    filter : string
        A letter indicating the filter from UBVRI. Relevant for computing the quantum efficiency.
    subsampling : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''
    def __init__(self, detector_grid, filter, EM_gain = 1, EM_saturate = None , subsampling=1):
        NoisyDetector.__init__(self, detector_grid, subsampling)

        # Setting the start charge level.
        self.accumulated_charge = 0

        # The parameters.
        self.dark_current_rate = 0.001
        self.read_noise = 0.125
        self.flat_field = 0
        self.include_photon_noise = True
        self.max_fps = 50
        self.detector_size = 512
        self.shutter_type = 'Global'
        self.detector_type = "Test"

        # Set Quantum Efficiency based on filter
        if filter == 'U' or filter == 'u':
            self.QE = 0.494
        elif filter == 'B' or filter == 'b':
            self.QE = 0.795
        elif filter == 'V' or filter == 'v':
            self.QE = 0.902
        elif filter == 'R' or filter == 'r':
            self.QE = 0.894
        elif filter == 'I' or filter == 'i':
            self.QE = 0.499
        else:
            raise ValueError("Error, invalid filter name.")

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Identical to the integrate funcion of NoisyDetector except loss due to Quantum Efficiency is included.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        
        # The power that the detector detects during the integration.
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

        # Adding the generated dark current.
        self.accumulated_charge += self.dark_current_rate * dt * weight

class Test_Rolling_50FPS(NoisyDetector):
    '''A subclass of NoisyDetector. Used to test the rolling vs global shutter question.\n

    Parameters based off the ProEM®-HS:512BX3\n

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector samples.
    filter : string
        A letter indicating the filter from UBVRI. Relevant for computing the quantum efficiency.
    subsampling : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''
    def __init__(self, detector_grid, filter, EM_gain = 1, EM_saturate = None , subsampling=1):
        NoisyDetector.__init__(self, detector_grid, subsampling)

        # Setting the start charge level.
        self.accumulated_charge = 0

        # The parameters.
        self.dark_current_rate = 0.001
        self.read_noise = 0.125
        self.flat_field = 0
        self.include_photon_noise = True
        self.max_fps = 50
        self.detector_size = 512
        self.shutter_type = 'Rolling'
        self.detector_type = "Test"
        self.number_of_subdivisions = 32

        # Set Quantum Efficiency based on filter
        if filter == 'U' or filter == 'u':
            self.QE = 0.494
        elif filter == 'B' or filter == 'b':
            self.QE = 0.795
        elif filter == 'V' or filter == 'v':
            self.QE = 0.902
        elif filter == 'R' or filter == 'r':
            self.QE = 0.894
        elif filter == 'I' or filter == 'i':
            self.QE = 0.499
        else:
            raise ValueError("Error, invalid filter name.")

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Identical to the integrate funcion of NoisyDetector except loss due to Quantum Efficiency is included.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        
        # The power that the detector detects during the integration.
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

        # Adding the generated dark current.
        self.accumulated_charge += self.dark_current_rate * dt * weight
    
    def roll_shutter(self, wavefronts, layer, prop, exposure_time):
        '''Simulates a rolling shutter.

        A combination of integrate and read_out that simulates the effects of a rolling shutter.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        layer: Atmospheric layer
            The atmospheric layer 
        prop: Propagator
            The propagator for the shutter
        exposure_time: Scalar
            The total exposure time in seconds

        Returns
        ----------
        Field.shaped
            The final detector image shaped.
        '''
        number_of_rows = int(np.sqrt(self.detector_grid.size))
        row_readout_time = 1/(self.max_fps*number_of_rows)
        row_differential_time = row_readout_time*number_of_rows/self.number_of_subdivisions
        layer.t += exposure_time-(row_differential_time*self.number_of_subdivisions)
        for k in wavefronts:
            self.integrate(prop((layer(k))),exposure_time-(row_differential_time*self.number_of_subdivisions)) 
        read_noise_temp = self.read_noise
        self.read_noise = 0 # Prevent double counting of the read noise when reading out the detector twice. 
        image_comb = self.read_out()
        for j in range(self.number_of_subdivisions):
            layer.t += row_differential_time
            for k in wavefronts:
                self.integrate(prop((layer(k))),row_differential_time)       
            self.read_noise = read_noise_temp
            image_row = self.read_out()
            start = int(self.detector_grid.size*(j)/self.number_of_subdivisions)
            end = int(self.detector_grid.size*((j+1)/self.number_of_subdivisions))
            image_comb[start:end]+=image_row[start:end]        
        return image_comb.shaped

class Test_Rolling_200FPS(NoisyDetector):
    '''A subclass of NoisyDetector. Used to test the rolling vs global shutter question.\n

    Parameters based off the ProEM®-HS:512BX3\n

    Parameters
    ----------
    detector_grid : Grid
        The grid on which the detector samples.
    filter : string
        A letter indicating the filter from UBVRI. Relevant for computing the quantum efficiency.
    subsampling : integer or scalar or ndarray
        The number of subpixels per pixel along one axis. For example, a
        value of 2 indicates that 2x2=4 subpixels are used per pixel. If
        this is a scalar, it will be rounded to the nearest integer. If
        this is an array, the subsampling factor will be different for
        each dimension. Default: 1.
    '''
    def __init__(self, detector_grid, filter, EM_gain = 1, EM_saturate = None , subsampling=1):
        NoisyDetector.__init__(self, detector_grid, subsampling)

        # Setting the start charge level.
        self.accumulated_charge = 0

        # The parameters.
        self.dark_current_rate = 0.001
        self.read_noise = 0.125
        self.flat_field = 0
        self.include_photon_noise = True
        self.max_fps = 200
        self.detector_size = 512
        self.shutter_type = 'Rolling'
        self.detector_type = "Test"
        self.number_of_subdivisions = 32

        # Set Quantum Efficiency based on filter
        if filter == 'U' or filter == 'u':
            self.QE = 0.494
        elif filter == 'B' or filter == 'b':
            self.QE = 0.795
        elif filter == 'V' or filter == 'v':
            self.QE = 0.902
        elif filter == 'R' or filter == 'r':
            self.QE = 0.894
        elif filter == 'I' or filter == 'i':
            self.QE = 0.499
        else:
            raise ValueError("Error, invalid filter name.")

    def integrate(self, wavefront, dt, weight=1):
        '''Integrates the detector.

        Identical to the integrate funcion of NoisyDetector except loss due to Quantum Efficiency is included.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        dt : scalar
            The integration time in units of time.
        weight : scalar
            Weight of every unit of integration time.
        '''
        
        # The power that the detector detects during the integration.
        if hasattr(wavefront, 'power'):
            power = wavefront.power
        else:
            power = wavefront

        self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

        # Adding the generated dark current.
        self.accumulated_charge += self.dark_current_rate * dt * weight

    def roll_shutter(self, wavefronts, layer, prop, exposure_time):
        '''Simulates a rolling shutter.

        A combination of integrate and read_out that simulates the effects of a rolling shutter.

        Parameters
        ----------
        wavefront : Wavefront or array_like
            The wavefront sets the amount of power generated per unit time.
        layer: Atmospheric layer
            The atmospheric layer 
        prop: Propagator
            The propagator for the shutter
        exposure_time: Scalar
            The total exposure time in seconds

        Returns
        ----------
        Field.shaped
            The shaped final detector image.
        '''
        number_of_rows = int(np.sqrt(self.detector_grid.size))
        row_readout_time = 1/(self.max_fps*number_of_rows)
        row_differential_time = row_readout_time*number_of_rows/self.number_of_subdivisions
        layer.t += exposure_time-(row_differential_time*self.number_of_subdivisions)
        for k in wavefronts:
            self.integrate(prop((layer(k))),exposure_time-(row_differential_time*self.number_of_subdivisions)) 
        read_noise_temp = self.read_noise
        self.read_noise = 0 # Prevent double counting of the read noise when integrating the detector twice. 
        image_comb = self.read_out()
        for j in range(self.number_of_subdivisions):
            layer.t += row_differential_time
            for k in wavefronts:
                self.integrate(prop((layer(k))),row_differential_time)       
            self.read_noise = read_noise_temp
            image_row = self.read_out()
            start = int(self.detector_grid.size*(j)/self.number_of_subdivisions)
            end = int(self.detector_grid.size*((j+1)/self.number_of_subdivisions))
            image_comb[start:end]+=image_row[start:end]        
        return image_comb.shaped
