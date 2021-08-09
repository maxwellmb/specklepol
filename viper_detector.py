import numpy as np
import sys
#sys.path.append('C:\\Users\\User\\hcipy\\hcipy')
#from .util import large_poisson
#from .field import subsample_field, make_supersampled_grid
from hcipy import *


class Detector(object):
	'''Base class for a detector.

	Parameters
	----------
	detector_grid : Grid
		The grid on which the detector returns its images. These indicate
		the centers of the pixels.
	subsamping : integer or scalar or ndarray
		The number of subpixels per pixel along one axis. For example, a
		value of 2 indicates that 2x2=4 subpixels are used per pixel. If
		this is a scalar, it will be rounded to the nearest integer. If
		this is an array, the subsampling factor will be different for
		each dimension. Default: 1.

	Attributes
	----------
	input_grid : Grid
		The grid that is expected as input.
	'''
	def __init__(self, detector_grid, subsamping=1):
		self.detector_grid = detector_grid
		self.subsamping = subsamping

		if subsamping > 1:
			self.input_grid = make_supersampled_grid(detector_grid, subsamping)
		else:
			self.input_grid = detector_grid

	def integrate(self, wavefront, dt, weight=1):
		'''Integrates the detector.

		Parameters
		----------
		wavefront : Wavefront or array_like
			The wavefront sets the amount of power generated per unit time.
		dt : scalar
			The integration time in units of time.
		weight : scalar
			Weight of every unit of integration time.
		'''
		raise NotImplementedError()

	def read_out(self):
		'''Reads out the detector.

		No noise will be added to the image.

		Returns
		----------
		Field
			The final detector image.
		'''
		raise NotImplementedError()

	def __call__(self, wavefront, dt=1, weight=1):
		'''Integrate and read out the detector.

		This is a convenience function to avoid having to call two functions
		in quick succession.

		Parameters
		----------
		wavefront : Wavefront or array_like
			The wavefront sets the amount of power generated per unit time.
		dt : scalar
			The integration time in units of time.
		weight : scalar
			Weight of every unit of integration time.

		Returns
		----------
		Field
			The final detector image.
		'''
		self.integrate(wavefront, dt, weight)
		return self.read_out()

class ProEM512(Detector):
	def __init__(self, detector_grid, filter, FPS=61, detector_size = 512, subsampling=1):
		Detector.__init__(self, detector_grid, subsampling)

		# Setting the start charge level.
		self.accumulated_charge = 0

		# The parameters.
		self.dark_current_rate = 0.001
		self.read_noise = 0.125
		self.flat_field = 0
		self.include_photon_noise = True
		self.fps = FPS
		self.detector_size = detector_size
		self.shutter_type = 'Global'

		# Set Quantum Efficiency based on filter
		if filter == 'U' or filter == 'u':
			self.QE = 0.494/2
		elif filter == 'B' or filter == 'b':
			self.QE = 0.795/2
		elif filter == 'V' or filter == 'v':
			self.QE = 0.902/2
		elif filter == 'R' or filter == 'r':
			self.QE = 0.894/2
		elif filter == 'I' or filter == 'i':
			self.QE = 0.499/2
		else:
			raise ValueError("Error, invalid filter name.")

	@property
	def flat_field(self):
		return self._flat_field

	@flat_field.setter
	def flat_field(self, flat_field):
		# If the flatfield parameters was a scalar, we will generate a flat field map that will
		# be constant for this object until flat_field is manually changed.
		if np.isscalar(flat_field):
			self._flat_field = np.random.normal(loc=1.0, scale=flat_field, size=self.detector_grid.size)
		else:
			self._flat_field = flat_field

	def integrate(self, wavefront, dt, weight=1):
		# The power that the detector detects during the integration.
		if hasattr(wavefront, 'power'):
			power = wavefront.power
		else:
			power = wavefront

		self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

		# Adding the generated dark current.
		self.accumulated_charge += self.dark_current_rate * dt * weight

	def read_out(self):
		# Make sure not to overwrite output
		output_field = self.accumulated_charge.copy()

		# Adding photon noise.
		if self.include_photon_noise:
			output_field = large_poisson(output_field, thresh=1e6)

		# Adding flat field errors.
		output_field *= self.flat_field

		# Adding read-out noise.
		output_field += np.random.normal(loc=0, scale=self.read_noise, size=output_field.size)

		# Reset detector
		self.accumulated_charge = 0

		return output_field
	def output_read_noise(self):
		return self.read_noise
	def output_dark_current(self):
		return self.dark_current_rate
	def output_flat_field(self):
		return self.flat_field
	def output_photon_noise(self):
		return self.include_photon_noise
	def output_fps(self):
		return self.fps
	def output_detector_size(self):
		return self.detector_size
	def output_shutter_type(self):
		return self.shutter_type
	def output_QE(self):
		return self.QE

class iXon887(Detector):
	def __init__(self, detector_grid, FPS=56, detector_size = 512, subsampling=1):
		Detector.__init__(self, detector_grid, subsampling)

		# Setting the start charge level.
		self.accumulated_charge = 0

		# The parameters.
		self.dark_current_rate = 0.003
		self.read_noise = 0.089
		self.flat_field = 0
		self.include_photon_noise = True
		self.fps = FPS
		self.detector_size = detector_size
		self.shutter_type = 'Global'

		# Set Quantum Efficiency based on filter
		if filter == 'U' or filter == 'u':
			self.QE = 0.258/2
		elif filter == 'B' or filter == 'b':
			self.QE = 0.735/2
		elif filter == 'V' or filter == 'v':
			self.QE = 0.957/2
		elif filter == 'R' or filter == 'r':
			self.QE = 0.869/2
		elif filter == 'I' or filter == 'i':
			self.QE = 0.479/2
		else:
			raise ValueError("Error, invalid filter name.")

	@property
	def flat_field(self):
		return self._flat_field

	@flat_field.setter
	def flat_field(self, flat_field):
		# If the flatfield parameters was a scalar, we will generate a flat field map that will
		# be constant for this object until flat_field is manually changed.
		if np.isscalar(flat_field):
			self._flat_field = np.random.normal(loc=1.0, scale=flat_field, size=self.detector_grid.size)
		else:
			self._flat_field = flat_field

	def integrate(self, wavefront, dt, weight=1):
		# The power that the detector detects during the integration.
		if hasattr(wavefront, 'power'):
			power = wavefront.power
		else:
			power = wavefront

		self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

		# Adding the generated dark current.
		self.accumulated_charge += self.dark_current_rate * dt * weight

	def read_out(self):
		# Make sure not to overwrite output
		output_field = self.accumulated_charge.copy()

		# Adding photon noise.
		if self.include_photon_noise:
			output_field = large_poisson(output_field, thresh=1e6)

		# Adding flat field errors.
		output_field *= self.flat_field

		# Adding read-out noise.
		output_field += np.random.normal(loc=0, scale=self.read_noise, size=output_field.size)

		# Reset detector
		self.accumulated_charge = 0

		return output_field
	def output_read_noise(self):
		return self.read_noise
	def output_dark_current(self):
		return self.dark_current_rate
	def output_flat_field(self):
		return self.flat_field
	def output_photon_noise(self):
		return self.include_photon_noise
	def output_fps(self):
		return self.fps
	def output_detector_size(self):
		return self.detector_size
	def output_shutter_type(self):
		return self.shutter_type
	def output_QE(self):
		return self.QE

class ORCA_Quest(Detector):
	def __init__(self, detector_grid, filter, FPS=532, detector_size = 512, subsampling=1):
		Detector.__init__(self, detector_grid, subsampling)

		# Setting the start charge level.
		self.accumulated_charge = 0

		# The parameters.
		self.dark_current_rate = 0.006
		self.read_noise = 0.43
		self.flat_field = 0
		self.include_photon_noise = True
		self.fps = FPS
		self.detector_size = detector_size
		self.shutter_type = 'Rolling'
		
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

	@property
	def flat_field(self):
		return self._flat_field

	@flat_field.setter
	def flat_field(self, flat_field):
		# If the flatfield parameters was a scalar, we will generate a flat field map that will
		# be constant for this object until flat_field is manually changed.
		if np.isscalar(flat_field):
			self._flat_field = np.random.normal(loc=1.0, scale=flat_field, size=self.detector_grid.size)
		else:
			self._flat_field = flat_field

	def integrate(self, wavefront, dt, weight=1):
		# The power that the detector detects during the integration.
		if hasattr(wavefront, 'power'):
			power = wavefront.power
		else:
			power = wavefront

		self.accumulated_charge += self.QE*subsample_field(power, subsampling=self.subsamping, new_grid=self.detector_grid, statistic='sum') * dt * weight

		# Adding the generated dark current.
		self.accumulated_charge += self.dark_current_rate * dt * weight

	def read_out(self):
		# Make sure not to overwrite output
		output_field = self.accumulated_charge.copy()

		# Adding photon noise.
		if self.include_photon_noise:
			output_field = large_poisson(output_field, thresh=1e6)

		# Adding flat field errors.
		output_field *= self.flat_field

		# Adding read-out noise.
		output_field += np.random.normal(loc=0, scale=self.read_noise, size=output_field.size)

		# Reset detector
		self.accumulated_charge = 0
		return output_field
	
	def roll_shutter(self, wavefronts, layer, prop, exposure_time, number_of_subdivisions):
		number_of_rows = int(np.sqrt(self.detector_grid.size))
		row_readout_time = 1/(self.fps*number_of_rows)
		row_differential_time = row_readout_time*number_of_rows/number_of_subdivisions
		layer.t += exposure_time-(row_differential_time*number_of_subdivisions)
		for k in wavefronts:
			self.integrate(prop((layer(k))),exposure_time-(row_differential_time*number_of_subdivisions)) 
		image_comb = self.read_out()
		for j in range(number_of_subdivisions):
			layer.t += row_differential_time
			for k in wavefronts:
				self.integrate(prop((layer(k))),row_differential_time) 
			image_row = self.read_out()
			start = int(self.detector_grid.size*(j)/number_of_subdivisions)
			end = int(self.detector_grid.size*((j+1)/number_of_subdivisions))
			image_comb[start:end]+=image_row[start:end]        
		return image_comb.shaped
	
	def output_read_noise(self):
		return self.read_noise
	def output_dark_current(self):
		return self.dark_current_rate
	def output_flat_field(self):
		return self.flat_field
	def output_photon_noise(self):
		return self.include_photon_noise
	def output_fps(self):
		return self.fps
	def output_detector_size(self):
		return self.detector_size
	def output_shutter_type(self):
		return self.shutter_type
	def output_QE(self):
		return self.QE

class FrameCorrector(object):
	def correct(self, img):
		return img

class BasicFrameCorrector(FrameCorrector):
	def __init__(self, dark=0, flat_field=1):
		self.dark = dark
		self.flat_field = flat_field

	def correct(self, img):
		return (img - self.dark) / self.flat_field
