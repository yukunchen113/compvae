import tensorflow as tf
import numpy as np
from utilities.standard import ImageMSE
from core.model.architectures import CondVAE

def make_mask_config(config_obj):
	"""Converts a regular config to a mask config by adding functionality

	WARNING: this may alter the config_obj in place
	
	Args:
		config_obj (class): config object
	"""
	config_obj._get_model = CondVAE

	def hparam_schedule(steps):
		# use increasing weight hyper parameter
		gamma = (steps-10000)/30000
		return gamma

	config_obj.hparam_schedule = hparam_schedule
	return config_obj

def make_comp_config(config_obj, mask_obj, randomize_mask_step=False):
	"""Converts a given config to be compatible with CompVAE networks

	This alters the number of channels to the relevant amount
	wraps preprocessing to what a compVAE would take in
	wraps loss function 

	WARNING: this may alter the config_obj in place

	"""
	# set config mask
	

	config_obj.mask_obj = mask_obj

	config_obj.num_channels = 6

	def loss_process(loss): # apply preprocess to loss
		loss_recon = config_obj.mask_obj.apply(loss[:,:,:,:3])
		loss = tf.concat((loss_recon, loss[:,:,:,3:]), -1)
		return loss

	config_obj.loss_func = ImageMSE(loss_process=loss_process)
	_preprocessing = config_obj.preprocessing
	def preprocessing(*args, **kwargs):
		inputs = _preprocessing(*args, **kwargs)
		config_obj.mask_obj(inputs)
		inputs = np.concatenate((inputs, config_obj.mask_obj.mask), -1)
		return inputs

	config_obj.preprocessing = preprocessing

	if randomize_mask_step:
		def latent_space_distance():
			mean = 0.2
			std = 0.05
			return np.abs(np.random.normal(mean,std))

		mask_obj._default_latent_space_distance = latent_space_distance

	return config_obj