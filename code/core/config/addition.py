import tensorflow as tf
import numpy as np
from utilities.standard import ImageMSE
import core.model.architectures as ar
from core.train.manager import TrainProVLAE,TrainLVAE

###############
# VLAE Method #
###############
class LargeProVLAE(ar.ProVLAE):
	def create_default_vae(self, **kwargs):
		# default encoder decoder pair:
		print("WARNING: this is using a depreciated method")
		self.create_large_provlae64(**kwargs)

class LargeVLAE(ar.VLAE):
	def create_default_vae(self, **kwargs):
		# default encoder decoder pair:
		self.create_large_ladder64(**kwargs)

class LargeLVAE(ar.LVAE):
	def create_default_vae(self, **kwargs):
		# default encoder decoder pair:
		self.create_large_ladder64(**kwargs)

def make_lvae_large(config_obj):
	config_obj._get_model = LargeLVAE
	# training object
	config_obj.TrainVAE = TrainLVAE
	return config_obj

def make_lvae_small(config_obj):
	config_obj._get_model = ar.LVAE

	#mask = ar.LatentMaskLVAE([12,6,3])
	#config_obj._get_model = mask(config_obj._get_model)
	#subspace = ar.LatentSubspaceLVAE([[0,0,1,1,2,2,3,3,4,4,5,5],[0,0,1,1,2,2]])
	#config_obj._get_model = subspace(config_obj._get_model)

	# training object
	config_obj.TrainVAE = TrainLVAE
	return config_obj

def make_vlae_large(config_obj):
	config_obj._get_model = lambda *arg,**kw: ar.Sequential(base=LargeVLAE(*arg,**kw), additionals=[
			ar.ResidualLatentRouting(apply_on_decoder=False),
			ar.LatentPriorRouting(),
			#LatentMask(apply_on_regularization=False),
			])
	# model parameter setup
	if not hasattr(config_obj, "gamma"): config_obj.gamma = 0.1
	def hparam_schedule(step, start_step=5000, alpha_duration=5000):
		# start_step is where the a new latent space starts getting integrated
		# alpha_duration is how long a new latent space takes for architecture to get integrated
		# beta_duration is how long it takes for a beta value to drop to a certain value

		# changes alpha
		alpha = [0]*(4)
		alpha[-1] = 1
		for i in range(1,len(alpha)):
			alpha[len(alpha)-i-1] = np.clip((step-start_step*i)/alpha_duration, 0, 1) # after the first alpha_duration steps, evolve alpha for a steps
		return dict(alpha=alpha)
	if not hasattr(config_obj, "hparam_schedule"): config_obj.hparam_schedule = hparam_schedule
	# training object
	config_obj.TrainVAE = TrainProVLAE
	return config_obj

def make_vlae_small(config_obj):
	config_obj._get_model = lambda *arg,**kw: ar.Sequential(base=ar.VLAE(*arg,**kw), additionals=[
		ar.LatentPriorRouting(),
		ar.ResidualLatentRouting(apply_on_decoder=False),
		#LatentMask(apply_on_regularization=False),
		])
	# model parameter setup
	if not hasattr(config_obj, "gamma"): config_obj.gamma = 0.5
	def hparam_schedule(step, start_step=5000, alpha_duration=5000):
		# start_step is where the a new latent space starts getting integrated
		# alpha_duration is how long a new latent space takes for architecture to get integrated
		# beta_duration is how long it takes for a beta value to drop to a certain value
		# changes alpha
		alpha = [0]*(3)
		alpha[-1] = 1
		for i in range(1,len(alpha)):
			alpha[len(alpha)-i-1] = np.clip((step-start_step*i)/alpha_duration, 0, 1) # after the first alpha_duration steps, evolve alpha for a steps
		return dict(alpha=alpha)
	if not hasattr(config_obj, "hparam_schedule"): config_obj.hparam_schedule = hparam_schedule
	# training object
	config_obj.TrainVAE = TrainProVLAE
	return config_obj

def make_vlae_large_old(config_obj):
	config_obj._get_model = LargeProVLAE
	# model parameter setup
	if not hasattr(config_obj, "gamma"): config_obj.gamma = 0.1
	def hparam_schedule(step, start_step=5000, alpha_duration=5000):
		# start_step is where the a new latent space starts getting integrated
		# alpha_duration is how long a new latent space takes for architecture to get integrated
		# beta_duration is how long it takes for a beta value to drop to a certain value

		# changes alpha
		alpha = [0]*(4)
		alpha[-1] = 1
		for i in range(1,len(alpha)):
			alpha[len(alpha)-i-1] = np.clip((step-start_step*i)/alpha_duration, 0, 1) # after the first alpha_duration steps, evolve alpha for a steps
		return dict(alpha=alpha)
	if not hasattr(config_obj, "hparam_schedule"): config_obj.hparam_schedule = hparam_schedule
	# training object
	config_obj.TrainVAE = TrainProVLAE
	return config_obj

def make_vlae_small_old(config_obj):
	config_obj._get_model = ar.ProVLAE
	# model parameter setup
	if not hasattr(config_obj, "gamma"): config_obj.gamma = 0.5
	def hparam_schedule(step, start_step=5000, alpha_duration=5000):
		# start_step is where the a new latent space starts getting integrated
		# alpha_duration is how long a new latent space takes for architecture to get integrated
		# beta_duration is how long it takes for a beta value to drop to a certain value
		# changes alpha
		alpha = [0]*(3)
		alpha[-1] = 1
		for i in range(1,len(alpha)):
			alpha[len(alpha)-i-1] = np.clip((step-start_step*i)/alpha_duration, 0, 1) # after the first alpha_duration steps, evolve alpha for a steps
		return dict(alpha=alpha)
	if not hasattr(config_obj, "hparam_schedule"): config_obj.hparam_schedule = hparam_schedule
	# training object
	config_obj.TrainVAE = TrainProVLAE
	return config_obj

########################
# Comp and Mask Method #
########################
def make_mask_config(config_obj):
	"""Converts a regular config to a mask config by adding functionality

	WARNING: this may alter the config_obj in place
	
	Args:
		config_obj (class): config object
	"""
	config_obj._get_model = ar.CondVAE

	def hparam_schedule(step):
		# use increasing weight hyper parameter
		gamma = (step-10000)/30000
		return dict(gamma=gamma)

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

	assert False, "no variable channel support, need to reconfigure this. TODO"
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