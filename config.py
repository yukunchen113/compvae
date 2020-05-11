"""This file contains default configuration for training a compvae
"""
import tensorflow as tf
import os
import utils as ut 
import numpy as np
from utilities import ConfigMetaClass
from utilities import ImageMSE
import train
#limit GPU usage (from tensiorflow code)
gpus = tf.config.experimental.list_physical_devices('GPU')
for i in gpus:
	tf.config.experimental.set_memory_growth(i, True)

def get_inputs_test_handles(group_size, dataset_manager, dataset):
	dataset(group_size, False, True)
	return dataset_manager.last_group_list

class Config(metaclass=ConfigMetaClass):
	"""
	This is the default Config. Applied to regular CelebA dataset. Uses a regular TCVAE

	Cuts to a size of 64x64 for inputs
	"""
	def __init__(self):
		self._set_paths()
		self._set_dataset()
		self._set_model()
		self._set_training()

	def _set_paths(self):
		self.image_dir = "images"
		self.model_setup_dir = "model_setup"

		# save files
		self.model_save_file = "model_weights.h5" # model weights save file
		self.model_parameters_path = "model_parameters.txt"

	def _set_dataset(self):
		self.dataset_manager, self.dataset = ut.dataset.get_celeba_data(
			ut.general_constants.datapath, 
			is_HD=False,
			group_num=8)
		self.inputs_test_handle = get_inputs_test_handles(2, self.dataset_manager, self.dataset)

	@property
	def inputs_test(self):
		# do not store inputs test data to minimize pickle size
		# assumes that dataset function will not change
		self.dataset_manager.load(self.inputs_test_handle)
		return self.dataset_manager.images
	
	def _set_model(self):
		self.num_latents = 10
		self.num_channels = 3
		self.beta_value = 30

	def _set_training(self):
		self.batch_size = 32
		self.approve_run = True
		self.loss_func = ImageMSE()
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
		self.total_steps = 100000
		self.model_save_steps = 1000
		self.TrainVAE = train.TrainVAE

	def get_model(self, *args, **kwargs):
		model = ut.tf_custom.architectures.variational_autoencoder.BetaTCVAE(
			*args, **kwargs)
		return model

	def preprocessing(self, inputs, image_crop_size = [128,128], final_image_size=[64,64]):
		inputs=tf.image.crop_to_bounding_box(inputs, 
			(inputs.shape[-3]-image_crop_size[0])//2,
			(inputs.shape[-2]-image_crop_size[1])//2,
			image_crop_size[0],
			image_crop_size[1],
			)
		inputs = tf.image.convert_image_dtype(inputs, tf.float32)
		inputs = tf.image.resize(inputs, final_image_size)
		return inputs

class Config64(Config):
	def _set_dataset(self):
		self.dataset_manager, self.dataset = ut.dataset.get_celeba_data(
			ut.general_constants.datapath, 
			is_HD=64,
			group_num=8)
		self.inputs_test_handle = get_inputs_test_handles(2, self.dataset_manager, self.dataset)
	
	def preprocessing(self, inputs, image_crop_size=[50,50], final_image_size=[64,64]):
		return super().preprocessing(inputs, image_crop_size, final_image_size)

class Config256(Config):
	def _set_dataset(self):
		self.dataset_manager, self.dataset = ut.dataset.get_celeba_data(
			ut.general_constants.datapath, 
			is_HD=256,
			group_num=8)
		self.inputs_test_handle = get_inputs_test_handles(2, self.dataset_manager, self.dataset)

	def _set_training(self):
		super()._set_training()
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
		self.total_steps = 100000

	def get_model(self, *args, **kwargs):
		model = super().get_model(*args, **kwargs)
		model.create_encoder_decoder_256(**kwargs)
		return model

	def preprocessing(self, inputs, image_crop_size=[200,200], final_image_size=[256,256]):
		return super().preprocessing(inputs, image_crop_size, final_image_size)

def make_comp_config(config_obj, mask_obj):
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
	


	"""
	class CompConfig(Config):
		def __init__(self, mask_obj=None):
			assert not mask_obj is None, "mask_obj must be defined in config for compvae"
			self.mask_obj = mask_obj
			super().__init__()
		
		def _set_model(self):
			super()._set_model()
			self.num_channels = 6
		
		def _set_training(self):
			super()._set_training()
			def loss_process(loss): # apply preprocess to loss
				loss_recon = self.mask_obj.apply(loss[:,:,:,:3])
				loss = tf.concat((loss_recon, loss[:,:,:,3:]), -1)
				return loss
			self.loss_func = ImageMSE(loss_process=loss_process) 
		
		def preprocessing(self, *args, **kwargs):
			inputs = super().preprocessing(*args, **kwargs)
			self.mask_obj(inputs)
			inputs = np.concatenate((inputs, self.mask_obj.mask), -1)
			return inputs
	"""
	return config_obj
