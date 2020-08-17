"""This file contains default configuration for training a compvae
"""
import tensorflow as tf
import disentangle as dt 
import numpy as np
from utilities.standard import ConfigMetaClass, ImageMSE, ImageBCE
from core.train.manager import TrainVAE
import tensorflow_datasets as tfds


#limit GPU usage (from tensiorflow code)
gpus = tf.config.experimental.list_physical_devices('GPU')
for i in gpus:
	tf.config.experimental.set_memory_growth(i, True)

def _get_inputs_test_handles(group_size, dataset_manager):
	gl = dataset_manager.groups_list
	dataset_manager.groups_list = gl[group_size:]
	return gl[:group_size]

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
		self.train_status_path = "train_status.npz"

	def _set_dataset(self):
		self.dataset_manager, self.dataset = dt.dataset.get_celeba_data(
			dt.general.constants.datapath, 
			is_HD=False,
			group_num=8)
		self.inputs_test_handle = _get_inputs_test_handles(2, self.dataset_manager)

	@property
	def inputs_test(self):
		# do not store inputs test data to minimize pickle size
		# assumes that dataset function will not change
		self.dataset_manager.load(self.inputs_test_handle)
		return self.dataset_manager.images
	
	def _set_model(self):
		self.random_seed = None
		self.num_latents = 10
		self.beta = 30
		self._get_model = dt.architectures.vae.BetaTCVAE
		
	def _set_training(self):
		self.batch_size = 32
		self.approve_run = True
		self.loss_func = ImageBCE()
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
		#self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
		self.total_steps = 100000
		self.model_save_steps = 1000
		self.is_train = True
		self.TrainVAE = TrainVAE

	def get_model(self, *args, **kwargs):
		model = self._get_model(*args, **kwargs)
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
		self.dataset_manager, self.dataset = dt.dataset.get_celeba_data(
			dt.general.constants.datapath, 
			is_HD=64,
			group_num=8)
		self.inputs_test_handle = _get_inputs_test_handles(2, self.dataset_manager)
	
	def preprocessing(self, inputs, image_crop_size=[50,50], final_image_size=[64,64]):
		input_shape = inputs.shape[1:-1]
		if not (input_shape == np.asarray(final_image_size)).all():
			inputs = tf.image.convert_image_dtype(inputs, tf.float32)
			inputs = tf.image.resize(inputs, final_image_size)
		inputs = super().preprocessing(inputs, image_crop_size, final_image_size)
		return inputs

class Config256(Config):
	def _set_dataset(self):
		self.dataset_manager, self.dataset = dt.dataset.get_celeba_data(
			dt.general.constants.datapath, 
			is_HD=256,
			group_num=8)
		self.inputs_test_handle = _get_inputs_test_handles(2, self.dataset_manager)

	def _set_training(self):
		super()._set_training()
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
		self.total_steps = 100000

	def get_model(self, *args, **kwargs):
		model = super().get_model(*args, **kwargs)
		model.create_encoder_decoder_256(**kwargs)
		return model

	def preprocessing(self, inputs, image_crop_size=[200,200], final_image_size=[256,256]):
		input_shape = inputs.shape[1:-1]
		if not (input_shape == np.asarray(final_image_size)).all():
			inputs = tf.image.resize(inputs, final_image_size)
		return super().preprocessing(inputs, image_crop_size, final_image_size)

class TFDSShapes3DWrapper():
	def __init__(self, dataset_manager):
		self.dataset_manager = dataset_manager

	def __call__(self, batch=None):
		dm = self.dataset_manager
		if not batch is None:
			dm=dm.batch(batch)
		dataset = dm.as_numpy_iterator().__iter__().__next__()
		return dataset

	def batch(self, batch_size):
		self.dataset_manager = self.dataset_manager.batch(batch_size, drop_remainder=True)
		return self

class ConfigShapes3D(Config):
	"""Config used to train shapes3d
	"""
	def _set_dataset(self):
		self._dataset = None
		self._dataset_manager = None	
		self._inputs_test = None
	def _set_model(self):
		super()._set_model()
		self.beta = 8
		self.num_latents = 3
	
	@property
	def dataset(self):
		self._dataset = TFDSShapes3DWrapper(self.dataset_manager)
		return self._dataset

	@property
	def dataset_manager(self):
		if self._dataset_manager is None:
			def get_shape_3d_data(data):
				images = data["image"]
				labels = tf.convert_to_tensor([v for k,v in data.items(
					) if not k =="image"], dtype=tf.float32)
				return images, labels

			self._dataset_manager = tfds.load('shapes3d',
			 	data_dir=dt.general.constants.tfds_datapath,
			 	split="train", # this is all split examples for shape3d
		 		)
			self._dataset_manager = self._dataset_manager.shuffle(20000)
			self._dataset_manager = self._dataset_manager.repeat(-1)
			self._dataset_manager = self._dataset_manager.prefetch(tf.data.experimental.AUTOTUNE)
			self._dataset_manager = self._dataset_manager.map(get_shape_3d_data)

		if not self._dataset is None:
			self._dataset_manager = self._dataset.dataset_manager

		return self._dataset_manager

	@property
	def inputs_test(self):
		# do not store inputs test data to minimize pickle size
		# assumes that dataset function will not change
		if self._inputs_test is None:
			self._inputs_test, _ = self.dataset(32)
		return self._inputs_test
	
	def preprocessing(self, inputs, *ar, **kw):
		return tf.image.convert_image_dtype(inputs, tf.float32)

	def _set_training(self):
		super()._set_training()
		self.batch_size = 100
		#self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)