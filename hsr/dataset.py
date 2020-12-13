"""
# TBD: make input test size parameterizable
"""
import tensorflow as tf
import disentangle as dt 
import numpy as np
import tensorflow_datasets as tfds


#limit GPU usage (from tensiorflow code)
gpus = tf.config.experimental.list_physical_devices('GPU')
for i in gpus:
	tf.config.experimental.set_memory_growth(i, True)

def _get_test_handles(group_size, dataset_manager):
	gl = dataset_manager.groups_list
	dataset_manager.groups_list = gl[group_size:]
	return gl[:group_size]

class _Base:
	def __init__(self):
		self.dataset_manager = None
	def test(self):
		return None
	def train(self, batch_size):
		return self.dataset_manager.batch(batch_size)

class CelebA(_Base):
	"""
	CelebA

	Cuts to a size of 64x64 for inputs
	"""
	def __init__(self):
		self.dataset_manager, self.dataset = dt.dataset.get_celeba_data(
			dt.general.constants.datapath, 
			is_HD=False,
			group_num=8)
		self.test_handle = _get_test_handles(2, self.dataset_manager)

	def test(self):
		# do not store inputs test data to minimize pickle size
		# assumes that dataset function will not change
		self.dataset_manager.load(self.test_handle)
		return self.dataset_manager.images
	
	def train(self,*ar,**kw):
		return self.dataset_manager.batch(*ar,**kw)
	
	def preprocess(self, inputs, image_crop_size = [128,128], final_image_size=[64,64]):
		inputs=tf.image.crop_to_bounding_box(inputs, 
			(inputs.shape[-3]-image_crop_size[0])//2,
			(inputs.shape[-2]-image_crop_size[1])//2,
			image_crop_size[0],
			image_crop_size[1],
			)
		inputs = tf.image.convert_image_dtype(inputs, tf.float32)
		inputs = tf.image.resize(inputs, final_image_size)
		return inputs
	
class CelebAHQ64(CelebA):
	"""
	Uses CelebA-HQ
	
	Cuts to a size of 64x64 for inputs
	"""
	def __init__(self):
		self.dataset_manager, self.dataset = dt.dataset.get_celeba_data(
			dt.general.constants.datapath, 
			is_HD=64,
			group_num=8)
		self.test_handle = _get_test_handles(2, self.dataset_manager)
	
	def preprocess(self, inputs, image_crop_size=[50,50], final_image_size=[64,64]):
		input_shape = inputs.shape[1:-1]
		if not (input_shape == np.asarray(final_image_size)).all():
			inputs = tf.image.convert_image_dtype(inputs, tf.float32)
			inputs = tf.image.resize(inputs, final_image_size)
		inputs = super().preprocess(inputs, image_crop_size, final_image_size)
		return inputs

class CelebAHQ256(CelebA):
	"""
	Uses CelebA-HQ
	
	Cuts to a size of 256x256 for inputs
	"""
	def __init__(self):
		self.dataset_manager, self.dataset = dt.dataset.get_celeba_data(
			dt.general.constants.datapath, 
			is_HD=256,
			group_num=8)
		self.test_handle = _get_test_handles(2, self.dataset_manager)

	def preprocess(self, inputs, image_crop_size=[200,200], final_image_size=[256,256]):
		input_shape = inputs.shape[1:-1]
		if not (input_shape == np.asarray(final_image_size)).all():
			inputs = tf.image.resize(inputs, final_image_size)
		return super().preprocess(inputs, image_crop_size, final_image_size)

class TFDSWrapper():
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

class _TFDSBase(_Base):
	"""
	Base TFDS configuration
	"""
	def __init__(self):
		self._dataset = None
		self._dataset_manager = None	
		self._test = None
	
	@property
	def dataset(self):
		self._dataset = TFDSWrapper(self.dataset_manager)
		return self._dataset

	@property
	def dataset_manager(self):
		raise Exception("Config Undefined")

	def test(self):
		# do not store inputs test data to minimize pickle size
		# assumes that dataset function will not change
		# TBD: make input test size parameterizable
		if self._test is None:
			self._test, _ = self.dataset(32)
		return self._test
	
	def preprocess(self, inputs, *ar, **kw):
		return tf.image.convert_image_dtype(inputs, tf.float32)

class Shapes3D(_TFDSBase):
	"""
	Uses Shapes3D from tfds
	
	size of 64x64 for inputs
	"""
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
