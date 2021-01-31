"""
# TBD: make input test size parameterizable
"""
import tensorflow as tf
import disentangle as dt 
import numpy as np
import tensorflow_datasets as tfds
import hiershapes as hs

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

##########################
# Observational Datasets #
##########################
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

	def test(self, batch_size=None, return_labels=False):
		# do not store inputs test data to minimize pickle size
		# assumes that dataset function will not change
		# TBD: make input test size parameterizable
		if self._test is None or not batch_size is None:
			if batch_size is None: batch_size = 32
			self._test, labels = self.dataset(batch_size)
		if return_labels: return self._test, labels
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

#######################
# Generative Datasets #
#######################
class HierShapesBase(_Base):
	def __init__(self, use_server=True, use_pool=True, run_once=False, num_proc=20, prefetch=20, pool_size=10, port=None):
		self.use_server = use_server
		self.port = port
		if use_server: assert use_pool, "if using server, must be with pool"
		self.num_proc=num_proc
		assert prefetch>=num_proc, "prefetch must greater than num_proc"
		self.prefetch=prefetch
		self.pool_size=pool_size
		self.use_pool = use_pool
		self.run_once = run_once
		self.test_labels = None
		self.test_images = None
		self.server = None
		self.dataset_manager = self.start_server()

	def test(self, batch_size=None, return_labels=False):
		if self.test_labels is None or not batch_size is None:
			if batch_size is None: batch_size = 64
			self.test_images,self.test_labels = self.dataset_manager.get_batch(batch_size)
		else:
			self.test_images = self.dataset_manager.get_images(self.test_labels)
		if return_labels: return self.test_images, self.test_labels
		return self.test_images
	
	def preprocess(self, inputs, *ar, **kw):
		return tf.image.convert_image_dtype(inputs, tf.float32)
	
	def start_server(self):
		# Server #
		scene, parameters = self.get_scene_parameters()
		if self.use_server:
			self.server = hs.dataset.ServerClient(scene=scene, randomize_parameters_func=parameters, 
				num_proc=self.num_proc, prefetch=self.prefetch, pool_size=self.pool_size, port=self.port, retrieval_batch_size=100)
			client = self.server()
		elif self.use_pool:
			client = hs.dataset.ParallelBatch(scene=scene, randomize_parameters_func=parameters, 
				num_proc=self.num_proc, prefetch=self.prefetch, pool_size=self.pool_size,retrieval_batch_size=100)
		else:
			client = hs.dataset.MultiProcessBatch(scene=scene, randomize_parameters_func=parameters, 
				num_proc=self.num_proc, prefetch=self.prefetch, run_once=self.run_once)
		return client

	def close(self, is_terminate_server=False):
		self.dataset_manager.close()
		if is_terminate_server and not self.server is None: self.server.close() 

	def __getstate__(self):
		d = dict(self.__dict__)
		d["test_images"] = None
		d["server"] = None
		return d

class HierShapesBoxheadSimple2(HierShapesBase):
	def get_scene_parameters(self, default_params={}, is_filter_val=True):
		# get_intermediate_values must be false when generating factors
		
		adjustable_parameters = ["color","scale","eye_color","azimuth","_overall_eye_color","_wall_color","_floor_color"]
		for k in default_params.keys(): assert k in adjustable_parameters, f"{k} not in adjustable_parameters: {adjustable_parameters}" 
		#boxhead = hs.scene.BoxHead(eyes=[0])
		boxhead = hs.scene.BoxHeadCentralEye() # this is only used for parameter bounds here.
		# parameters #
		parameters = hs.utils.Parameters(is_filter_val=is_filter_val)
		def head():
			np.random.seed()
			out = {}
			out["color"] = hs.utils.quantized_uniform(*boxhead.parameters["color"][0],n_quantized=10) if not "color" in default_params else default_params["color"]
			scale = hs.utils.quantized_uniform(1,1.25,n_quantized=10)
			out["scale"] = np.asarray([scale, scale, scale]) if not "scale" in default_params else default_params["scale"]
			return out
		parameters.add_parameters("head", head)

		def eyes(color, **kw):
			np.random.seed()
			out = {}
			out["_overall_eye_color"] = hs.utils.quantized_uniform(0,1,n_quantized=7) if not "_overall_eye_color" in default_params else default_params["_overall_eye_color"]
			out["eye_color"] = np.mod(hs.utils.quantized_normal(0, 0.2, size=4, n_quantized=7)+out["_overall_eye_color"], 1) if not "eye_color" in default_params else default_params["eye_color"]
			return out
		parameters.add_parameters("eyes", eyes, ["head"])

		def view():
			np.random.seed()
			out = {}
			floor, wall = hs.utils.quantized_uniform(*boxhead.parameters["bg_color"][0], # [floor, wall]
				size=boxhead.parameters["bg_color"][1], n_quantized=10)
			out["_floor_color"] = floor if not "_floor_color" in default_params else default_params["_floor_color"]
			out["_wall_color"] = wall if not "_wall_color" in default_params else default_params["_wall_color"]
			out["bg_color"] = np.asarray([out["_floor_color"],out["_wall_color"]])
			out["azimuth"] = hs.utils.quantized_uniform(*boxhead.parameters["azimuth"][0],
				size=boxhead.parameters["azimuth"][1],n_quantized=10) if not "azimuth" in default_params else default_params["azimuth"]
			return out
		parameters.add_parameters("view", view)
		return boxhead, parameters

class HierShapesBoxheadSimple(HierShapesBase):
	def get_scene_parameters(self, default_params={}, is_filter_val=True):
		# get_intermediate_values must be false when generating factors
		
		adjustable_parameters = ["color","scale","eye_color","azimuth","_overall_eye_color","_wall_color","_floor_color"]
		for k in default_params.keys(): assert k in adjustable_parameters, f"{k} not in adjustable_parameters: {adjustable_parameters}" 
		#boxhead = hs.scene.BoxHead(eyes=[0])
		boxhead = hs.scene.BoxHeadCentralEye() # this is only used for parameter bounds here.
		# parameters #
		parameters = hs.utils.Parameters(is_filter_val=is_filter_val)
		def head():
			np.random.seed()
			out = {}
			out["color"] = hs.utils.quantized_uniform(*boxhead.parameters["color"][0],n_quantized=10) if not "color" in default_params else default_params["color"]
			scale = hs.utils.quantized_uniform(1,1.25,n_quantized=10)
			out["scale"] = np.asarray([scale, scale, scale]) if not "scale" in default_params else default_params["scale"]
			return out
		parameters.add_parameters("head", head)

		def eyes(color, **kw):
			np.random.seed()
			out = {}
			out["_overall_eye_color"] = hs.utils.quantized_uniform(0,1,n_quantized=7) if not "_overall_eye_color" in default_params else default_params["_overall_eye_color"]
			out["eye_color"] = np.mod(hs.utils.quantized_normal(0, 0.1, size=4, n_quantized=7)+out["_overall_eye_color"], 1) if not "eye_color" in default_params else default_params["eye_color"]
			return out
		parameters.add_parameters("eyes", eyes, ["head"])

		def view():
			np.random.seed()
			out = {}
			floor, wall = hs.utils.quantized_uniform(*boxhead.parameters["bg_color"][0], # [floor, wall]
				size=boxhead.parameters["bg_color"][1], n_quantized=10)
			out["_floor_color"] = floor if not "_floor_color" in default_params else default_params["_floor_color"]
			out["_wall_color"] = wall if not "_wall_color" in default_params else default_params["_wall_color"]
			out["bg_color"] = np.asarray([out["_floor_color"],out["_wall_color"]])
			out["azimuth"] = hs.utils.quantized_uniform(*boxhead.parameters["azimuth"][0],
				size=boxhead.parameters["azimuth"][1],n_quantized=10) if not "azimuth" in default_params else default_params["azimuth"]
			return out
		parameters.add_parameters("view", view)
		return boxhead, parameters

class HierShapesBoxhead(HierShapesBase):
	def get_scene_parameters(self, default_params={}, is_filter_val=True):
		# get_intermediate_values must be false when generating factors
		
		adjustable_parameters = ["color","scale","eye_color","azimuth","_overall_eye_color","_wall_color","_floor_color"]
		for k in default_params.keys(): assert k in adjustable_parameters, f"{k} not in adjustable_parameters: {adjustable_parameters}" 
		#boxhead = hs.scene.BoxHead(eyes=[0])
		boxhead = hs.scene.BoxHeadCentralEye() # this is only used for parameter bounds here.
		# parameters #
		parameters = hs.utils.Parameters(is_filter_val=is_filter_val)
		def head():
			np.random.seed()
			out = {}
			out["color"] = hs.utils.quantized_uniform(*boxhead.parameters["color"][0],n_quantized=10) if not "color" in default_params else default_params["color"]
			scale = hs.utils.quantized_uniform(1,1.25,n_quantized=15)
			out["scale"] = np.asarray([scale, scale, scale]) if not "scale" in default_params else default_params["scale"]
			return out
		parameters.add_parameters("head", head)

		def eyes(color, **kw):
			np.random.seed()
			out = {}
			out["_overall_eye_color"] = hs.utils.quantized_normal(0,0.2,n_quantized=7)+color if not "_overall_eye_color" in default_params else default_params["_overall_eye_color"]
			out["eye_color"] = np.mod(hs.utils.quantized_uniform(-0.1, 0.1,size=4,n_quantized=7)+out["_overall_eye_color"], 1) if not "eye_color" in default_params else default_params["eye_color"]
			return out
		parameters.add_parameters("eyes", eyes, ["head"])

		def view():
			np.random.seed()
			out = {}
			floor, wall = hs.utils.quantized_uniform(*boxhead.parameters["bg_color"][0], # [floor, wall]
				size=boxhead.parameters["bg_color"][1], n_quantized=10)
			out["_floor_color"] = floor if not "_floor_color" in default_params else default_params["_floor_color"]
			out["_wall_color"] = wall if not "_wall_color" in default_params else default_params["_wall_color"]
			out["bg_color"] = np.asarray([out["_floor_color"],out["_wall_color"]])
			out["azimuth"] = hs.utils.quantized_uniform(*boxhead.parameters["azimuth"][0],
				size=boxhead.parameters["azimuth"][1],n_quantized=10) if not "azimuth" in default_params else default_params["azimuth"]
			return out
		parameters.add_parameters("view", view)
		return boxhead, parameters
