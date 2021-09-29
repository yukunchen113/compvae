"""
# TBD: make input test size parameterizable
"""
import tensorflow as tf
import disentangle as dt 
import numpy as np
import tensorflow_datasets as tfds
import hiershapes as hs
import os
import time
import h5py
import pickle
import random

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
	def __init__(self, use_server=True, use_pool=True, run_once=False, use_preloaded=False, num_proc=20, prefetch=20, pool_size=10, port=None):
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
		self.use_preloaded=use_preloaded
		self.dataset_manager = self.start_server()

	def test(self, batch_size=None, return_labels=False):
		if self.test_labels is None or not batch_size is None:
			if batch_size is None: batch_size = 64
			if not self.use_preloaded:
				self.test_images,self.test_labels = self.dataset_manager.get_batch(batch_size)
			else:
				self.test_images,self.test_labels = self.dataset_manager.load([0,0])
		else:
			if self.use_preloaded:
				self.test_images,_ = self.dataset_manager.load([0,0])
			else:
				self.test_images = self.dataset_manager.get_images(self.test_labels)
		if return_labels: return self.test_images, self.test_labels
		return self.test_images
	
	def preprocess(self, inputs, *ar, **kw):
		return tf.image.convert_image_dtype(inputs, tf.float32)
	
	def start_server(self):
		if self.use_preloaded: return HDF5Dataset(self.folder)
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
		if self.use_preloaded: return
		self.dataset_manager.close()
		if is_terminate_server and not self.server is None: self.server.close() 

	def __getstate__(self):
		d = dict(self.__dict__)
		d["test_images"] = None
		d["server"] = None
		return d

class HierShapesBoxheadSimple2(HierShapesBase):
	folder = "dataset/boxheadsimple2"
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
	folder = "dataset/boxheadsimple"
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
	folder = "dataset/boxhead_07"
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

class HDF5Dataset:
	def __init__(self, folder, capacity=5000, is_cycle=True, get_random=True):
		"""
		Writing: To indicate you are finished writing data, you must call close().   

		This dataset object will pick off where was left off from before, accounting for sudden stops.
		If there is any datasets that were not finished, there might be corruptions. 
		To avoid this, this dataset sets up a system to delete any unfinished datasets and start from there.

		dset: hdf5 files
	
		"""
		if not os.path.exists(folder): os.makedirs(folder)
		self.folder = folder
		self.capacity = capacity
		self.load_index = [0,0]
		self.current_file = None
		self.current_dataset = None
		self.get_random = get_random
		self.is_cycle = is_cycle
		self.batch_size = None

	def clean(self):
		"""
		This method checks for and cleans previous unfinished dataset if unfinished dataset exists.
		"""
		for filename in os.listdir(self.folder): 
			if filename.startswith("dataset") and filename.endswith(".hdf5") and "tmp" in filename: 
				os.remove(os.path.join(self.folder,filename))

	def get_latest_dset(self):
		"""
		This method gets the latest hdf5 file

		if the latest file is at capacity and has tmp, remove tmp from prev filename

		will get new tmp name
		"""
		# find latest file,
		if self.current_file is None:
			self.clean()
			files = [(int(i.split("_")[1]), os.path.join(self.folder, i)) for i in os.listdir(self.folder) if i.startswith("dataset")]
			if not files: 
				self.current_file = os.path.join(self.folder, f"dataset_0_tmp.hdf5")
			else:
				self.current_file = sorted(files)[-1][1]

		# get the capacity (return if not at capacity)
		if not os.path.exists(self.current_file): return self.current_file, 0
		with h5py.File(self.current_file, "r") as storage:
			curcap = len(storage.keys())
			is_past_capacity = curcap>=self.capacity
		if not is_past_capacity: return self.current_file, curcap
		
		# remove tmp with self.close() if at capacity and has tmp
		self.close()
		filenum = int(os.path.basename(self.current_file).split("_")[1])+1
		self.current_file = os.path.join(self.folder, f"dataset_{filenum}_tmp.hdf5")
		return self.current_file, 0

	def save(self, data, labels):
		"""saves the data as latest group in hdf5 dataset. 
		"""
		dsetfile, dsetnum = self.get_latest_dset()
		with h5py.File(dsetfile, "a") as storage:
			grp = storage.create_group(str(dsetnum))
			grp.create_dataset("labels", data=np.asarray(pickle.dumps(labels)))
			grp.create_dataset("data", data=data, compression="gzip")
		dsetfile, dsetnum = self.get_latest_dset()

	def load(self, load_index, get_random=False):
		"""
		loads data from file and group using load_index 

		return data if found, None if data doesn't exist at load index  
		"""
		filenum, dsetnum = load_index
		
		if get_random:
			np.random.seed()
			filenum = np.random.choice([int(i.split("_")[1]) for i in os.listdir(self.folder) if i.startswith("dataset") and not "tmp" in i])
		filepath = os.path.join(self.folder, f"dataset_{filenum}_.hdf5")
		if not os.path.exists(filepath): return None

		# return None if no data is found
		with h5py.File(filepath, "r") as storage:
			if get_random: dsetnum = np.random.choice(list(storage.keys()))
			if not str(dsetnum) in storage: return None
			group = storage[str(dsetnum)]
			data, labels = group["data"][()], pickle.loads(group["labels"][()].item())
		return data, labels

	def increment_load(self):
		"""
		loads data from file and group using load_index 

		if load_index is specified, the use that, otherwise, use load_index state

		if dataset is finished - no files/groups left, returns None
		"""
		data = self.load(self.load_index, self.get_random)
		if data is None:
			self.load_index[0]+=1
			self.load_index[1]=0
			data = self.load(self.load_index)
		if self.is_cycle and data is None: 
			self.set_load_index()
			data = self.load(self.load_index)
		self.load_index[1]+=1
		return data

	def __iter__(self):
		return self

	def __next__(self):
		data = list(self.increment_load())
		assert self.batch_size is None or self.batch_size<=len(data[0]), f"batch_size is too big, max batch size is {len(data)}"
		if not self.batch_size is None: 
			data[0] = data[0][:self.batch_size]
			data[1] = data[1][:self.batch_size]
		return data

	def batch(self, batch_size):
		self.batch_size = batch_size
		return self

	def set_load_index(self, load_index=None):
		"""sets load index, resets if nothing is specified
		"""
		if load_index is None: load_index = [0,0]
		self.load_index = load_index

	def close(self):
		"""
		if there is a tmp on current dataset, will remove the tmp
		"""
		if not self.current_file is None and self.current_file.endswith("_tmp.hdf5"): os.rename(self.current_file, self.current_file.replace("_tmp.hdf5", "_.hdf5"))
