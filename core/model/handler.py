from utils.other_library_tools.disentanglementlib_tools import gaussian_log_density, total_correlation 
from utils.tf_custom.loss import kl_divergence_with_normal, kl_divergence_between_gaussians
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # used to silence mask warning not being trained
import utils as ut
import numpy as np
import pprint
import dill as pickle
import copy
from utilities.mask import Mask
from core.train.manager import DualTrainer
import core.config as cfg
import shutil
import time
import tensorflow as tf
#import pickle

class ModelHandler:
	def __init__(self, base_path, config=None, train_new=False, 
		load_model=True, config_processing=None, initialize_only=False, 
		overwrite_config=False, overwrite_config_processing=False):

		self.base_path = base_path
		self.config_path = os.path.join(self.base_path, "config")
		self.model = None
		self.is_load_model = load_model
		self.is_train_new = train_new
		self._config_processing = config_processing

		# load past config
		if (not self.load()) or overwrite_config:
			assert not config is None, "Must specify config, as previous is not found"
			self._config = config

		if overwrite_config_processing:
			self._config_processing = config_processing

		if not initialize_only:
			self.activate_paths_and_model()

		# dynamically changed attributes
		self.training_object = None

	def activate_paths_and_model(self):
		# start modifying files
		self.get_paths(base_path=self.base_path)
		if self.is_load_model:
			self.create_model(load_prev= not self.is_train_new)

	def get_paths(self, base_path):
		config = self.config

		# define paths and create main directory
		paths_obj = ut.general_tools.StandardizedPaths(self.base_path)

		image_dir_name = "image_dir" # these are the names
		model_setup_dir_name = "model_setup_dir" # these are the names
		
		# define model paths 
		paths_obj.add_path(image_dir_name, config.image_dir,
			"images saved across training. Used for debugging and as a quick check to see network performance")
		paths_obj.add_path(model_setup_dir_name, config.model_setup_dir,
			"where model files are saved, with related model information. Includes config files, model weights, model summary, etc.")
		
		#create model paths
		self.image_dir = paths_obj.get_path(image_dir_name)
		self.model_setup_dir = paths_obj.get_path(model_setup_dir_name)

		# create save files
		self.model_save_file = os.path.join(self.model_setup_dir, config.model_save_file) # model weights
		self.model_parameters_path = os.path.join(self.base_path, config.model_parameters_path) # model parameters path

	def create_model(self, load_prev=True):
		config = self.config
		tf.random.set_seed(config.random_seed)
		self.model = config.get_model(
			beta=config.beta, 
			num_latents=config.num_latents)
		
		#self.model.save_weights(self.model_save_file)

		if load_prev and os.path.exists(self.model_save_file):
			print("found existing model weights. Loading...")
			self.model.load_weights(self.model_save_file)
			print("Done")

	def _configure_train(self, **kwargs):
		if not self.training_object is None:
			return self.training_object
		
		config = self.config

		# training
		dataset = config.dataset_manager.batch(config.batch_size)

		# define parameters
		self.training_object = config.TrainVAE(
			model = self.model,
			dataset = dataset,
			inputs_test = config.inputs_test, # validation set
			preprocessing = config.preprocessing,
			image_dir = self.image_dir,
			model_setup_dir = self.model_setup_dir,
			model_save_file = self.model_save_file,
			loss_func = config.loss_func,
			optimizer = config.optimizer,
			approve_run = config.approve_run,
			is_train=config.is_train,
			**kwargs
			)
		return self.training_object
		
	def train(self):
		config = self.config
		train_status_path = os.path.join(self.base_path, config.train_status_path) # model parameters path

		# define training configureation
		self._configure_train()

		#run training
		# custom training loop for monitoring during training and saving of step for training resume 
		if (not self.is_train_new) and os.path.exists(train_status_path):
			train_status = np.load(train_status_path)
			step = train_status["step"]
		else:		
			step = -1 # the previous step.

		for data in config.dataset_manager.batch(config.batch_size):
			if np.isnan(step):
				break
			step = self.training_object.train_step(
				step=step, model_save_steps=config.model_save_steps, 
				total_steps=config.total_steps,
				custom_inputs=data[0])

			if np.isnan(step) or not (step%config.model_save_steps): 
				np.savez(train_status_path, step=step)
				self.save()

		print("finished beta",config.beta)


	def train_stats(self):
		self._configure_train()
		self.training_object(model_save_steps=self.config.model_save_steps, 
				total_steps=self.config.total_steps, measure_time=True)

	def inference(self, inputs):
		config = self.config
		inputs = config.preprocessing(inputs)
		return self.model(inputs)

	@property
	def model_parameters(self):
		if self.model is None:
			return "Model Not Initialized"
		return self.model.get_config() 

	def load(self):
		if not os.path.exists(self.config_path):
			print("Prev config not found...")
			return False
		with open(self.config_path, "rb") as f:
			self._config, self._config_processing = pickle.load(f)
		print("Prev config found...")
		#import core.config.config as cfg
		#print([i for i in cfg.Config64().__dict__.keys() if not i in self._config.__dict__.keys()])
		return True

	@property
	def config(self):
		return self.get_config()

	def get_config(self):
		if not self._config_processing is None:
			config = self._config_processing(copy.deepcopy(self._config))
		else:
			config = self._config
		config = self.config_connect(config)# subclass overwritten function
		return config

	def config_connect(self, config):
		return config

	def save(self, config_processing=None, save_config_processing=True):
		"""
		Saves the imported config file and model summary
		"""
		# write parameters
		mpstr=pprint.pformat(self.model_parameters, width=100)
		with open(self.model_parameters_path, "w") as f:
			f.write(mpstr)
		# save configs
		if self._config_processing is None:
			process_save = config_processing
		else:
			process_save = self._config_processing 
		if not save_config_processing:
			process_save = None
		with open(self.config_path, "wb") as f:
			pickle.dump([self._config, process_save], f)


class ProVLAEModelHandler(ModelHandler):
	def __init__(self, *args, config_processing=None, **kwargs):
		if config_processing is None:
			config_processing = cfg.addition.make_vlae_large
		super().__init__(*args, config_processing=config_processing, **kwargs)

	def create_model(self, load_prev=True):
		config = self.config
		tf.random.set_seed(config.random_seed)
		self.model = config.get_model(
			beta=config.beta,
			latent_connections = config.latent_connections,
			gamma = config.gamma,
			num_latents=config.num_latents)

		if load_prev and os.path.exists(self.model_save_file):
			print("found existing model weights. Loading...")
			self.model.load_weights(self.model_save_file)
			print("Done")

	def config_connect(self,config):
		config.hparam_schedule = self._config.hparam_schedule
		return config

	def _configure_train(self, *args, hparam_schedule=None, **kwargs):
		if hparam_schedule is None:
			hparam_schedule = self.config.hparam_schedule
		return super()._configure_train(*args, hparam_schedule=hparam_schedule, **kwargs)
	


class DualModelHandler():

	"""Will handle training. For other ModelHandler options, individually call comp_mh, or mask_mh 
	
	For training parameters such as number of steps, train_status_path (basename only), or save model steps, will use mask model 

	Attributes:
	    comp_mh (ModelHandler class): MaskVAE ModelHandler object
	    mask_mh (ModelHandler class): ComponentVAE ModelHandler object
	"""
	
	def __init__(self, base_path, mask_config=None, comp_config=None, train_new=False, 
		load_model=True, randomize_mask_step=False, is_combine_models=True):
		"""Initializes dual model handling. Setup paths, related models, and DualTrainer object. 

		This class is abstracted from model flow, which is the job of the DualTrainer object 
		
		Also handles save and loading of models.

		Args:
		    base_path (string): path to store model data, training.
		    mask_config (config object, None, optional): config object for masking network, use None only for loading if config is in mask model basepath
		    comp_config (config object, None, optional): config object for comp network, use None only for loading if config is in comp model basepath. If specfied, make sure that mask_latent_of_focus is specified as an attribute
		    train_new (bool, optional): if previous trained weights are found, they will be loaded if this is True
		    load_model (bool, optional): will create the tensorflow model if this is True
		    randomize_mask_step (bool, optional): Whether to randomize the mask step size for not. If this is True, will randomie. This is used for mask config additions
		    is_combine_models (bool, optional): If this is true, will immediately combine the models, otherwise user will need to call setup_combination_model() to combine the models
		"""
		# get the paths for the models
		self.base_path = base_path
		paths_obj = ut.general_tools.StandardizedPaths(self.base_path)
		self.mask_base_path = paths_obj("mask_model", description="MaskVAE model base path")
		self.comp_base_path = paths_obj("comp_model", description="ComponentVAE model path")

		# create configs
		self._mask_config = mask_config
		self._comp_config = comp_config
		self.train_new = train_new
		self.load_model = load_model
		self.randomize_mask_step = randomize_mask_step

		self.mask_mh = ModelHandler(base_path=self.mask_base_path, 
				config=mask_config, train_new=self.train_new, load_model=False) # load model is false so we only activate paths
		self.comp_mh = ModelHandler(base_path=self.comp_base_path, 
				config=comp_config, train_new=self.train_new, load_model=False)
	
		self.model_save_steps = self.mask_config.model_save_steps
		self.total_steps = self.mask_config.total_steps

		if is_combine_models:
			self.setup_combination_model()

		# dynamically changed attributes
		self.training_object = None

	@property
	def mask_config(self):
		if self.mask_mh is None:
			return self._mask_config
		else:
			return self.mask_mh.config	

	@property
	def comp_config(self):
		if self.comp_mh is None:
			config = self._comp_config
		else:
			config = self.comp_mh.config	
		assert "mask_latent_of_focus" in config.__dict__.keys()
		return config

	def setup_combination_model(self):
		# setup mask and config processing for combination
		def mask_config_processing(config_obj, base_path):
			return cfg.addition.make_mask_config(config_obj)
		self.mask_mh._config_processing = mask_config_processing
		self.mask_mh.is_load_model = self.load_model # approve load model here, as we only allowed paths before
		self.mask_mh.activate_paths_and_model()
		# setup comp ModelHandler and config processing

		mask_obj = Mask(self.mask_mh.model, self.comp_config.mask_latent_of_focus)
		def comp_config_processing(config_obj, base_path):
			#base path is the base path for compvae, will be processed to be relative to mask
			return cfg.addition.make_comp_config(config_obj, mask_obj, randomize_mask_step=self.randomize_mask_step)
		self.comp_mh._config_processing = comp_config_processing
		self.comp_mh.is_load_model = self.load_model # approve load model here, as we only allowed paths before
		self.comp_mh.activate_paths_and_model()

	def _configure_train(self):
		if not self.training_object is None:
			return self.training_object

		mask_train_obj = self.mask_mh._configure_train(hparam_schedule=self.mask_config.hparam_schedule)
		comp_train_obj = self.comp_mh._configure_train()

		comp_train_obj.mask_latent_of_focus = self.comp_config.mask_latent_of_focus

		# select the larger dataset
		if self.mask_mh.model.shape_input[1] < self.comp_mh.model.shape_input[1]:
			dataset = comp_train_obj.dataset
			inputs_test = comp_train_obj.inputs_test
		else:
			dataset = mask_train_obj.dataset
			inputs_test = mask_train_obj.inputs_test

		# use dual training, where some parameters are shared
		self.training_object = DualTrainer(mask_train_obj, comp_train_obj, dataset, inputs_test)
		return self.training_object
	
	def train(self):
		# get train paths
		train_status_path = os.path.join(self.base_path, self.mask_config.train_status_path)
		
		# define training configureation
		self._configure_train()

		#run training
		if (not self.train_new) and (not train_status_path is None) and os.path.exists(train_status_path):
			train_status = np.load(train_status_path)
			step = train_status["step"]
		else:
			step = -1 # the previous step.
		while 1: # set this using validation
			if np.isnan(step):
				break
			step = self.training_object.train_step(
				step=step, model_save_steps=self.model_save_steps, 
				total_steps=self.total_steps)
			if np.isnan(step) or not (step%self.model_save_steps): 
				np.savez(train_status_path, step=step)

		print("Finished DualModel")

	def train_stats(self):
		self._configure_train()
		self.training_object(model_save_steps=self.model_save_steps, 
				total_steps=self.total_steps, measure_time=True)


	def save(self):
		self.mask_mh.save(save_config_processing=False)
		self.comp_mh.save(save_config_processing=False)
