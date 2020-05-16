from utils.tf_custom.architectures.variational_autoencoder import BetaTCVAE
from utils.other_library_tools.disentanglementlib_tools import gaussian_log_density, total_correlation 
from utils.tf_custom.loss import kl_divergence_with_normal, kl_divergence_between_gaussians
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # used to silence mask warning not being trained
import utils as ut
import numpy as np
import pprint
import dill as pickle
import copy
from utilities import Mask
from train import DualTrainer
import config as cfg
import shutil
import time
import tensorflow as tf
#import pickle

class ModelHandler:
	def __init__(self, base_path, config=None, train_new=False, load_model=True, config_processing=None, initialize_only=False):
		self.base_path = base_path
		self.config_path = os.path.join(self.base_path, "config")
		self._config_processing = config_processing
		self.model = None
		self.is_load_model = load_model
		self.is_train_new = train_new
		# load past config
		if config is None:
			self.load()
		else:
			self._config = config

		if not initialize_only:
			self.activate_paths_and_model()

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

	def create_model(self, load_prev):
		config = self.config
		tf.random.set_seed(config.random_seed)
		self.model = config.get_model(
			beta=config.beta_value, 
			num_latents=config.num_latents, 
			num_channels=config.num_channels)
		if load_prev and os.path.exists(self.model_save_file):
			print("found existing model weights. Loading...")
			self.model.load_weights(self.model_save_file)
			print("Done")

	def _configure_train(self, **kwargs):
		config = self.config

		# training
		dataset = ut.dataset.DatasetBatch(config.dataset, config.batch_size).get_next 

		# define parameters
		training_object = config.TrainVAE(
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
		return training_object

	def train(self, measure_time=False):
		config = self.config

		# define training configureation
		training_object = self._configure_train()

		#run training
		training_object(model_save_steps=config.model_save_steps, 
				total_steps=config.total_steps, measure_time=measure_time)
		print("finished beta %d"%config.beta_value)

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
		assert os.path.exists(self.config_path), "specify a config, or use base path with previous config"
		
		with open(self.config_path, "rb") as f:
			self._config, self._config_processing = pickle.load(f)
	@property
	def config(self):
		return self.get_config()

	def get_config(self):
		if not self._config_processing is None:
			return self._config_processing(copy.deepcopy(self._config), self.base_path)
		else:
			return self._config	

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

class DualModelHandler():

	"""Will handle training. For other ModelHandler options, individually call comp_mh, or mask_mh 
	
	Attributes:
	    comp_mh (ModelHandler class): MaskVAE ModelHandler object
	    mask_mh (ModelHandler class): ComponentVAE ModelHandler object
	"""
	
	def __init__(self, base_path, mask_config=None, comp_config=None, train_new=True, load_model=True, randomize_mask_step=False, is_combine_models=True):
		# get the paths for the models
		paths_obj = ut.general_tools.StandardizedPaths(base_path)
		self.mask_base_path = paths_obj("mask_model", description="MaskVAE model base path")
		self.comp_base_path = paths_obj("comp_model", description="ComponentVAE model path")
		self._mask_config = mask_config
		self._comp_config = comp_config
		self.train_new = train_new
		self.load_model = load_model
		self.randomize_mask_step = randomize_mask_step

		self.mask_mh = ModelHandler(base_path=self.mask_base_path, 
				config=mask_config, train_new=self.train_new, load_model=self.load_model)
		self.comp_mh = ModelHandler(base_path=self.comp_base_path, 
				config=comp_config, train_new=self.train_new, load_model=self.load_model)
	
		if is_combine_models:
			self.setup_combination_model()

	@property
	def mask_config(self):
		if self.mask_mh is None:
			return self._mask_config
		else:
			return self.mask_mh.config	

	@property
	def comp_config(self):
		if self.comp_mh is None:
			return self._comp_config
		else:
			return self.comp_mh.config	

	def setup_combination_model(self):
		# setup mask and config processing for combination
		def mask_config_processing(config_obj, base_path):
			return cfg.make_mask_config(config_obj)
		self.mask_mh._config_processing = mask_config_processing
		self.mask_mh.activate_paths_and_model()
		# setup comp ModelHandler and config processing

		mask_obj = Mask(self.mask_mh.model, self.comp_config.mask_latent_of_focus)
		def comp_config_processing(config_obj, base_path):
			#base path is the base path for compvae, will be processed to be relative to mask
			return cfg.make_comp_config(config_obj, mask_obj, randomize_mask_step=self.randomize_mask_step)
		self.comp_mh._config_processing = comp_config_processing
		self.comp_mh.activate_paths_and_model()

	def _configure_train(self):
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
		training_object = DualTrainer(mask_train_obj, comp_train_obj, dataset, inputs_test)
		return training_object
	
	def train(self, measure_time=False):
		# define training configureation
		training_object = self._configure_train()

		#run training
		training_object(model_save_steps=self.mask_config.model_save_steps, 
				total_steps=self.mask_config.total_steps, measure_time=measure_time)
		print("finished")

	def save(self):
		self.mask_mh.save(save_config_processing=False)
		self.comp_mh.save(save_config_processing=False)


def randomize_mask_step_experiment():
	# use pretrained mask
	time.sleep(3600*3)
	mask_model_path = "exp/mask_only/beta_30"
	mask_config = cfg.Config64()
	mask_config.beta_value = 30
	mask_handler = ModelHandler(mask_model_path, mask_config)


	base_path = "exp/train_both/randomized_mask_step"
	# use mask config above
	#mask_config = cfg.Config64()
	#mask_config.beta_value = 30
	comp_config = cfg.Config256()
	comp_config.beta_value = 500
	comp_config.mask_latent_of_focus = 3
	model_handler = DualModelHandler(base_path, mask_config, comp_config, randomize_mask_step=True, is_combine_models=False)
	
	# copy the previously trained mask file
	shutil.copy(mask_handler.model_save_file, model_handler.mask_mh.model_save_file)
	
	# activate the models, save the model configurations, and start training.
	model_handler.setup_combination_model()
	model_handler.save()
	model_handler.train()

def pretrain_experiment():
	# we train it here so we keep these parameters
	mask_model_path = "exp/mask_only/beta_30"
	mask_config = cfg.Config64()
	mask_config.beta_value = 30
	mask_handler = ModelHandler(mask_model_path, mask_config)
	#mask_handler.save()
	#mask_handler.train()

	base_path = "exp/test"
	# use mask config above
	#mask_config = cfg.Config64()
	#mask_config.beta_value = 30
	comp_config = cfg.Config256()
	comp_config.beta_value = 500
	comp_config.mask_latent_of_focus = 8#int(input("please enter latent of focus: ")) # pause for analysis


	# run model without combining (this will stop the activation of model). Model activation include loading previous models.
	# this will only create directories.
	mask_config.is_train=False
	model_handler = DualModelHandler(base_path, mask_config, comp_config, train_new=False, is_combine_models=False)
	
	# copy the previously trained mask file
	shutil.copy(mask_handler.model_save_file, model_handler.mask_mh.model_save_file)

	# activate the models, save the model configurations, and start training.
	model_handler.setup_combination_model()
	model_handler.save()
	model_handler.train()

def run_masks(beta=30, random_seed=1):
	# we train it here so we keep these parameters
	mask_model_path = "exp/mask_only/beta_%d_seed_%d"%(beta, random_seed)
	print("running %s"%mask_model_path)
	mask_config = cfg.Config64()
	mask_config.beta_value = beta
	mask_config.random_seed = random_seed
	mask_handler = ModelHandler(mask_model_path, mask_config)
	mask_handler.save()
	mask_handler.train()

import multiprocessing
if __name__ == '__main__':

	betas = [20,25,30]
	random_seed = [1,10,25,30]
	
	args = [(b,r) for b in betas for r in random_seed]
	with multiprocessing.Pool(5) as pool:
		pool.starmap(run_masks, args)

	#randomize_mask_step_experiment()