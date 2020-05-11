from utils.tf_custom.architectures.variational_autoencoder import VariationalAutoencoder
from utils.other_library_tools.disentanglementlib_tools import gaussian_log_density, total_correlation 
from utils.tf_custom.loss import kl_divergence_with_normal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # used to silence mask warning not being trained
import utils as ut
import numpy as np
import pprint
import dill as pickle
import copy
#import pickle

class ModelHandler:
	def __init__(self, base_path, config=None, train_new=False, load_model=True):
		self.base_path = base_path
		self.config_path = os.path.join(self.base_path, "config")
		self._config_processing = None
		self.model = None
		# load past config
		if config is None:
			self.load()
		else:
			self._config = config

		self.get_paths(base_path=base_path)
		if load_model:
			self.create_model(load_prev= not train_new)

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
		self.model = config.get_model(
			beta=config.beta_value, 
			num_latents=config.num_latents, 
			num_channels=config.num_channels)
		if load_prev and os.path.exists(self.model_save_file):
			self.model.load_weights(self.model_save_file)

	def train(self, measure_time=False):
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
			)

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
		if not self._config_processing is None:
			return self._config_processing(copy.deepcopy(self._config), self.base_path)
		else:
			return self._config	

	def save(self, config_processing=None):
		"""
		Saves the imported config file and model summary
		"""
		# write parameters
		mpstr=pprint.pformat(self.model_parameters, width=100)
		with open(self.model_parameters_path, "w") as f:
			f.write(mpstr)

		# save configs
		if self._config_processing is None:
			self._config_processing = config_processing
		with open(self.config_path, "wb") as f:
			pickle.dump([self._config, self._config_processing], f)


class CompVAE(VariationalAutoencoder):
	def __init__(self, beta, name="BetaTCVAE", **kwargs):
		super().__init__(name=name, **kwargs)
		self.create_encoder_decoder_512() # use the larger model
		self.beta = beta

	def call(self, inputs, m_sampled, m_logvar, m_mean):
		sample, mean, logvar = self.encoder(inputs)
		reconstruction = self.decoder(sample)
		self.add_loss(self.regularizer(sample, mean, logvar, 
				m_sampled, m_logvar, m_mean))
		return reconstruction

	def regularizer(self, sample, mean, logvar, m_sampled, m_logvar, m_mean):
		# regularization uses disentanglementlib method
		kl_loss = kl_divergence_with_normal(mean, logvar)
		tc = (self.beta - 1) * total_correlation(sample, mean, logvar)
		return tc + kl_loss
