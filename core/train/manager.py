import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # used to silence mask warning not being trained
import matplotlib.pyplot as plt
from utils import general_tools as gt 
import utils as ut
import numpy as np
import time
import shutil
from utilities.standard import image_traversal, kld_loss_reduction, ImageMSE, GPUMemoryUsageMonitor, TrainObjMetaClass
from utilities.mask import mask_traversal
from utilities.vlae_method import vlae_traversal
from functools import reduce
from .optimizer import OptimizerManager, CondOptimizerManager

class TrainObj(metaclass=TrainObjMetaClass):
	def __call__(self, model_save_steps, total_steps, measure_time=False):
		"""Trains the model.
		Steps to save the images across training and model are defined here, along with displaying curret loss and metrics 

		Total training time: 100000 steps, TBD: if want wvalidation switch this. 
		"""	
		step = -1
		if measure_time: 
			timer_func = ut.general_tools.Timer()
		else:
			timer_func = None
		while 1: # set this using validation
			if np.isnan(step):
				break
			step = self.train_step(
				step=step, model_save_steps=model_save_steps, 
				total_steps=total_steps, timer_func=timer_func)
	
	def train_step(self, *args, **kwargs):
		pass


class TrainVAE(TrainObj):
	def __init__(self, model, dataset, inputs_test, preprocessing, image_dir, model_setup_dir, optimizer, loss_func, model_save_file, approve_run=False, hparam_schedule=None, is_train=True):
		"""Creates a training object for your model. See code for defaults.
		
		Args:
			model (tf.keras.Models): VAE model from yukun's library.
			dataset (ut.dataset object): dataset object
			inputs_test (np array): batch of raw images for testing new input data
			preprocessing (function): processing function on dataset inputs
			image_dir (string): where the images across training will be stored
			model_setup_dir (string): where the model setup is defined
			model_save_file (string): where the model_save file is defined.
			approve_run (bool, optional): bool, will automatically overwrite models
		"""
		# setup constants
		self.image_dir = image_dir
		self.model_setup_dir = model_setup_dir
		self.model_save_file = model_save_file
		# setup datasets
		self.dataset = dataset
		self.inputs_test = inputs_test
		self._preprocessing = preprocessing

		# setup model
		self.model = model

		# create save paths
		self.make_new_save_dir(approve_run)

		# training optimizer config
		self.opt_man = OptimizerManager(model, loss_func, optimizer, is_train=is_train)
		self.hparam_schedule = hparam_schedule

		# dynamically changing parameters
		self.reconstruction_loss = None
		self.regularization_loss = None

		# image visualize items
		self.image_visualize = [1,5]

	def make_new_save_dir(self, approve_run):
		"""Creates a new save directory
		
		Args:
			approve_run (bool): This specifies if the models should be overwritten if they exist 
		"""
		model_dirs = [self.image_dir, self.model_setup_dir]
		# load large data: (below is modeled off tensorflow website)
		if reduce(lambda x,y: os.path.exists(x) or os.path.exists(y), model_dirs) and not approve_run:
			while 1:
				answer = input("Are you sure you want to overwrite the files? [yes/no]")
				if answer == "yes":
					break
				elif answer == "no":
					exit()
				else:
					print("Invalid answer \"%s\" please enter yes, or no"%answer)


		for i in model_dirs:
			if os.path.exists(i):
				shutil.rmtree(i)
			os.mkdir(i)

	def preprocess(self, inputs=None, **kwargs):
		if inputs is None: inputs, _ = self.dataset()
		return self._preprocessing(inputs)

	def save_model_weights(self):
		self.model.save_weights(self.model_save_file)

	def save_image(self, step):
		t_im = mask_traversal(
			self.model,
			self.preprocess(inputs=self.inputs_test[self.image_visualize]),
			min_value=-3, 
			max_value=3, 
			num_steps=30, 
			return_traversal_object=True)
		t_im.save_gif(os.path.join(self.image_dir, "%d.gif"%step))

	@staticmethod
	def save_image_step(step):
		steps = [500, 1000, 2500, 5000]#[1,2,3,5,7,10,15,20,30,40,75,100,200,300,500,700,1000,1500,2500]
		return step in steps or step%5000 == 0
	
	@staticmethod
	def print_step(step):
		return step%500 == 0

	def train_step(self, step, model_save_steps, total_steps, timer_func=None):
		step+=1
		inputs = self.preprocess(None, measure_time=not timer_func is None)

		# apply gradient tape
		hparams = {}
		if not self.hparam_schedule is None:
			hparams = self.hparam_schedule(step)
		tape, loss = self.opt_man.tape_gradients(inputs, **hparams)
		if not timer_func is None: timer_func("taped gradients")
	
		if np.isnan(loss.numpy()):
			return np.nan

		self.opt_man.run_optimizer(tape, loss)
		if not timer_func is None: timer_func("applied gradients")


		print("step %d\r"%step, end="")
		if self.print_step(step):
			print('training step %s:\t rec loss = %s\t, reg loss = %s\t' % (
				step, 
				self.opt_man.reconstruction_loss.numpy(),
				self.opt_man.regularization_loss.numpy(),
				))

		if self.save_image_step(step):
			self.save_image(step)
			if not timer_func is None: timer_func("saved figure")

		if not step%model_save_steps:
			self.save_model_weights()
			if not timer_func is None: timer_func("saved weights")

		if step>=total_steps:
			return np.nan

		if not timer_func is None: print()
		if not timer_func is None and step>=5: exit()
		return step

class TrainProVLAE(TrainVAE):
	#def save_image(self, step):
	#	pass
	def save_image(self, step):
		t_im = vlae_traversal(
			self.model,
			self.preprocess(inputs=self.inputs_test[self.image_visualize]),
			min_value=-3, 
			max_value=3, 
			num_steps=30, 
			return_traversal_object=True)
		t_im.save_gif(os.path.join(self.image_dir, "%d.gif"%step))

	def train_step(self, *ar, **kw):
		return super().train_step(*ar, **kw)


class DualTrainer(TrainObj):
	def __init__(self, mask_train_obj, cond_train_obj, dataset, inputs_test):
		"""Creates a training object for your model. See code for defaults.
		"""
		# setup datasets
		self.dataset = dataset
		self.inputs_test = inputs_test

		self.mtr_obj = mask_train_obj
		self.mtr_obj.opt_man = CondOptimizerManager(self.mtr_obj.model, 
				self.mtr_obj.opt_man.loss_func, self.mtr_obj.opt_man.optimizer, self.mtr_obj.opt_man.is_train)
		self.ctr_obj = cond_train_obj

	def save_image(self, step):
		self.mtr_obj.save_image(step)
		self.ctr_obj.save_image(step)

	def train_step(self, step, model_save_steps, total_steps, timer_func=None):
		step+=1
		if not timer_func is None: timer_func("Start")
		# preprocess data
		raw_data,_ = self.dataset()
		if not timer_func is None: timer_func("got data")

		mask_inputs = self.mtr_obj.preprocess(raw_data, measure_time=not timer_func is None)
		if not timer_func is None: timer_func("preprocessed mask")
		comp_inputs = self.ctr_obj.preprocess(raw_data, measure_time=not timer_func is None)
		if not timer_func is None: timer_func("preprocessed comp")

		# Tape gradients. Opt_man is used to handle models for loss and optimzation during training. 
		comp_tape, comp_loss = self.ctr_obj.opt_man.tape_gradients(comp_inputs, timer_func=timer_func) # evaluate comp first
		if not timer_func is None: timer_func("taped gradients: comp")
		cond_samples, cond_logvar, cond_mean = self.ctr_obj.opt_man.latent_space
		hparams = self.mtr_obj.hparam_schedule(step)
		mask_tape, mask_loss = self.mtr_obj.opt_man.tape_gradients( # get mask gradient tape
						inputs=mask_inputs, 
						cond_logvar=cond_logvar, 
						cond_mean=cond_mean,  
						latent_to_condition=self.ctr_obj.mask_latent_of_focus
						**hparams)
		if not timer_func is None: timer_func("taped gradients: mask")

		# optimize models
		self.mtr_obj.opt_man.run_optimizer(mask_tape, mask_loss)
		self.ctr_obj.opt_man.run_optimizer(comp_tape, comp_loss)
		if not timer_func is None: timer_func("applied gradients")

		if np.isnan(mask_loss.numpy()) or np.isnan(comp_loss.numpy()):
			return np.nan

		# print training log
		print("step %d\r"%step, end="")
		if self.mtr_obj.print_step(step):
			print('training step %s:\t mask rec loss = %s\t, mask reg loss = %s\t, cond rec loss = %s\t, cond reg loss = %s\t' % (
				step, 
				self.mtr_obj.opt_man.reconstruction_loss.numpy(),
				self.mtr_obj.opt_man.regularization_loss.numpy(),
				self.ctr_obj.opt_man.reconstruction_loss.numpy(),
				self.ctr_obj.opt_man.regularization_loss.numpy(),
				))
		# save testing image
		if self.mtr_obj.save_image_step(step):
			#self.save_image(step)
			pass

		# save model weights
		if not step%model_save_steps:
			self.mtr_obj.save_model_weights()
			self.ctr_obj.save_model_weights()
		if not timer_func is None: timer_func("saved weights")

		# end training
		if step>=total_steps:
			return np.nan
		if not timer_func is None: print()
		if not timer_func is None and step>=5: exit()
		return step
