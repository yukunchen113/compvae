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
from utilities import mask_traversal, kld_loss_reduction, ImageMSE
from functools import reduce

class OptimizerManager():
	def __init__(self, model, loss_func, optimizer, is_train=True):
		self.model = model
		self.loss_func = loss_func
		self.optimizer = optimizer
		self.is_train = is_train

	def tape_gradients(self, inputs):
		with tf.GradientTape() as tape:
			reconstruct = self.model(inputs)

			# get loss
			reconstruction_loss = self.loss_func(inputs, reconstruct)

			for l in self.model.losses:
				regularization_loss = kld_loss_reduction(l)
			loss = reconstruction_loss+regularization_loss 
		self.reconstruction_loss = reconstruction_loss
		self.regularization_loss = regularization_loss
		return tape, loss

	def run_optimizer(self, tape, loss):
		grads = tape.gradient(loss, self.model.trainable_weights)
		if self.is_train:
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


class TrainVAE():
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
			self.preprocess(inputs=self.inputs_test[1:3]),
			min_value=-3, 
			max_value=3, 
			num_steps=30, 
			return_traversal_object=True)
		t_im.save_gif(os.path.join(self.image_dir, "%d.gif"%step))
		#plt.imshow(t_im)
		#plt.savefig(
		#	os.path.join(self.image_dir, "%d.svg"%step),
		#	format="svg", dpi=700)

	@staticmethod
	def save_image_step(step):
		steps = [5000]#[1,2,3,5,7,10,15,20,30,40,75,100,200,300,500,700,1000,1500,2500]
		return step in steps or step%5000 == 0
	
	@staticmethod
	def print_step(step):
		return step%100 == 0

	def __call__(self, model_save_steps, total_steps, measure_time=False):
		"""Trains the model.
		Steps to save the images across training and model are defined here, along with displaying curret loss and metrics 

		Total training time: 100000 steps, TBD: if want wvalidation switch this. 
		"""	
		step = -1
		if measure_time: timer_func = ut.general_tools.Timer()
		while 1: # set this using validation
			step+=1
			inputs = self.preprocess(None, measure_time=measure_time)

			# apply gradient tape
			tape, loss = self.opt_man.tape_gradients(inputs)
			if measure_time: timer_func("taped gradients")

			self.opt_man.run_optimizer(tape, loss)
			if measure_time: timer_func("applied gradients")

			print("step %d\r"%step, end="")
			if self.print_step(step):
				print('training step %s:\t rec loss = %s\t, reg loss = %s\t' % (
					step, 
					self.opt_man.reconstruction_loss.numpy(),
					self.opt_man.regularization_loss.numpy(),
					))

			if self.save_image_step(step):
				self.save_image(step)
				if measure_time: timer_func("saved figure")

			if step%model_save_steps == 0:
				self.save_model_weights()

			if measure_time: timer_func("saved weights")

			#TBD: this should be replaced with validation
			if step>=total_steps:
				break
			if measure_time: print()
			if measure_time and step>=5: exit()

class CondOptimizerManager(OptimizerManager):
	def tape_gradients(self, inputs, cond_logvar, cond_mean, gamma, latent_to_condition):
		"""Cond optimizer for the Cond model.
		"""
		with tf.GradientTape() as tape:
			reconstruct = self.model(
					inputs = inputs, 
					cond_logvar = cond_logvar, 
					cond_mean = cond_mean, 
					gamma = gamma, 
					latent_to_condition = latent_to_condition)
			# get loss
			reconstruction_loss = self.loss_func(inputs, reconstruct)

			for l in self.model.losses:
				regularization_loss = kld_loss_reduction(l)
			loss = reconstruction_loss+regularization_loss 
		self.reconstruction_loss = reconstruction_loss
		self.regularization_loss = regularization_loss
		return tape, loss

class DualTrainer():
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


	def __call__(self, model_save_steps, total_steps, measure_time=False):
		"""Trains the model.
		Steps to save the images across training and model are defined here, along with displaying curret loss and metrics 

		TBD: currently only connects the comp encoder to mask reg. Should also connect 
			comp preprocessing and losses here as well for speed boost
		"""	
		step = -1
		if measure_time: timer_func = ut.general_tools.Timer()
		while 1: # set this using validation
			step+=1

			# preprocess data
			raw_data,_ = self.dataset()
			mask_inputs = self.mtr_obj.preprocess(raw_data, measure_time=measure_time)
			comp_inputs = self.ctr_obj.preprocess(raw_data, measure_time=measure_time)

			# apply gradient tape
			cond_tape, cond_loss = self.ctr_obj.opt_man.tape_gradients(comp_inputs) # evaluate comp first
			_, cond_logvar, cond_mean = self.ctr_obj.model.get_latent_space()
			gamma = self.mtr_obj.hparam_schedule(step)
			mask_tape, mask_loss = self.mtr_obj.opt_man.tape_gradients( # get mask gradient tape
							inputs=mask_inputs, 
							cond_logvar=cond_logvar, 
							cond_mean=cond_mean, 
							gamma=gamma, 
							latent_to_condition=self.ctr_obj.mask_latent_of_focus)
			if measure_time: timer_func("taped gradients")

			# optimize models
			self.mtr_obj.opt_man.run_optimizer(mask_tape, mask_loss)
			self.ctr_obj.opt_man.run_optimizer(cond_tape, cond_loss)
			if measure_time: timer_func("applied gradients")

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
				self.mtr_obj.save_image(step)
				self.ctr_obj.save_image(step)

			# save model weights
			if step%model_save_steps == 0:
				self.mtr_obj.save_model_weights()
				self.ctr_obj.save_model_weights()
			if measure_time: timer_func("saved weights")

			# end training
			if step>=total_steps:
				break
			if measure_time: print()
			if measure_time and step>=5: exit()