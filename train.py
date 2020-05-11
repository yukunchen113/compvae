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
from utilities import image_traversal, kld_loss_reduction, ImageMSE
from functools import reduce

class TrainVAE():
	def __init__(self, model, dataset, inputs_test, preprocessing, image_dir, model_setup_dir, optimizer, loss_func, model_save_file, approve_run=False):
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

		# training parameters
		self.loss_func = loss_func
		self.optimizer = optimizer

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

	def __call__(self, model_save_steps, total_steps, measure_time=False):
		"""Trains the model.
		Steps to save the images across training and model are defined here, along with displaying curret loss and metrics 

		Total training time: 100000 steps, TBD: if want wvalidation switch this. 
		"""
		def save_image_step(step):
			steps = [5000]#[1,2,3,5,7,10,15,20,30,40,75,100,200,300,500,700,1000,1500,2500]
			return step in steps or step%5000 == 0
		
		def print_step(step):
			return step%100 == 0
		
		step = -1
		if measure_time: timer_func = ut.general_tools.Timer()
		while 1: # set this using validation
			step+=1
			inputs = self.preprocess(None, measure_time=measure_time)

			# apply gradient tape
			with tf.GradientTape() as tape:
				reconstruct = self.model(inputs)

				# get loss
				reconstruction_loss = self.loss_func(inputs, reconstruct)
				if measure_time: timer_func("applied loss")

				regularization_loss = kld_loss_reduction(self.model.losses[0])
				loss = reconstruction_loss+regularization_loss 
				if measure_time: timer_func("created loss")


			grads = tape.gradient(loss, self.model.trainable_weights)
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
			print("step %d\r"%step, end="")
			if measure_time: timer_func("applied gradients")

			if print_step(step):
				print('training step %s:\t rec loss = %s\t, reg loss = %s\t' % (
					step, 
					reconstruction_loss.numpy(),
					regularization_loss.numpy(),
					))
			if save_image_step(step):
				t_im = image_traversal(
					self.model,
					self.preprocess(inputs=self.inputs_test[:2]),
					min_value=0, 
					max_value=3/15*10, 
					num_steps=10, is_visualizable=True, 
					return_traversal_object=False)
				if measure_time: timer_func("Traversed latents")
				plt.imshow(t_im)
				plt.savefig(
					os.path.join(self.image_dir, "%d.svg"%step),
					format="svg", dpi=700)
				if measure_time: timer_func("saved figure")

			if step%model_save_steps == 0:
				self.model.save_weights(self.model_save_file)
			if measure_time: timer_func("saved weights")

			#TBD: this should be replaced with validation
			if step>=total_steps:
				break
			if measure_time: print()
			if measure_time and step>=5: exit()
