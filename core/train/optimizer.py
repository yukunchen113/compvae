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
from utilities.standard import image_traversal, kld_loss_reduction, ImageMSE, GPUMemoryUsageMonitor
from utilities.mask import mask_traversal
from functools import reduce


class OptimizerManager():
	def __init__(self, model, loss_func, optimizer, is_train=True):
		self.model = model
		self.loss_func = loss_func
		self.optimizer = optimizer
		self.is_train = is_train

	@property
	def latent_space(self):
		return self.model.get_latent_space()
	

	def tape_gradients(self, inputs, timer_func=None, **kwargs):
		with tf.GradientTape() as tape:
			reconstruct = self.model(inputs, **kwargs)
			if not timer_func is None: timer_func("OPTIMIZER: got reconstruction")
			
			regularization_loss = 0
			for l in self.model.losses:
				regularization_loss += kld_loss_reduction(l)
			
			# get loss
			reconstruction_loss = self.loss_func(inputs, reconstruct)
			if not timer_func is None: timer_func("OPTIMIZER: got reconstruct loss")

			loss = reconstruction_loss+regularization_loss 

		self.reconstruction_loss = reconstruction_loss
		self.regularization_loss = regularization_loss
		return tape, loss

	def run_optimizer(self, tape, loss):
		self.grads = tape.gradient(loss, self.model.trainable_weights)
		if self.is_train:
			self.optimizer.apply_gradients(zip(self.grads, self.model.trainable_weights))



class CondOptimizerManager(OptimizerManager):
	def tape_gradients(self, inputs, cond_logvar, cond_mean, gamma, latent_to_condition, timer_func=None):
		"""Cond optimizer for the Cond model.
		"""
		with tf.GradientTape() as tape:
			reconstruct = self.model(
					inputs = inputs, 
					cond_logvar = cond_logvar, 
					cond_mean = cond_mean, 
					gamma = gamma, 
					latent_to_condition = latent_to_condition,
					)

			# get loss
			reconstruction_loss = self.loss_func(inputs, reconstruct)

			regularization_loss = 0
			for l in self.model.losses:
				regularization_loss += kld_loss_reduction(l)

			loss = reconstruction_loss+regularization_loss 
		self.reconstruction_loss = reconstruction_loss
		self.regularization_loss = regularization_loss
		return tape, loss


