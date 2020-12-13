import tensorflow as tf
tf.keras.backend.set_floatx('float32')
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # used to silence mask warning not being trained
import numpy as np
from hsr.utils.loss import kld_loss_reduction
from hsr.utils.regular import GPUMemoryUsageMonitor


class Trainer():
	def __init__(self, model, loss_func, optimizer):
		self.model = model
		self.loss_func = loss_func
		self.optimizer = optimizer
	
	def tape_gradients(self, inputs, **kwargs):
		with tf.GradientTape() as tape:
			reconstruct = self.model(inputs, **kwargs)
			
			regularization_loss = 0
			for l in self.model.losses:
				regularization_loss += kld_loss_reduction(l)
			
			# get loss
			reconstruction_loss = self.loss_func(inputs, reconstruct)

			loss = reconstruction_loss+regularization_loss 

		self.reconstruction_loss = reconstruction_loss
		self.regularization_loss = regularization_loss
		self.reconstruct = reconstruct
		return tape, loss

	def run_optimizer(self, tape, loss):
		self.grads = tape.gradient(loss, self.model.trainable_weights)
		self.optimizer.apply_gradients(zip(self.grads, self.model.trainable_weights))
	
	def __call__(self, step, inputs, hparams={}):
		"""Trains model on inputs for a given step, breaks if nan value in loss and returns nan, normal operation will return step

		increments step in the beginning
		"""
		step+=1

		# apply gradient tape
		tape, loss = self.tape_gradients(inputs, **hparams)
		
		# check nan in loss
		if np.isnan(loss.numpy()):
			string="Nan Loss on step %s:\t rec loss = %s\t, reg loss = %s\t" % (
				step, 
				self.reconstruction_loss.numpy(),
				self.regularization_loss.numpy(),
				)
			return np.nan

		# optimize
		self.run_optimizer(tape, loss)

		print('\033[Kstep %s:\t rec loss = %s\t, reg loss = %s\t' % (
			step, 
			self.reconstruction_loss.numpy(),
			self.regularization_loss.numpy(),
			), "\r", end="")

		return step