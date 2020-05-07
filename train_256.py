import tensorflow as tf
tf.keras.backend.set_floatx('float32')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # used to silence mask warning not being trained
import matplotlib.pyplot as plt
from utils import general_tools as gt 
import utils as ut
import numpy as np
import time
import cv2
import shutil
import sys
import commonly_used_objects_hd256 as cuo
from functools import reduce
import sys
from mask.mask import Mask

# reconstruction loss
class ImageMSE():
	def __init__(self, loss_process=lambda x:x):
		self.loss_process = loss_process

	def __call__(self, actu, pred):
		reduction_axis = range(1,len(actu.shape))

		# per point
		loss = tf.math.squared_difference(actu, pred)

		# apply processing to first 3 channels
		loss = self.loss_process(loss)

		# per sample
		loss = tf.math.reduce_sum(loss, reduction_axis)
		# per batch
		loss = tf.math.reduce_mean(loss)
		return loss

# regularization loss
def kld_loss_reduction(kld_loss):
	# per batch
	kld_loss = tf.math.reduce_mean(kld_loss)
	return kld_loss

class TrainVAE():
	def __init__(self, model, dataset, inputs_test, preprocessing, image_dir, model_setup_dir, model_save_file, approve_run=False, optimizer=None):
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
		self.model = model
		self.dataset = dataset
		self.inputs_test = inputs_test
		self.preprocessing = preprocessing

		# create model
		self.make_new_save_dir(approve_run)
		if optimizer is None:
			optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5)
		self.setup_training_specific_objects(optimizer)

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

	def setup_training_specific_objects(self, optimizer, total_metric=tf.keras.metrics.Mean()):
		"""Setup the objects specific to training, this is already called by default, by can be called again if you want to customize it.
		
		Args:
		    learning_rate (float): The learning rate for Adam optimizer.
		    beta_1 (float): beta_1 for Adam optimizer
		    total_metric (tf.keras.metrics): The metric to use to evaluate model, currently, mean
		"""
		# training setup
		self.loss_func = ImageMSE()
		self.optimizer = optimizer
		self.total_metric = total_metric

	def preprocessed_data(self, x=None, **kwargs):
		inputs, _ = self.dataset()
		return self.preprocessing(inputs)

	def __call__(self, measure_time=False):
		"""Trains the model.
		Steps to save the images across training and model are defined here, along with displaying curret loss and metrics 

		Total training time: 100000 steps, TBD: if want wvalidation switch this. 
		"""
		def save_image_step(step):
			steps = [5000]#[1,2,3,5,7,10,15,20,30,40,75,100,200,300,500,700,1000,1500,2500]
			return step in steps or step%5000 == 0

		step = -1
		if measure_time: timer_func = ut.general_tools.Timer()
		while 1: # set this using validation
			step+=1
			inputs = self.preprocessed_data(None, measure_time=measure_time)

			# apply gradient tape
			with tf.GradientTape() as tape:
				reconstruct = self.model(inputs)

				# get loss
				reconstruction_loss = self.loss_func(inputs, reconstruct)
				if measure_time: timer_func("applied mask to loss")

				regularization_loss = kld_loss_reduction(self.model.losses[0])
				loss = reconstruction_loss+regularization_loss 
				if measure_time: timer_func("created loss")


			grads = tape.gradient(loss, self.model.trainable_weights)
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
			print("step %d\r"%step, end="")
			if measure_time: timer_func("applied gradients")


			if save_image_step(step):
				print('training step %s:\t rec loss = %s\t, reg loss = %s\t' % (
					step, 
					reconstruction_loss.numpy(),
					regularization_loss.numpy(),
					))
				t_im = cuo.image_traversal(
					self.model,
					self.inputs_test[:2],
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

			if step%10000 == 0:
				self.model.save_weights(self.model_save_file)
			if measure_time: timer_func("saved weights")

			#TBD: this should be replaced with validation
			if step>=100000:
				break
			if measure_time: print()
			if measure_time and step>=5: exit()

class TrainMaskedVAE(TrainVAE):
	def __init__(self, *args, mask_obj=None, **kwargs):
		self.mask_obj = mask_obj
		super().__init__(*args, **kwargs)

	def preprocessed_data(self, x=None, measure_time=False):
		inputs = super().__init__(x)
		if measure_time: timer_func("loaded inputs")
		mask_obj(inputs, measure_time=measure_time)
		inputs = np.concatenate((inputs, mask_obj.mask), -1)
		if measure_time: timer_func("masked inputs")
		return inputs

	def setup_training_specific_objects(self, optimizer, total_metric=tf.keras.metrics.Mean()):
		super().setup_training_specific_objects(optimizer, total_metric)
		# overwrite default loss process
		def loss_process(loss):
			loss_recon = mask_obj.apply(loss[:,:,:,:3])
			loss = tf.concat((loss_recon, loss[:,:,:,3:]), -1)
			return loss
		self.loss_func = ImageMSE(loss_process=loss_process)

def main():
	"""To run, must specify beta value as commandline argument. 
	I have tried a for loop for models but there was something weird with 
	it, where the first model that was trained must be loaded first.
	"""
	# create masking object
	# TBD: This masking object right now must be defined before model, need to fix this, there seems to be an issue with load weights, when loading from hdf5 file.

	beta_value = int(sys.argv[1])
	mask_latent = 0 if not len(sys.argv) > 2 else int(sys.argv[2])
	mask_beta_value = 15 if not len(sys.argv) > 3 else int(sys.argv[3])


	experiment_base_path = "exp_1" if not len(sys.argv) > 4 else sys.argv[4]
	experiment_path = os.path.join(experiment_base_path, "beta_%d_mlatent_%d"%(beta_value, mask_latent))
	os.makedirs(experiment_path, exist_ok=True)

	# initialize model and dataset objects
	image_dir = os.path.join(experiment_path, cuo.image_dir)
	model_setup_dir = os.path.join(experiment_path, cuo.model_setup_dir)
	model_save_file = os.path.join(experiment_path, cuo.model_save_file)

	dataset_manager, dataset = cuo.dataset_manager, cuo.dataset
	model = cuo.get_model(beta_value, num_latents=10)
	#model = cuo.get_model(beta_value, num_channels=6)
	preprocessing = cuo.preprocessing
	inputs_test = cuo.inputs_test
	inputs_test = preprocessing(inputs_test[:32])
	"""
	# initialize mask object
	mask_obj = Mask(mask_latent, beta_value=mask_beta_value)
	mask_obj(inputs_test)
	plt.imshow(mask_obj.view_mask_traversals(inputs_test[:2]))
	plt.savefig(
		"%s/mask.svg"%experiment_base_path,
		format="svg", dpi=1200)
	inputs_test = np.concatenate((inputs_test, mask_obj.mask), -1)
	#"""
	# set Parameters
	approve_run = True
	batch_size = 32

	# run model and dataset objects
	dataset = ut.dataset.DatasetBatch(dataset, batch_size).get_next


	# define parameters
	training_object = TrainVAE(
		model=model,
		dataset=dataset,
		inputs_test=inputs_test,
		preprocessing=preprocessing,
		image_dir=image_dir,
		model_setup_dir=model_setup_dir,
		model_save_file=model_save_file,
		approve_run=approve_run,
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # use default
		#mask_obj=mask_obj
		)

	#run training
	training_object(measure_time = False)
	print("finished beta %d"%beta_value)

if __name__ == '__main__':
	main()