import matplotlib.pyplot as plt
from utils import general_tools as gt 
import utils as ut
import tensorflow as tf
import numpy as np
import time
import cv2
import os
import shutil
import sys
import commonly_used_objects as cuo
from functools import reduce


# reconstruction loss
class ImageMSE(tf.keras.losses.Loss):
	def call(self, actu, pred):
		reduction_axis = range(1,len(actu.shape))
		# per sample
		loss = tf.math.reduce_sum(tf.math.squared_difference(actu, pred), reduction_axis)
		# per batch
		loss = tf.math.reduce_mean(loss)
		return loss

# regularization loss
def kld_loss_reduction(kld_loss):
	# per batch
	kld_loss = tf.math.reduce_mean(kld_loss)
	return kld_loss


class Train():
	def __init__(self, model, dataset, inputs_test, preprocessing, image_dir, model_setup_dir, model_save_file, approve_run=False):
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
		self.setup_training_specific_objects() #TBD this has some hard coded training parameters

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

	def setup_training_specific_objects(self, learning_rate=0.0005, beta_1=0.5, total_metric=tf.keras.metrics.Mean()):
		"""Setup the training, this is already called by default, by can be called again if you want to customize it.
		
		Args:
		    learning_rate (float): The learning rate for Adam optimizer.
		    beta_1 (float): beta_1 for Adam optimizer
		    total_metric (tf.keras.metrics): The metric to use to evaluate model, currently, mean
		"""
		# training setup
		self.loss_func = ImageMSE()
		self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1)
		self.total_metric = total_metric

	def preprocessed_data(self, x=None):
		inputs, _ = self.dataset()
		return self.preprocessing(inputs)

	def __call__(self):
		"""Trains the model.
		Steps to save the images across training and model are defined here, along with displaying curret loss and metrics 

		Total training time: 100000 steps, TBD: if want wvalidation switch this. 
		"""
		def save_image_step(step):
			steps = [1,2,3,5,7,10,15,20,30,40,75,100,200,300,500,700,1000,1500,2500]
			return step in steps or step%5000 == 0

		step = -1
		while 1: # set this using validation
			step+=1
			inputs = self.preprocessed_data(None)
			with tf.GradientTape() as tape:
				reconstruct = self.model(inputs)
				reconstruction_loss = self.loss_func(inputs, reconstruct)
				regularization_loss = kld_loss_reduction(self.model.losses[0])
				loss = reconstruction_loss+regularization_loss
			grads = tape.gradient(loss, self.model.trainable_weights)
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

			if save_image_step(step):
				print('training step %s:\t rec loss = %s\t, reg loss = %s\t' % (
					step, 
					reconstruction_loss.numpy(),
					regularization_loss.numpy(),
					))
				#true_inputs = self.preprocessing(self.inputs_test[:10])
				#t_im = cuo.image_traversal(self.model, true_inputs)
				#"""
				true_inputs = self.preprocessing(self.inputs_test[:2])
				reconstruct_test = self.model(true_inputs).numpy()
				# concatenate the two reconstructions.
				a = np.concatenate((reconstruct_test[0], reconstruct_test[1]), axis=1)
				
				#concatenate the two true images
				b = np.concatenate((true_inputs[0], true_inputs[1]), axis=1)

				t_im = np.concatenate((a,b), axis=0)
				#"""
				plt.imshow(t_im)
				plt.savefig(os.path.join(self.image_dir, "%d.png"%step))
			
			if step%10000 == 0:
				self.model.save_weights(self.model_save_file)

			#TBD: this should be replaced with validation
			if step>=100000:
				break

def main():
	experiment_base_path = "exp_2"

	for beta_value in [1,2,5,10,15,20,30,50,75,100]:
		experiment_path = os.path.join(experiment_base_path, "beta_%d"%beta_value)
		os.makedirs(experiment_path, exist_ok=True)

		# initialize model and dataset objects
		image_dir = os.path.join(experiment_path, cuo.image_dir)
		model_setup_dir = os.path.join(experiment_path, cuo.model_setup_dir)
		model_save_file = os.path.join(experiment_path, cuo.model_save_file)

		dataset_manager, dataset = cuo.dataset_manager, cuo.dataset
		model = cuo.get_model(beta_value)
		preprocessing = cuo.preprocessing
		inputs_test = cuo.inputs_test

		# set Parameters
		approve_run = False
		batch_size = 32

		# run model and dataset objects
		dataset = ut.dataset.DatasetBatch(dataset, batch_size).get_next

		# define parameters
		training_object = Train(model=model,
			dataset=dataset,
			inputs_test=inputs_test,
			preprocessing=preprocessing,
			image_dir=image_dir,
			model_setup_dir=model_setup_dir,
			model_save_file=model_save_file,
			approve_run=True)

		#run training
		training_object()
		print("finished beta %d"%beta_value)



if __name__ == '__main__':
	main()