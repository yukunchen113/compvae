"""This file holds experiments that are performed, to achieve an understanding outlined in the readme file.
"""
import tensorflow as tf
import numpy as np
import utils as ut
try:
	from . import commonly_used_objects as cuo
except ImportError:
	import commonly_used_objects as cuo
import matplotlib.pyplot as plt
import os
import pathlib

cur_folder_path = pathlib.Path(__file__).parent.absolute()



class Mask():
	"""traversal mask object
	"""
	def __init__(self, default_latent_of_focus, default_latent_space_distance=1/5, beta_value=30, mask_base_path = os.path.join(cur_folder_path,"exp_1")):
		"""Initializes mask
		
		Args:
		    beta_value (int): Beta value for pretrained model, loads default model for default path
		"""
		self.default_latent_of_focus = default_latent_of_focus
		self.default_latent_space_distance = default_latent_space_distance
		self.mask = None
		self.model = None
		self.load_model(beta_value, mask_base_path=mask_base_path)
		self.model.trainable=False

	def __call__(self, inputs, latent_space_distance=1/5, override_latent_of_focus=None, measure_time=False):
		"""Creates the mask

		TBD: make a parameter beta_value in cuo.mask_traversal, so you don't need to traverse complete latent space

		
		Args:
		    inputs (numpy array): The inputs to traverse and mask
		    latent_space_distance (float, optional): the distance along the latent of focus to traverse to create the mask
		    override_latent_of_focus (None, int, optional): will temporily use this as latent of focus for this one mask.
		
		Returns:
		    numpy array: returns output of get_mask, the most recent mask
		
		"""
		if measure_time: timer_func = ut.general_tools.Timer()
		if latent_space_distance is None:
			latent_space_distance = self.default_latent_space_distance

		lof = self.default_latent_of_focus
		if not override_latent_of_focus is None:
			lof = override_latent_of_focus
		if measure_time: timer_func("MASK: loaded parameters")
		
		# shape inputs
		input_shape = inputs.shape[1:-1]
		inputs = tf.image.resize(inputs, [64,64])
		if measure_time: timer_func("MASK: resized image")

		traverse = cuo.mask_traversal(self.model,
			inputs,
			min_value=0, max_value=latent_space_distance, 
			num_steps=2, is_visualizable=False, latent_of_focus=lof, return_traversal_object=True)
		self.mask = traverse.get_mask(0)
		if measure_time: timer_func("MASK: got mask/done traversal")

		# make mask the same shape as inputs
		self.mask = tf.image.resize(self.mask, input_shape).numpy().astype(int)
		if measure_time: timer_func("MASK: made mask same shape as inputs")
		
		return self.mask

	def load_model(self, beta_value=30, mask_base_path = "exp_1"):
		"""Model to traverse to get mask
		"""

		experiment_path = os.path.join(mask_base_path, "beta_%d"%beta_value)
		assert os.path.exists(experiment_path), "please specify a valid mask_base_path with load model"

		# initialize model, get preprocessing
		self.model_save_file = os.path.join(experiment_path, cuo.model_save_file)

		self.model = ut.tf_custom.architectures.variational_autoencoder.BetaTCVAE(beta_value) # beta shouldn't matter here since no training. model loaded is specified by model_save_file
		self.model.load_weights(self.model_save_file)

	def get_mask(self):
		"""Returns the most recent mask
		"""
		return self.mask

	def apply(self, inputs, null_mask=0):
		"""
		Applies the mask on the inputs, must have the same shape as the mask
		"""
		assert not self.mask is None, "mask was not set yet"
		outputs = tf.where(self.mask, inputs, null_mask)
		return outputs

	def view_mask_traversals(self, inputs, latent_space_distance=None, num_steps=15):
		if latent_space_distance is None:
			latent_space_distance = self.default_latent_space_distance
		inputs = tf.image.resize(inputs, [64,64])
		image_of_traversals = cuo.mask_traversal(self.model,
			inputs,
			min_value=0, max_value=latent_space_distance*num_steps, 
			num_steps=num_steps, is_visualizable=True, return_traversal_object=False, is_interweave=True)
		return image_of_traversals

	@property
	def shape(self):
		return self.mask.shape
	


def main():
	preprocessing = cuo.preprocessing
	inputs_test = cuo.inputs_test
	inputs_test = preprocessing(inputs_test)
	inputs_test = inputs_test[:32]
	plt.imshow(inputs_test[1])
	plt.show()
	exit()
	mask = Mask(0, beta_value= 15) # mask the 0th element

	mask(inputs_test[:2])
	
	# generate mask
	generated = mask.get_mask()[0]

	# apply mask
	generated = mask.apply(inputs_test[:2])[0]

	# check traversal
	#generated = mask.view_mask_traversals(inputs_test[:2])

	plt.imshow(generated)
	plt.savefig(
		"exp_1/mask.svg",
		format="svg", dpi=1200)


def test_reconstruction():
	"""Success
	"""
	a = model(inputs_test[:2])
	plt.imshow(a[0])
	plt.show()

if __name__ == '__main__':
	main()