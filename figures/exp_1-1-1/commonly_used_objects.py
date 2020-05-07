"""This file contains the commonly used objects by both the train.py and the experiments.py
"""
import tensorflow as tf
import os
import utils as ut 
import numpy as np

#limit GPU usage (from tensiorflow code)
gpus = tf.config.experimental.list_physical_devices('GPU')
for i in gpus:
	tf.config.experimental.set_memory_growth(i, True)
"""
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
		tf.config.experimental.set_virtual_device_configuration(gpus[0],
			[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])

		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)
######################
"""

image_dir = "images"
model_setup_dir = "model_setup"
model_save_file = os.path.join(model_setup_dir, "model_weights.h5")

dataset_manager, dataset = ut.dataset.get_celeba_data(
	ut.general_constants.datapath, 
	is_HD=False,
	group_num=8)

inputs_test, _ = dataset(2, False, True)


def get_model(*args, **kwargs):
	model = ut.tf_custom.architectures.variational_autoencoder.BetaTCVAE(
		*args, **kwargs)
	return model

def preprocessing(inputs):
	# crop to 128x128 (centered), this number was experimentally found
	image_crop_size = [128,128]
	inputs=tf.image.crop_to_bounding_box(inputs, 
		(inputs.shape[-3]-image_crop_size[0])//2,
		(inputs.shape[-2]-image_crop_size[1])//2,
		image_crop_size[0],
		image_crop_size[1],
		)
	inputs = tf.image.convert_image_dtype(inputs, tf.float32)
	inputs = tf.image.resize(inputs, [64,64])
	return inputs

def image_traversal(model, inputs, min_value=0, max_value=3, num_steps=15, is_visualizable=True, Traversal=ut.visualize.Traversal):
	"""Standard raversal of the latent space
	
	Args:
	    model (Tensorflow Keras Model): Tensorflow VAE from utils.tf_custom
	    inputs (numpy arr): Input images in NHWC
	    min_value (int): min value for traversal
	    max_value (int): max value for traversal
	    num_steps (int): The number of steps between min and max value
	    is_visualizable (bool, optional): If false, will return a traversal tensor of shape [traversal_steps, num_images, W, H, C]
	
	Returns:
	    Numpy arr: image
	"""
	t = ut.general_tools.Timer()
	traverse = Traversal(model, inputs)
	traverse.traverse_complete_latent_space(min_value=min_value, max_value=max_value, num_steps=num_steps)
	traverse.create_samples()
	if not is_visualizable:
		return traverse.samples
	image = traverse.construct_single_image()
	return image 
