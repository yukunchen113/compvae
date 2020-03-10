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

def image_traversal(model, inputs, latent_of_focus=3, min_value=0, max_value=3, num_steps=15):
	"""Traverses the latent space
	
	Args:
		model (Tensorflow Keras Model): Tensorflow VAE from utils.tf_custom
		inputs (numpy arr): Input images in NHWC
		latent_of_focus (int): Latent element to traverse, arbitraly set to 0 as default
		min_value (int): min value for traversal
		max_value (int): max value for traversal
		num_steps (int): The number of steps between min and max value
	
	Returns:
		Numpy arr: image
	"""
	# get latents in terms of [size, batch_size]
	_, latent_rep, latent_logvar = model.encoder(inputs)

	latent_rep = np.zeros(latent_rep.shape)

	stddev = np.sqrt(np.exp(latent_logvar.numpy()[:,latent_of_focus]))

	# apply delta on the inputs in both ways
	latent_rep_trav = []
	for i in np.linspace(min_value, max_value, 15):
		mod_latent_rep = latent_rep.copy()
		addition = np.zeros(mod_latent_rep.shape)
		addition[:,latent_of_focus] = i
		mod_latent_rep=latent_rep+addition
		latent_rep_trav.append(mod_latent_rep.copy())
	latent_rep_trav = np.asarray(latent_rep_trav)
	latent_rep = np.vstack(latent_rep_trav)

	generated = model.decoder(latent_rep)

	reconst = tf.reshape(generated, (*latent_rep_trav.shape[:2],*generated.shape[1:])).numpy()
	reconst = np.concatenate(reconst,-2) # concatenate horizontally
	reconst = np.concatenate(reconst,-3) # concatenate vertically
	
	real = np.concatenate(inputs,-3)

	image = np.concatenate((real, reconst),-2)
	return image


inputs_test, _ = dataset(2, False, True)
