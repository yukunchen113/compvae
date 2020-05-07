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
	is_HD=64,
	group_num=8)

inputs_test, _ = dataset(2, False, True)


def get_model(*args, **kwargs):
	model = ut.tf_custom.architectures.variational_autoencoder.BetaTCVAE(
		*args, **kwargs)
	return model

def preprocessing(inputs):
	#crop to  (centered), this number was experimentally found
	image_crop_size = [50,50]
	inputs=tf.image.crop_to_bounding_box(inputs, 
		(inputs.shape[-3]-image_crop_size[0])//2,
		(inputs.shape[-2]-image_crop_size[1])//2,
		image_crop_size[0],
		image_crop_size[1],
		)
	inputs = tf.image.convert_image_dtype(inputs, tf.float32)
	inputs = tf.image.resize(inputs, [64,64])
	return inputs

class MaskedTraversal(ut.visualize.Traversal):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.masked_inputs = None

	def create_samples(self, is_interweave=False, mask_only=False):
		"""Will keep last the same
		
		Args:
		    is_interweave (bool, optional): Will interweave the image with rows of masked, unmasked traversals, used for comparisons
		"""	
		super().create_samples()

		# get mask
		generated =self.samples
		im_n = 0
		batch_num = 0
		g0 = generated[:-1] # for 1.1.1
		g1 = generated[1:]
		mask = np.abs(g0 - g1)
		mask = mask>0.005
		masked_g0 = np.where(mask, g0, 0)
		self.mask = np.where(mask[0], 1, 0)
		if not is_interweave:
			self.samples[:-1] = masked_g0
			"""
			# this is to view the effect of the mask
			diff = np.concatenate((mask, g0, masked_g0), -3)
			print(diff.shape)
			diff = np.concatenate(diff, -2)
			plt.imshow(diff)
			plt.show()
			"""
		elif not mask_only:
			# set mask to samples
			s_shape = self.samples.shape
			self.samples = np.expand_dims(self.samples,0)
			self.samples = np.concatenate((self.samples, self.samples),0)
			self.samples[0,:-1] = masked_g0
			self.samples = self.samples.transpose((1,2,0,3,4,5))
			self.samples = self.samples.reshape((s_shape[0], -1, *s_shape[2:]))

			# change inputs
			i_shape = self.inputs.shape
			self.inputs = np.expand_dims(self.inputs,0)
			masked_inputs = np.where(mask[0], self.inputs, 0)
			self.inputs = np.concatenate((masked_inputs, self.inputs),0)

			self.inputs = self.inputs.transpose((1,0,2,3,4))
			self.inputs = self.inputs.reshape((-1, *i_shape[-3:]))
	
	def get_mask(self, latent_num=slice(None), image_num=slice(None)):
		"""This will retrieve a masked latent
		
		Args:
			latent_num (int, optional): This is the latent number, an index for the nth traversal to see
			image_num (int, None, optional): this is the image number in the batch of images
		
		Returns:
			TYPE: specified masked_input
		"""
		assert not self.mask is None, "create_samples must be called first to construct a mask"
		mask = self.mask

		# below indexing assumes one, partial or complete traversal. 
		mask = np.reshape(mask, (self.orig_inputs.shape[0], self.mask.shape[0]//self.orig_inputs.shape[0], *mask.shape[1:]))
		
		try:
			mask = mask[image_num, latent_num]
		except IndexError:
			raise Exception("only traversed %d dimensions with %d images"%(self.mask.shape[0]//self.orig_inputs.shape[0], self.orig_inputs.shape[0]))
			
		return mask


def mask_traversal(model, inputs, min_value=0, max_value=3, num_steps=15, is_visualizable=True, latent_of_focus=None, 
						Traversal=MaskedTraversal, return_traversal_object=False, is_interweave=False):
	"""Standard raversal of the latent space
	
	Args:
	    model (Tensorflow Keras Model): Tensorflow VAE from utils.tf_custom
	    inputs (numpy arr): Input images in NHWC
	    min_value (int): min value for traversal
	    max_value (int): max value for traversal
	    num_steps (int): The number of steps between min and max value
	    is_visualizable (bool, optional): If false, will return a traversal tensor of shape [traversal_steps, num_images, W, H, C]
	    Traversal (Traversal object, optional): This is the traversal object to use
	    return_traversal_object (bool, optional): Whether to return the traversal or not
	
	Returns:
	    Numpy arr: image
	"""
	t = ut.general_tools.Timer()
	traverse = Traversal(model, inputs)
	#t("Timer Creation")
	if latent_of_focus is None:
		traverse.traverse_complete_latent_space(min_value=min_value, max_value=max_value, num_steps=num_steps)
	else:
		traverse.traverse_latent_space(latent_of_focus=latent_of_focus, min_value=min_value, max_value=max_value, num_steps=num_steps)

	#t("Timer Traversed")
	traverse.create_samples(is_interweave=is_interweave, mask_only=return_traversal_object)
	#t("Timer Create Samples")
	if return_traversal_object:
		return traverse
	if not is_visualizable:
		return traverse.samples
	image = traverse.construct_single_image()
	return image 

def main():
	# test dataset, and model
	import matplotlib.pyplot as plt
	batch_size = 32
	dset = ut.dataset.DatasetBatch(dataset, batch_size).get_next
	t = ut.general_tools.Timer()
	data, _ = dset()
	data = preprocessing(data)
	a = get_model(15)
	a(data)
	plt.imshow(data[0])
	plt.show()

if __name__ == '__main__':
	main()