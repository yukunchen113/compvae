import utils as ut
import tensorflow as tf 
import numpy as np 
import importlib.util
#################
# Visualization #
#################
class VLAETraversal(ut.visualize.Traversal): #create wrapper for model encoder and decoder

	"""
	TODO:
		- add capabilities for image vertical stack for each hierarchical layer.
	"""
	
	def __init__(self,*ar,**kw):
		super().__init__(*ar, **kw)
		self.latent_hierarchy = None

	def encode(self, inputs):
		latent_space = self.model.encoder(inputs)
		# latent space is of shape [num latent space, 3-[sanmes, mean, logvar], batchsize, num latents]
		self.latent_hierarchy = [i[0].shape[-1] for i in latent_space]
		latent_space = tf.concat(latent_space, -1)
		return latent_space

	def decode(self, samples):
		samples = np.split(samples, np.cumsum(self.latent_hierarchy)[:-1], axis=-1) # we don't include last as that is the end
		ret = self.model.decoder(samples)
		return ret
	@property
	def num_latents(self):
		num_latents = sum([i.num_latents for i in self.model.ladders if not i is None])+self.model.num_latents
		return num_latents

	@property
	def samples_list(self):
		s = self.samples.shape
		samples = self.samples.reshape(s[0],-1,np.sum(self.latent_hierarchy),*s[2:])
		samples = np.split(samples, np.cumsum(self.latent_hierarchy)[:-1], axis=2)
		self.inputs = np.broadcast_to(np.expand_dims(self.orig_inputs,1), samples[0].shape)
		self.inputs = self.inputs.reshape(self.inputs.shape[0],-1, *self.inputs.shape[-3:])
		samples = [i.reshape(i.shape[0],-1, *i.shape[-3:]) for i in samples]
		samples = [self.inputs]+samples
		return samples

def vlae_traversal(model, inputs, min_value=-3, max_value=3, num_steps=30, is_visualizable=True, latent_of_focus=None, Traversal=VLAETraversal, return_traversal_object=False):
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
	traverse = Traversal(model=model, inputs=inputs)
	#t("Timer Creation")
	if latent_of_focus is None:
		traverse.traverse_complete_latent_space(min_value=min_value, max_value=max_value, num_steps=num_steps)
	else:
		traverse.traverse_latent_space(latent_of_focus=latent_of_focus, min_value=min_value, max_value=max_value, num_steps=num_steps)

	#t("Timer Traversed")
	traverse.create_samples()
	#t("Timer Create Samples")
	if return_traversal_object:
		return traverse
	if not is_visualizable:
		return traverse.samples
	image = traverse.construct_single_image()
	return image 
