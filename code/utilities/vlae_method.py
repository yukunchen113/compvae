"""
TBD: latent shortcut doesn't account for multiple parents
"""
import disentangle as dt
import tensorflow as tf 
import numpy as np 
import scipy.stats as scist


#################
# Visualization #
#################
class VLAETraversal(dt.visualize.Traversal): #create wrapper for model encoder and decoder
	def __init__(self,*ar,hparam_obj=None,**kw):
		super().__init__(*ar, **kw)
		self.latent_hierarchy = None
		self.hparam_obj = hparam_obj

	def encode(self, inputs):
		latent_space = self.model.encoder(inputs)
		self._original_latent_space = latent_space

		# latent space is of shape [num latent space, 3-[sanmes, mean, logvar], batchsize, num latents]
		self.latent_hierarchy = [i[0].shape[-1] for i in latent_space]
		latent_space = tf.concat(latent_space, -1)
		return latent_space

	def decode(self, samples):
		if not self.hparam_obj is None:
			if hasattr(self.hparam_obj, "get_latent_mask"):
				# detect latent that are being traversed 
				mode = scist.mode(samples,axis=0)[0]
				activated_latents = (samples-mode).astype(bool)
				
				# apply mask given latent of focus by usng hparam_obj.off_step() 
				start = 0
				for i,lh in enumerate(self.latent_hierarchy):
					end = start+lh
					active_latent = activated_latents[:,start:end]
					active_latent = np.argwhere(active_latent)[:,1]
					for j in active_latent:
						# get current selected latent
						selected_latent = (i,j) 
						latent_mask = self.hparam_obj.off_step(selected_latent)["latent_mask"]

						# mask the latents
						lmask = []
						for lm in latent_mask:
							lmask.append(self.model.setup_mask(lm, (1,lh)).numpy())
						mask = np.concatenate(lmask,-1)
						samples[i]*=mask.reshape([-1])
					start = end


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
		new_shape = (s[0],-1,np.sum(self.latent_hierarchy),*s[2:])
		samples = self.samples.reshape(new_shape)
		samples = np.split(samples, np.cumsum(self.latent_hierarchy)[:-1], axis=2)
		self.inputs = np.broadcast_to(np.expand_dims(self.orig_inputs,1), samples[0].shape)
		self.inputs = self.inputs.reshape(self.inputs.shape[0],-1, *self.inputs.shape[-3:])
		samples = [i.reshape(i.shape[0],-1, *i.shape[-3:]) for i in samples]
		samples = [self.inputs]+samples
		return samples

def vlae_traversal(model, inputs, min_value=-3, max_value=3, num_steps=30, is_visualizable=True, latent_of_focus=None, Traversal=VLAETraversal, return_traversal_object=False, hparam_obj=None):
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
	t = dt.general.tools.Timer()
	traverse = Traversal(model=model, inputs=inputs, hparam_obj=hparam_obj)
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
