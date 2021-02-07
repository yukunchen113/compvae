import disentangle as dt
import tensorflow as tf 
import numpy as np 
import scipy.stats as scist


#################
# Visualization #
#################
class LVAETraversal(dt.visualize.Traversal): #create wrapper for model encoder and decoder
	def __init__(self,*ar,hparam_obj=None,**kw):
		super().__init__(*ar, **kw)
		self.latent_hierarchy = None
		self.hparam_obj = hparam_obj

	def sample_latent_space(self):
		"""Will sample all latent space. 
		The num images and num latents dimensions will be flattened to one dimension
		shape of latents will be: [num images, num latents]
		"""
		latent_reps = []
		inputs = None
		
		# accumulate images for all the different latent representations, for all images
		for i in range(self.num_latents):
			self.sample_latent_layer(layernum=None,latent_of_focus=i)
			latent_reps.append(self.latent_rep_trav.copy())
			
			if inputs is None:
				inputs = np.empty((self.num_latents, *self.inputs.shape))
			inputs[i] = self.inputs

		# latents
		latent_reps = np.asarray(latent_reps)
		latent_reps = np.transpose(latent_reps, (2,0,1,3)) 
		self.latent_rep_trav = latent_reps.reshape((-1, *latent_reps.shape[-2:])).transpose((1,0,2))
		# self.latent_rep_trav is flattened from [num_images, num_latents]

		# inputs duplication
		inputs = np.transpose(inputs, (1,0,2,3,4))
		inputs = np.reshape(inputs, (-1, *inputs.shape[-3:]))
		self.inputs = inputs

	def sample_latent_layer(self, layernum=None, latent_of_focus=None):
		"""samples a latent layer.
		
		Args:
			layernum (int): Latent layer to sample
			latent_of_focus (int,None, optional): latent number, this will be used to find layernum if that is not specified
		
		"""
		assert (layernum is None) != (latent_of_focus is None)
		# initialize latent representation of images
		_, mean, logvar = self.encode(self.inputs)
		mean = mean.numpy()
		logvar = logvar.numpy()
		
		layer_separation = [0]+list(np.cumsum(self.latent_hierarchy))
		
		# find layernum if not specified
		if layernum is None:
			for layernum,sep in enumerate(layer_separation[1:]):
				if latent_of_focus < sep:break

		#mean = np.split(mean, np.cumsum(self.latent_hierarchy)[:-1], axis=-1)[layernum]
		#logvar = np.split(logvar, np.cumsum(self.latent_hierarchy)[:-1], axis=-1)[layernum]

		# create latent sample
		self.latent_rep_trav = np.expand_dims(np.exp(0.5*logvar)*np.random.normal(size=logvar.shape)+mean,0) #expand dim as 0th axis will be used for gif
		assert not np.any(np.isnan(self.latent_rep_trav)), "samples has nan"

		# resample lower layer values
		# This should be sampled after top layer latent comes out.
		self.latent_rep_trav[...,:layer_separation[layernum]] = np.nan
		#print(layer_separation[layernum],layernum,latent_of_focus,layer_separation[-1])
		# set inputs
		self.inputs = self.orig_inputs

	def traverse_latent_space(self, latent_of_focus, min_value=-3, max_value=3, num_steps=30, add_min_max=False):
		"""traverses the latent space, focuses on one latent for each given image.
		
		Args:
			latent_of_focus (int): Latent element to traverse, arbitraly set to 0 as default
			min_value (int): min value for traversal
			max_value (int): max value for traversal
			num_steps (int): The number of steps between min and max value
		
		"""
		# initialize latent representation of images
		_, latent_rep, _ = self.encode(self.inputs)
		latent_rep = latent_rep.numpy()
		
		layer_separation = [0]+list(np.cumsum(self.latent_hierarchy))
		split_latent_rep = np.split(latent_rep, np.cumsum(self.latent_hierarchy)[:-1], axis=-1)
		for i,sep in enumerate(layer_separation[1:]):
			layernum=i
			if latent_of_focus < sep:
				# traverse based on prior space
				if i<len(split_latent_rep)-1:
					# get the prior space for that layer
					_,prior=self.model.decoder(
						[None for _ in range(len(split_latent_rep)-1)]+[np.zeros_like(split_latent_rep[-1])],
						return_ls=True, use_mean=True)
					prior=prior[i]
					mean,stddev = prior[1:]
					# get min and max of latent space
					stddev = np.sqrt(np.exp(stddev.numpy()))
					lnum = latent_of_focus-layer_separation[i]
					min_value = min_value*stddev[:,lnum]+mean[:,lnum].numpy()
					max_value = max_value*stddev[:,lnum]+mean[:,lnum].numpy()
				break

		# create latent traversal
		latent_rep_trav = []
		for i in np.linspace(min_value, max_value, num_steps):
			mod_latent_rep = latent_rep.copy()
			addition = np.zeros(mod_latent_rep.shape)
			addition[:,latent_of_focus] = i
			mod_latent_rep=latent_rep
			if not add_min_max:
				mod_latent_rep[:,latent_of_focus]=0
			mod_latent_rep+=addition
			latent_rep_trav.append(mod_latent_rep.copy())
		self.latent_rep_trav = np.asarray(latent_rep_trav)

		assert not np.any(np.isnan(self.latent_rep_trav)), "traversal has nan"

		# resample lower layer values
		# This should be sampled after top layer latent comes out.
		self.latent_rep_trav[...,:layer_separation[layernum]] = np.nan
		#print(layer_separation[layernum],layernum,latent_of_focus,layer_separation[-1])
		# set inputs
		self.inputs = self.orig_inputs

	def encode(self, inputs, **kw):
		latent_space = self.model.encoder(inputs,**kw)
		self._original_latent_space = latent_space

		# latent space is of shape [num latent space, 3-[samples, mean, logvar], batchsize, num latents]
		self.latent_hierarchy = [i[0].shape[-1] for i in latent_space]
		latent_space = tf.concat(latent_space, -1)
		return latent_space

	def create_samples(self, batch_size=16):
		"""Creates the sample from the latent representation traversal
		"""
		assert not self.latent_rep_trav is None, "Please call traverse_latent_space first to get latent elements for self.latent_rep_trav"

		# flattened latent traversal for one batch dimension (assuming that the latent traversal is of the size, [num traversal, N, num latents])
		latent_rep = np.vstack(self.latent_rep_trav)

		# get the samples
		generated = None
		for i in range(np.ceil(latent_rep.shape[0]/batch_size).astype(int)):
			gen = self.decode(latent_rep[i*batch_size:(i+1)*batch_size])
			if generated is None:
				generated = np.empty((latent_rep.shape[0],*gen.shape[1:]))
			generated[i*batch_size:(i+1)*batch_size] = gen
		# reshape back to [num traversal, N, W, H, C], as per self.latent_rep_trav
		self.samples = tf.reshape(generated, (*self.latent_rep_trav.shape[:2],*generated.shape[1:])).numpy()
	
	def decode(self, samples):
		num_nans = np.sum(np.isnan(samples),-1)
		unique_num_nans = list(set(num_nans))

		out = None
		for n in unique_num_nans:
			idx = n==num_nans
			sample = samples[idx]
			sample = np.split(sample, np.cumsum(self.latent_hierarchy)[:-1], axis=-1) # we don't include last as that is the end
			# check sample if need to resample layer
			sample = [None if np.all(np.isnan(s)) else s for s in sample]
			ret = self.model.decoder(sample, use_mean=True)

			# fill out
			if out is None:
				out = np.zeros([samples.shape[0]]+[s for s in ret.shape[1:]])+np.nan
			out[idx]=ret.numpy()
		return out
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

def lvae_traversal(model, inputs, min_value=-3, max_value=3, num_steps=30, is_visualizable=True, latent_of_focus=None, Traversal=LVAETraversal, return_traversal_object=False, hparam_obj=None, is_sample=False):
	"""Standard traversal of the latent space
	
	Args:
		model (Tensorflow Keras Model): Tensorflow VAE from utils.tf_custom
		inputs (numpy arr): Input images in NHWC
		min_value (int): min value for traversal of prior standard dev from prior mean.
		max_value (int): max value for traversal of prior standard dev from prior mean.
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
	if is_sample:
		traverse.sample_latent_space()
	else:
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

def vlae_traversal(model, inputs, min_value=-3, max_value=3, num_steps=30, is_visualizable=True, latent_of_focus=None, Traversal=VLAETraversal, return_traversal_object=False, hparam_obj=None, **kw):
	"""Standard traversal of the latent space
	
	Args:
		model (Tensorflow Keras Model): VLAE from hsr 
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

class VAETraversal(dt.visualize.Traversal):
	@property
	def samples_list(self):
		inputs = np.broadcast_to(self.inputs, self.samples[0].shape)
		print(inputs, self.samples.shape)
		return [inputs, self.samples]

def vae_traversal(model, inputs, min_value=-3, max_value=3, num_steps=30, is_visualizable=True, latent_of_focus=None, Traversal=VAETraversal, return_traversal_object=False, **kw):
	"""Standard raversal of the latent space
	
	Args:
		model (Tensorflow Keras Model): VAE 
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
