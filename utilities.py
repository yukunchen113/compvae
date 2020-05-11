import utils as ut
import tensorflow as tf 
import numpy as np 
import importlib.util

###########################
# Config and Setup  Utils #
###########################
def import_given_path(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod

class ConfigMetaClass(type):
	"""
	metaclass to control config.
	Config must only have static methods.
	This only need to be used by config_default, other configs
	should inhereit from config_default
	"""
	def __new__(cls, name, bases, body):
		"""
		for attr, value in body.items():
			if callable(value) and not attr.startswith("_"):
				body[attr] = staticmethod(value)
		"""
		return type.__new__(cls, name, bases, body)

##############
# Mask Utils #
##############
class Mask:
	"""traversal mask object
	"""
	def __init__(self, mask_model, default_latent_of_focus, default_latent_space_distance=1/5):
		"""Initializes mask
		"""
		self.model = mask_model
		self.default_latent_of_focus = default_latent_of_focus
		self.default_latent_space_distance = default_latent_space_distance

		self.mask = None
		self.model.trainable=False

	def __call__(self, inputs, latent_space_distance=None, override_latent_of_focus=None, measure_time=False):
		"""Creates the mask
		
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
		inputs = tf.image.resize(inputs, self.model.shape_input[:-1])
		if measure_time: timer_func("MASK: resized image")

		traverse = mask_traversal(self.model, inputs,
			min_value=0, max_value=latent_space_distance, 
			num_steps=2, is_visualizable=False, latent_of_focus=lof, return_traversal_object=True)
		self.mask = traverse.get_mask(0)
		if measure_time: timer_func("MASK: got mask/done traversal")

		# make mask the same shape as inputs
		self.mask = tf.image.resize(self.mask, input_shape).numpy().astype(int)
		if measure_time: timer_func("MASK: made mask same shape as inputs")
		
		return self.mask

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
		inputs = tf.image.resize(inputs, self.model.shape_input[:-1])
		image_of_traversals = mask_traversal(self.model,
			inputs,
			min_value=0, max_value=latent_space_distance*num_steps, 
			num_steps=num_steps, is_visualizable=True, return_traversal_object=False, is_interweave=True)
		return image_of_traversals

	@property
	def shape(self):
		return self.mask.shape

class MaskedTraversal(ut.visualize.Traversal):
	def __init__(self, *args, pixel_diff_threshold=0.005, **kwargs):
		super().__init__(*args, **kwargs)
		self.masked_inputs = None
		self.pixel_diff_threshold =pixel_diff_threshold
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
		mask = mask>self.pixel_diff_threshold
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

def mask_traversal(model, inputs, min_value=0, max_value=3, num_steps=15, is_visualizable=True, latent_of_focus=None, Traversal=MaskedTraversal, return_traversal_object=False, is_interweave=False):
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
	traverse.create_samples(is_interweave=is_interweave, mask_only=return_traversal_object)
	#t("Timer Create Samples")
	if return_traversal_object:
		return traverse
	if not is_visualizable:
		return traverse.samples
	image = traverse.construct_single_image()
	return image 



#################
# Regular Utils #
#################

def import_given_path(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


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

def image_traversal(model, inputs, min_value=0, max_value=3, num_steps=15, is_visualizable=True, Traversal=ut.visualize.Traversal, 
			return_traversal_object=False):
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
	#t = ut.general_tools.Timer()
	traverse = Traversal(model, inputs)
	#t("Timer Creation")
	traverse.traverse_complete_latent_space(min_value=min_value, max_value=max_value, num_steps=num_steps)
	#t("Timer Traversed")
	traverse.create_samples()
	#t("Timer Create Samples")
	if return_traversal_object:
		return traverse
	if not is_visualizable:
		return traverse.samples
	image = traverse.construct_single_image()
	return image 


