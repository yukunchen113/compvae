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


class GPUMemoryUsageMonitor:
	def __init__(self):
		from pynvml.smi import nvidia_smi
		self.nvsmi = nvidia_smi.getInstance()
	def get_memory_usage(self, gpu_num=0):
		"""returns amount of memory used on gpus as string
		"""
		memory_usage = self.nvsmi.DeviceQuery('memory.free, memory.total')
		mem_dict = memory_usage["gpu"][gpu_num]["fb_memory_usage"]
		return str(mem_dict["total"]-mem_dict["free"])+" "+mem_dict["unit"]


#################
# Regular Utils #
#################

def import_given_path(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


# reconstruction loss
class ImageMSE(): # mean squared error
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

class ImageBCE(): # binary cross entropy
	def __init__(self, loss_process=lambda x:x):
		self.loss_process = loss_process

	def __call__(self, actu, pred):
		reduction_axis = range(1,len(actu.shape))

		# per point
		sbf = actu.shape
		loss = actu*(-tf.math.log(pred))+(1-actu)*(-tf.math.log(1-pred))

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

def image_traversal(model, inputs, min_value=-3, max_value=3, num_steps=15, is_visualizable=True, latent_of_focus=None, Traversal=ut.visualize.Traversal, return_traversal_object=False):
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



