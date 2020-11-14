import disentangle as dt
import tensorflow as tf 
import numpy as np 
import importlib.util
from functools import reduce
##########################
# Config and Setup Utils #
##########################
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

class TrainObjMetaClass(type):
	"""
	metaclass to control Train objects.
	Config must only have static methods.
	This only need to be used by config_default, other configs
	should inhereit from config_default
	"""
	def __new__(cls, name, bases, body):
		assert "train_step" in body and callable(body["train_step"]), "TrainObj must have a method called train_step which is called to run one interation"				
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
class cprint: # color codes from https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	@classmethod
	def header(cls, *ar,**kw):
		print(cls.HEADER, *ar, cls.ENDC, **kw)
	@classmethod
	def blue(cls, *ar,**kw):
		print(cls.BLUE, *ar, cls.ENDC, **kw)
	@classmethod
	def green(cls, *ar,**kw):
		print(cls.GREEN, *ar, cls.ENDC, **kw)
	@classmethod
	def warning(cls, *ar,**kw):
		print(cls.WARNING, *ar, cls.ENDC, **kw)
	@classmethod
	def fail(cls, *ar,**kw):
		print(cls.FAIL, *ar, cls.ENDC, **kw)
	@classmethod
	def bold(cls, *ar,**kw):
		print(cls.BOLD, *ar, cls.ENDC, **kw)	
	@classmethod
	def underline(cls, *ar,**kw):
		print(cls.UNDERLINE, *ar, cls.ENDC, **kw)

def import_given_path(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod

class Compare:
	@classmethod
	def num(cls,a,b):
		if np.all(abs(a - b)<1e-8):
			return True
		return False

	@classmethod	
	def type(cls,i,j):
		j = type(i)(j)
		if type(i) == dict:
			if not cls.dictionary(i,j):
				cprint.fail("check_dictionary", i, j)
				return False
		elif type(i) in [list, tuple]:
			if not cls.list(i,j):
				cprint.fail("check_list", i, j)
				return False
		else:
			if not cls.num(i,j):
				cprint.fail("check_num", i, j)
				return False
		return True

	@classmethod	
	def dictionary(cls,a,b):
		if not len(a) == len(b): 
			return False
		for k in a.keys():
			if not k in a: return False
		for k in a.keys():
			if not cls.type(a[k],b[k]): return False
		return True

	@classmethod	
	def list(cls,a, b):
		if not len(a) == len(b): return False
		for i,j in zip(a,b):
			if not cls.type(i,j): return False
		return True


#######################
# Training/Loss Utils #
#######################

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

	def __call__(self, actu, pred, label_smooting_pad=1e-5):
		reduction_axis = range(1,len(actu.shape))

		# apply label smooting
		actu = actu*(1-2*label_smooting_pad)+label_smooting_pad
		pred = pred*(1-2*label_smooting_pad)+label_smooting_pad

		# per point
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


######################
# Architecture Utils #
######################
def split_latent_into_layer(inputs, num_latents):
	mean = inputs[:,:num_latents]
	logvar = inputs[:,num_latents:]
	sample = tf.exp(0.5*logvar)*tf.random.normal(
		tf.shape(logvar))+mean
	return sample, mean, logvar

def set_shape(layer, shape):
	sequence = tf.keras.Sequential([
		tf.keras.Input(shape), layer])
	return sequence

#######################
# Visualization Utils #
#######################
def image_traversal(model, inputs, min_value=-3, max_value=3, num_steps=15, is_visualizable=True, latent_of_focus=None, Traversal=dt.visualize.Traversal, return_traversal_object=False):
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
	#t = dt.general.tools.Timer()
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



