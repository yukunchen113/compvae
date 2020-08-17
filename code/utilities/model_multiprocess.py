import copy
import time
import numpy as np
from itertools import repeat

def train_wrapper(func):
	"""Handles the resource errors
	
	Args:
		func (nparray): Function to run training
	
	Returns:
		func return
	"""
	import tensorflow as tf
	while 1:
		try:
			ret = func()
			return ret
		except tf.errors.ResourceExhaustedError:
			time.sleep(np.random.randint(3,7)*60) # wait and try again
		except tf.errors.InternalError:
			time.sleep(np.random.randint(1,7)) # wait and try again
		except tf.errors.UnknownError:
			time.sleep(np.random.randint(1,3)*60) # wait and try again

def apply_kwargs(fn, kwargs):
	return fn(**kwargs)

def starmap_with_kwargs(pool, fn, kwargs_iter):
	args_for_starmap = zip(repeat(fn), kwargs_iter)
	return pool.starmap(apply_kwargs, args_for_starmap)

