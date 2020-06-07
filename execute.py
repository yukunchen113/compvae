import copy
import shutil
import time
import multiprocessing
import os
import numpy as np
import utils as ut
from collections import OrderedDict
import copy

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



def run_training(base_path, **kw):
	import core.config as cfg 
	from core.model.handler import ProVLAEModelHandler

	config = cfg.config.Config()
	for k,v in kw.items():
		setattr(config, k, v)
	modhand = ProVLAEModelHandler(config=config, base_path=base_path)
	modhand.save()
	train_wrapper(modhand.train)


def hparam_schedule_template(step, a=10000, b=20000, c=40000):
	# use increasing weight hyper parameter
	alpha = [0,0,1]
	alpha[1] = np.clip((step-b)/a, 0, 1) # after the first b steps, evolve alpha for a steps
	alpha[0] = np.clip((step-c)/a, 0, 1) # after the first c steps, evolve alpha for a steps
	return dict(alpha=alpha)

from itertools import repeat

def apply_kwargs(fn, kwargs):
	return fn(**kwargs)

def starmap_with_kwargs(pool, fn, kwargs_iter):
	args_for_starmap = zip(repeat(fn), kwargs_iter)
	return pool.starmap(apply_kwargs, args_for_starmap)

def main():
	
	# set parameters
	parameters = OrderedDict(
		#hparam_schedule = [
		#	lambda step: hparam_schedule_template(step=step, a=10000, b=20000, c=40000),
		#	lambda step: hparam_schedule_template(step=step, a=20000, b=20000, c=40000),
		#	],
		beta = [2,5,50],
		random_seed = [1,5,20],
		gamma = [0.5],
		num_latents = [3, 10],
		latent_connections = [None, [1,3], [1,2]])

	base_path = "exp/exp_"

	def mix_parameters(params):
		if params == {}:
			return [{}]

		k, v = params.popitem(last=False)
		ret = []
		for item in v:
			for s in mix_parameters(copy.deepcopy(params)):
				s[k] = item
				ret.append(s)	
		return ret

	kwargs = mix_parameters(parameters)
	for i,v in enumerate(kwargs):
		v["base_path"] = base_path+str(i)

	with multiprocessing.Pool(10) as pool:
		starmap_with_kwargs(pool, run_training, kwargs)


if __name__ == '__main__':
	main()