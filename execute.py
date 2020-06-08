import copy
import shutil
import time
import multiprocessing
import os
import numpy as np
import utils as ut
from collections import OrderedDict
import copy
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



def run_training(base_path, gpu_num=0, **kw):
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
	import core.config as cfg 
	from core.model.handler import ProVLAEModelHandler
	print("Running %s"%base_path)
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


def apply_kwargs(fn, kwargs):
	return fn(**kwargs)

def starmap_with_kwargs(pool, fn, kwargs_iter):
	args_for_starmap = zip(repeat(fn), kwargs_iter)
	return pool.starmap(apply_kwargs, args_for_starmap)

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


def main():
	
	# set parameters
	parameters = OrderedDict(
		#hparam_schedule = [
		#	lambda step: hparam_schedule_template(step=step, a=10000, b=20000, c=40000),
		#	lambda step: hparam_schedule_template(step=step, a=20000, b=20000, c=40000),
		#	],
		beta = [2,5],
		random_seed = [1,5,20],
		gamma = [0.5],
		num_latents = [3, 10],
		latent_connections = [None, [1,3], [1,2], [2]])

	base_path = "exp2/exp_"

	kwargs_set = mix_parameters(parameters)

	# create experiment path:
	sub_folder = ["beta", "random_seed"] # parameters for subfolders
	non_sub_folder = [i for i in parameters.keys() if not i in sub_folder]
	folder_index = {}
	i = 0
	for runset in kwargs_set:
		index_set = str([runset[i] for i in non_sub_folder])
		if not index_set in folder_index:
			folder_index[index_set] = i
			i+=1
		sf = ["%s_%s"%(i, runset[i]) for i in sub_folder] 
		runset["base_path"] = os.path.join(base_path+str(folder_index[index_set]), *sf)


	# set gpu device
	for i,v in enumerate(kwargs_set):
		v["gpu_num"] = i%2


	# run processing
	with multiprocessing.Pool(8) as pool:
		starmap_with_kwargs(pool, run_training, kwargs_set)


if __name__ == '__main__':
	main()