import os
import copy
import multiprocessing
from collections import OrderedDict
from utilities.model_multiprocess import train_wrapper, starmap_with_kwargs
import numpy as np
import execute_params as ep
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

def create_model(base_path, **kw):
	import core.config as cfg 
	from core.model.handler import ProVLAEModelHandler
	print("Running %s"%base_path)
	config = cfg.config.ConfigShapes3D()
	for k,v in kw.items():
		setattr(config, k, v)
	modhand = ProVLAEModelHandler(config=config, base_path=base_path, config_processing=cfg.addition.make_vlae_large)
	return modhand

def run_training(base_path, gpu_num=0, **kw):
	#os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
	modhand = create_model(base_path=base_path, **kw)
	modhand.save()
	train_wrapper(modhand.train)


def main():

	# this model conditions the 0th latent on the next 3 latents through KL
	# path
	base_path = "experiments/shapes3d/exp2/exp_"
	if os.path.exists(os.path.dirname(base_path)):
		if not "y" in input("do you want to use existing path?"):
			exit()
	
	# set parameters
	parameters = OrderedDict(
		random_seed = [1,5,20], #Using BetaVAE
		beta = [1], # will be overwritten
		num_latents = [7],
		latent_connections = [[0,1,2]],
		)
	parameters['hparam_schedule'] = [lambda step: ep.hparam_schedule_alpha_beta2(step, len(i)+1, final_beta=j) for i in parameters["latent_connections"] for j in parameters["beta"]]



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

	##################
	# run processing #
	##################
	for i in kwargs_set:
		run_training(**i)

	##########################
	# multiprocessing method #
	##########################
	# set gpu device
	#for i,v in enumerate(kwargs_set):
	#	v["gpu_num"] = i%2

	#with multiprocessing.Pool(2) as pool:
	#	starmap_with_kwargs(pool, run_training, kwargs_set)

if __name__ == '__main__':
	main()