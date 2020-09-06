import os
import copy
import multiprocessing
from collections import OrderedDict
from utilities.model_multiprocess import train_wrapper, starmap_with_kwargs
import utilities.constants as cn
import numpy as np
import utilities.hparams as hp
import shutil
import dill as pickle
import pathlib
import sys
from functools import reduce
def mix_parameters(params, enumerated=False):
	if params == {}:
		return [{}]

	k, v = params.popitem(last=False)
	if enumerated:
		v=enumerate(v)
	ret = []
	for item in v:
		for s in mix_parameters(copy.deepcopy(params),enumerated=enumerated):
			ret.append({**s,k:item})
	return ret

def create_model(base_path, model_size, **kw):
	import core.config as cfg 
	from core.model.handler import ProVLAEModelHandler
	print("Running %s"%base_path)
	config = cfg.config.ConfigShapes3D()
	for k,v in kw.items():
		setattr(config, k, v)
	if model_size:
		config_processing = cfg.addition.make_vlae_large
	else:
		config_processing = cfg.addition.make_vlae_small
	modhand = ProVLAEModelHandler(config=config, base_path=base_path, config_processing=config_processing)
	return modhand

def run_training(base_path, gpu_num=0, **kw):
	#os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
	modhand = create_model(base_path=base_path, **kw)
	modhand.save()
	train_wrapper(modhand.train)

def copy_code(base_path, ignore_folders_contain=["test","exp"],source_dir="."):
	# copy code files to basepath for preservation of experiments
	code_path = os.path.join(base_path,cn.code_folder)
	ignore_folders_contain = ["test","exp"]
	code_files = []# if i[-1].endswith(".py")]
	for i in os.walk(source_dir):
		for j in i[-1]: 
			is_valid = True
			for exc in ignore_folders_contain: 
				if not j.endswith(".py") or exc in i[0] or exc in j:
					is_valid = False
			if is_valid:
				files_set = (i[0],j)
				code_files.append(files_set)

	for dirn,file in code_files:
		dstdir = os.path.join(code_path,os.path.relpath(dirn,source_dir))
		if not os.path.exists(dstdir):
			os.makedirs(dstdir)
		filepath = os.path.join(dirn,file)
		fullpath = os.path.join(dstdir,file) #remove the period, which will always be the 0th element
		shutil.copyfile(filepath,fullpath)
	return code_path



def get_run_parameters():
	##################
	# set parameters #
	##################
	parameters = OrderedDict(
		num_latents = [8],
		beta = ["annealed"], # will be overwritten
		#beta = [400,350,300,250,175,150,125,100],
		random_seed = [1,5,10], #Using BetaVAE
		model_size=[1],
		)
	#parameters['hparam_schedule'] = [lambda step: ep.hparam_schedule_alpha_beta(step, 4, final_beta=j) for j in parameters["beta"]]
	#parameters['hparam_schedule'] = [
	#	lambda step: ep.hparam_schedule_alpha3(step,3)]
	###########################################################
	# WARNING: hparam_schedule_alpha_beta3 is last layer only #
	###########################################################
	num_latent_layers=4

	parameters['hparam_schedule'] = [
		hp.network.GradualCondKLDTrigger(
			num_layers=num_latent_layers, num_child=2, gp_kw={
					"duration":2000,"start_val":0,"final_val":0.1,"start_step":5000}, layerhps = [
				None,
				None,
				hp.layers.SpecifiedBetaHold(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=0, 
					start_step=5000, converge_beta=300, kld_detection_threshold=1),
				hp.layers.SpecifiedBetaHold(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=0, 
					start_step=5000, converge_beta=400, kld_detection_threshold=1),
			]
		),
		hp.network.GradualCondKLDTrigger(
			num_layers=num_latent_layers, num_child=2, gp_kw={
					"duration":2000,"start_val":0,"final_val":0.1,"start_step":5000}, layerhps = [
				None,
				None,
				hp.layers.SpecifiedBetaHold(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=0, 
					start_step=5000, converge_beta=200, kld_detection_threshold=1),
				hp.layers.SpecifiedBetaHold(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=0, 
					start_step=5000, converge_beta=200, kld_detection_threshold=1),
			]
		),
		hp.network.GradualCondKLDTrigger(
			num_layers=num_latent_layers, num_child=2, gp_kw={
					"duration":2000,"start_val":0,"final_val":0.25,"start_step":5000}, layerhps = [
				None,
				None,
				hp.layers.SpecifiedBetaHold(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=0, 
					start_step=5000, converge_beta=200, kld_detection_threshold=1),
				hp.layers.SpecifiedBetaHold(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=0, 
					start_step=5000, converge_beta=200, kld_detection_threshold=1),
			]
		),

	]

	return parameters



def run_models(parameters=None):
	if parameters is None:
		parameters=get_run_parameters()

	if not "COMPVAE_EXPERIMENT_BASEPATH" in os.environ:
		base_path = os.getcwd()
	else:
		base_path=os.environ["COMPVAE_EXPERIMENT_BASEPATH"]
	base_path = os.path.join(base_path,"experiments/shapes3d/multilayer/conditioning_gradual_lg2/")
	parallel_run = ParallelProcess(max_concurrent_procs_per_gpu=1,num_gpu=2)
	
	if os.path.exists(base_path):
		if not "y" in input("do you want to use existing path?"):
			exit()
	
	# do this to keep snapshot of code to run parallel processes.
	# we can't parallelize this code with multiprocess because of the pickling so we need to use subprocess and shells
	code_path = copy_code(base_path, source_dir=os.path.relpath(pathlib.Path(__file__).parent.absolute()))
	os.chdir(code_path)
	base_path = ".."


	# create experiment path:
	kwargs_set = mix_parameters(copy.deepcopy(parameters),enumerated=True)
	sub_folder = ["beta", "random_seed","num_latents"] # parameters for subfolders
	non_sub_folder = [i for i in parameters.keys() if not i in sub_folder]
	folder_index = {}
	i = 0
	for runset in kwargs_set:
		index_set = tuple([runset[i][0] for i in non_sub_folder])
		if not index_set in folder_index:
			folder_index[index_set] = i
			i+=1
		sf = ["%s_%s"%(i, runset[i][1]) for i in sub_folder] 
		for k,v in runset.items(): runset[k]=v[1]
		runset["base_path"] = os.path.join(base_path, "exp_"+str(folder_index[index_set]), *sf)
	
	##################
	# run processing #
	##################
	parallel_run.run(kwargs_set)

import time
import subprocess
class ParallelProcess():
	def __init__(self, max_concurrent_procs_per_gpu=float("inf"),num_gpu=1):
		self._exec_path = None
		self.queue_path="queue"
		self.max_concurrent_procs_per_gpu=max_concurrent_procs_per_gpu
		self.num_gpu = num_gpu

	def run(self, kwargs_set):
		procs=[[] for i in range(self.num_gpu)]
		idx=0
		finished = False
		while not finished:
			for cur_gpu in range(self.num_gpu):
				if not idx >=len(kwargs_set) and len(procs[cur_gpu])<self.max_concurrent_procs_per_gpu:
					envar=os.environ.copy()
					envar["CUDA_VISIBLE_DEVICES"] = str(cur_gpu)
					procs[cur_gpu].append(self.setup(kwargs_set[idx],envar=envar))
					time.sleep(0.5)#sleep for queue creation
					idx+=1
			try:
				finished=True
				for cur_gpu in range(self.num_gpu):
					procs[cur_gpu]=[i for i in procs[cur_gpu] if i.poll() is None]
					finished = finished and len(procs[cur_gpu])==0 and idx >= len(kwargs_set)
			except Exception as e:
				for cur_gpu in range(self.num_gpu):
					for i in procs[cur_gpu]:
						while not i.poll() is None:
							i.terminate()
				shutil.rmtree(self.exec_path)
				raise e
		print("Finished")
		shutil.rmtree(self.exec_path)

	@property
	def exec_path(self):
		if self._exec_path is None:
			num=0
			if os.path.exists(self.queue_path):
				num=len([i for i in os.listdir(self.queue_path) if i.startswith(
					self.queue_path) and os.path.isdir(os.path.join(self.queue_path,i))])
			self._exec_path=os.path.join(self.queue_path,self.queue_path+str(num))
		return self._exec_path

	def setup(self, kw, envar=None):
		if not os.path.exists(self.exec_path): os.makedirs(self.exec_path)
		name = "params_%d.pickle"%len(os.listdir(self.exec_path))
		path=os.path.join(self.exec_path,name)
		with open(path,"wb") as f:
			pickle.dump(kw,f)
		if envar is None:
			envar=os.environ.copy()
		proc=subprocess.Popen(["python%d.%d"%sys.version_info[:2],"execute.py",path],env=envar)
		return proc

	@classmethod
	def execute(cls, path):
		with open(path,"rb") as f:
			kw=pickle.load(f)
		print("running",path)
		run_training(**kw)
		os.remove(path)

import sys
def main():
	args=sys.argv
	if len(args)>1:
		ParallelProcess.execute(args[1])
	else:
		run_models()


if __name__ == '__main__':
	main()
