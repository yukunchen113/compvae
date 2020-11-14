import os
import copy
import utilities.constants as cn
import numpy as np
import shutil
import dill as pickle
import sys
import time
import subprocess

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

def run_models(parallel_run, parameters=None, base_path=None, sub_folder=[],source_dir=None):
	assert not source_dir is None, "source_dir must be specified."
	if os.path.exists(base_path):
		answer = input("do you want to use existing path?")
		if "y" in answer:
			pass
		elif "-rm" == answer:
			shutil.rmtree(base_path)
		else:
			exit()
	
	# do this to keep snapshot of code to run parallel processes.
	# we can't parallelize this code with multiprocess because of the pickling so we need to use subprocess and shells		
	code_path = copy_code(base_path, source_dir=source_dir)
	os.chdir(code_path)
	base_path = ".."


	# create experiment path:
	kwargs_set = mix_parameters(copy.deepcopy(parameters),enumerated=True)
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

class ParallelProcessBase():
	def __init__(self, execute_file,max_concurrent_procs_per_gpu=float("inf"),num_gpu=1):
		self._exec_path = None
		self.queue_path="queue"
		self.max_concurrent_procs_per_gpu=max_concurrent_procs_per_gpu
		self.num_gpu = num_gpu
		self.execute_file=execute_file

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
		proc=subprocess.Popen(["python%d.%d"%sys.version_info[:2],self.execute_file,path],env=envar)
		return proc

	@classmethod
	def execute(cls, path):
		with open(path,"rb") as f:
			kw=pickle.load(f)
		print("running",path)
		cls.run_training(**kw)
		os.remove(path)

	@classmethod
	def run_training(cls,*ar,**kw):
		raise Exception("Internal error: undefined run_training")
