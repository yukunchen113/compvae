import tensorflow as tf
import os
from model import ModelHandler
import inspect
from utilities import Mask
import config as cfg
import fcntl
import errno
import time
import subprocess
import sys
import numpy as np
import utils as ut

num_gpu = 2
class FileLock():
	def __init__(self, file):
		self.file = file

	def lock(self):
		if type(self.file) == list:
			for i in self.file:
				self._lock(i)
		elif type(self.file) == str:
			self._lock(self.file)
		else:
			raise Exception("Can't lock type: %s"%(type(self.file)))
	
	def unlock(self):
		if type(self.file) == list:
			for i in self.file:
				self._unlock(i)
		elif type(self.file) == str:
			self._unlock(self.file)
		else:
			raise Exception("Can't lock type: %s"%(type(self.file)))
	
	def _lock(self, filename):
		lockfile = filename+"_lock"
		while True:
			while os.path.exists(lockfile):
				time.sleep(np.random.uniform(0,1))
			with open(lockfile, "w+") as f:
				time.sleep(np.random.uniform(0,1))
				if f.readlines() == []:
					f.write("a")
					break

	def _unlock(self, filename):
		lockfile = filename+"_lock"
		if os.path.exists(lockfile):
			os.remove(lockfile)

def enqueue_train(model_path, queue_file):
	with open(queue_file, "a") as f:
		f.write(model_path+"\n")
	paths = []
	with open(queue_file) as f:
		 # load paths and remove duplicates
		for line in f.readlines():
			if (not line in paths) and os.path.exists(line.replace("\n", "")):
				paths.append(line)
	with open(queue_file, "w+") as f:
		for p in paths:
			f.write(p)

def dequeue_train(queue_file):
	lock = FileLock(queue_file)
	
	while not os.path.exists(queue_file):
		time.sleep(0.1)
	lock.lock()
	paths = []
	with open(queue_file) as f:
		 # load paths and remove duplicates
		for line in f.readlines():
			if (not line in paths) and os.path.exists(line.replace("\n", "")):
				paths.append(line)
	lock.unlock()

	paths = [i.replace("\n", "") for i in paths] # filter for paths that exists and remove \n
	if paths == []:
		return 0
	# load config from path and begin training
	procs = []
	for i, path in enumerate(paths):
		a = subprocess.Popen(["python3.7 template_train.py %s %s %d"%(path, queue_file, i%num_gpu)], shell=True)
		procs.append(a)
		time.sleep(2)
	
	for a in procs:
		a.wait()
	return 1


def make_mask_only(base_path, overwrite=False, is_overwrite_train=False, mask_config=cfg.Config64()):
	"""This is the function to control execution
	
	Args:
		base_path (TYPE): Description
		kw (TYPE): Description
	"""

	# create mask model
	mask_model_path = os.path.join(base_path, "mask")
	
	maskhandler = ModelHandler(mask_model_path, mask_config, load_model=False)
	if os.path.exists(maskhandler.config_path) and overwrite:
		os.remove(maskhandler.config_path)
	if not os.path.exists(maskhandler.config_path):
		maskhandler.save()
	if (not os.path.exists(maskhandler.model_save_file)) or is_overwrite_train:
		print("model already has saved weights, overwriting.")
		enqueue_train(mask_model_path, queue_file=mqfile)
	else:
		open(mqfile, 'a').close() # must be created at least
	return mask_model_path

def make_comp_only(base_path, mask_model_path, comp_config=cfg.Config256(), comp_model_path=None):
	# create compvae config

	if comp_model_path is None:
		comp_model_path = os.path.join(base_path, 
				"beta_%d_mlof_%d"%(comp_config.beta_value, comp_config.mask_latent_of_focus))

	# comp vae processing to the config. We use this loading function instead of 
	# directly using the mask object. This is because you can't pickle a model object
	relative_path_to_mask = os.path.relpath(mask_model_path, comp_model_path)
	def comp_config_processing(config_obj, base_path):
		#base path is the base path for compvae, will be processed to be relative to mask
		mask_path = os.path.normpath(os.path.join(base_path, relative_path_to_mask))
		assert os.path.exists(mask_path), "A mask model directory not found in %s"%mask_path
		mhandler = ModelHandler(mask_path)
		mask_obj = Mask(mhandler.model, config_obj.mask_latent_of_focus)
		return cfg.make_comp_config(config_obj, mask_obj)

	comphandler = ModelHandler(comp_model_path, comp_config, load_model=False)
	comphandler.save(config_processing=comp_config_processing) # save and add comp config
	enqueue_train(comp_model_path, queue_file=cqfile)
	return comp_model_path



mqfile = "queues/mask_queue"
cqfile = "queues/comp_queue"
def main():
	if len(sys.argv)>1 and sys.argv[1] == "dequeue":
		while True:
			is_mask_available = 1
			while is_mask_available:
				is_mask_available = dequeue_train(mqfile)
			dequeue_train(cqfile)


	path = "exp/exp_"

	lock = FileLock([mqfile, cqfile])
	lock.lock()
	for i, mask_beta in enumerate([30], 1):
		p = "%s%d"%(path, i)

		mask_config=cfg.Config64()
		mask_config.beta_value = mask_beta
		mask_model_path = make_mask_only(base_path=p, overwrite=False, 
				is_overwrite_train=False, mask_config=mask_config)

		# comp
		comp_path_obj = ut.general_tools.StandardizedPaths(p)
		comp_beta = 400
		for j in [3,4]:
			comp_model_path = comp_path_obj("beta_%d_mlof_%d"%(comp_beta, j), 
					description="CompVAE model with beta=%d, conditioned on mask latent %d"%(comp_beta, j))

			comp_config = cfg.Config256()
			comp_config.beta_value = comp_beta
			comp_config.mask_latent_of_focus = j
			make_comp_only(base_path=p, mask_model_path=mask_model_path, 
					comp_config=comp_config, comp_model_path=comp_model_path)
	lock.unlock()

def quick_run():
	model_path = "exp/beta_500_256"
	config=cfg.Config256()
	config.beta_value = 500
	handler = ModelHandler(model_path, config)
	handler.save()
	handler.train()

if __name__ == '__main__':
	quick_run()