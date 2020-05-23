import copy
import shutil
import time
import multiprocessing
import os
import numpy as np
import utils as ut
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

def train_wrapper(func):
	return func()

def pretrain_experiment(mbeta, mrandom_seed, mis_train, cbeta, clof, crandom_seed, is_tcvae=True, randmask=False, rootpath="exp", is_pretrain=True, is_train=True):
	if mbeta is None:
		return None 


	# we import this here as these files import tensorflow, which should be imported after the fork while multiprocessing.
	import core.config as cfg
	from core.model.handler import ModelHandler, DualModelHandler
	#path construction
	mask_extra_path_options = []
	if is_tcvae:
		mask_extra_path_options.append("tcvae")
	else:
		mask_extra_path_options.append("betavae")

	mask_model_path = os.path.join(rootpath, "mask_only",*mask_extra_path_options,"beta_%d_seed_%d"%(mbeta, mrandom_seed))
	

	extra_path_options = []
	if mis_train:
		extra_path_options.append("train_both")
	else:
		extra_path_options.append("comp_only")
	if randmask:
		extra_path_options.append("randomized_mask_step")
	else:
		extra_path_options.append("constant_mask_step")
	base_path = os.path.join(rootpath, "pretrain_mask", *mask_extra_path_options,"beta_%d_seed_%d"%(mbeta, mrandom_seed), *extra_path_options, "cbeta_%d_seed_%d_clof_%d"%(cbeta, crandom_seed, clof))
	print("RUNNING:", base_path)	

	# we train it here so we keep these parameters
	mask_config = cfg.config.Config64()
	mask_config.beta_value = mbeta
	mask_config.random_seed = mrandom_seed
	mask_handler = ModelHandler(mask_model_path, mask_config)

	comp_config = cfg.config.Config256()
	comp_config.beta_value = cbeta
	comp_config.mask_latent_of_focus = clof #int(input("please enter latent of focus: ")) # pause for analysis
	comp_config.random_seed = crandom_seed

	# run model without combining (this will stop the activation of model). Model activation include loading previous models.
	# this will only create directories.
	mask_config.is_train=mis_train
	model_handler = DualModelHandler(base_path, mask_config, comp_config, train_new=False, is_combine_models=False, randomize_mask_step=randmask)
	
	# copy the previously trained mask file
	if is_train and is_pretrain and os.path.exists(mask_handler.model_save_file):
		shutil.copy(mask_handler.model_save_file, model_handler.mask_mh.model_save_file)

	# activate the models, save the model configurations, and start training.
	model_handler.setup_combination_model()
	
	if is_train:
		model_handler.save()
		train_wrapper(model_handler.train)
	return model_handler

def run_mask(beta=30, random_seed=1, rootpath = "exp", is_tcvae=True, train=True):
	# we import this here as these files import tensorflow, which should be imported after the fork while multiprocessing.
	import core.config as cfg
	from core.model.handler import ModelHandler, DualModelHandler

	# we train it here so we keep these parameters
	extra_path_options = []
	if is_tcvae:
		extra_path_options.append("tcvae")
	else:
		extra_path_options.append("betavae")


	mask_model_path = os.path.join(rootpath, "mask_only", *extra_path_options, "beta_%d_seed_%d"%(beta, random_seed))
	print("running %s"%mask_model_path)
	mask_config = cfg.config.Config256()
	mask_config.beta_value = beta
	mask_config.random_seed = random_seed
	mask_config._get_model = ut.tf_custom.architectures.variational_autoencoder.BetaVAE

	mask_handler = ModelHandler(mask_model_path, mask_config, train_new=train)
	if train:
		mask_handler.save()
		train_wrapper(mask_handler.train)
	return mask_handler

def run_masks():
	betas = [10,25,50]
	random_seed = [1]
	rootpath = "exp2"
	args = [(b,r,rootpath) for b in betas for r in random_seed]
	with multiprocessing.Pool(1) as pool:
		pool.starmap(run_mask, args)

def run_comp():
	mbeta = [10, 25, 50]
	mrandom_seed = 1
	mis_train = [True, False]
	cbeta = 600
	clof = 4
	crandom_seed = 1
	randmask = [True, False]
	rootpath = "test/test1"
	is_tcvae = [True,False]
	is_pretrain = True
	args = [[b, mrandom_seed, it, cbeta, clof, crandom_seed, tc, rm, rootpath, is_pretrain, True] for tc in is_tcvae for it in mis_train for rm in randmask for b in mbeta]
	args[0][0] = None
	num_run = 6
	for i in range(int(len(args)/num_run)):
		start = i*num_run
		end = (i+1)*num_run
		with multiprocessing.Pool(1) as pool:
			ret = pool.starmap(pretrain_experiment, args[start:end])
		for m in ret:
			if not m is None:
				m.training_object.save_image(m.mask_mh.config.total_steps)

def test():
	import core.config as cfg
	from core.model.handler import ModelHandler, DualModelHandler
	base_path = "test/test3"
	mask_config = cfg.config.Config64()

	comp_config = cfg.config.Config256()
	comp_config.mask_latent_of_focus = 0

	model_handler = DualModelHandler(base_path, mask_config, comp_config, train_new=False, is_combine_models=True)
	model_handler.save()
	

if __name__ == '__main__':
	test()

	#randomize_mask_step_experiment()