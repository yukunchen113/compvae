import copy
import core.config as cfg
import shutil
import time
import multiprocessing
from core.model.handler import ModelHandler, DualModelHandler
import os

def randomize_mask_step_experiment(rootpath = "exp"):
	# use pretrained mask
	mask_model_path = os.path.join(rootpath, "mask_only/beta_30")
	mask_config = cfg.config.Config64()
	mask_config.beta_value = 30
	mask_handler = ModelHandler(mask_model_path, mask_config)


	base_path = os.path.join(rootpath, "train_both/randomized_mask_step")
	# use mask config above
	#mask_config = cfg.config.Config64()
	#mask_config.beta_value = 30
	comp_config = cfg.config.Config256()
	comp_config.beta_value = 500
	comp_config.mask_latent_of_focus = 3
	model_handler = DualModelHandler(base_path, mask_config, comp_config, randomize_mask_step=True, is_combine_models=False)
	
	# copy the previously trained mask file
	shutil.copy(mask_handler.model_save_file, model_handler.mask_mh.model_save_file)
	
	# activate the models, save the model configurations, and start training.
	model_handler.setup_combination_model()
	model_handler.save()
	model_handler.train()

def pretrain_experiment(mbeta, mrandom_seed, mis_train, cbeta, clof, crandom_seed, rootpath="exp"):
	# we train it here so we keep these parameters
	mask_model_path = os.path.join(rootpath, "mask_only/beta_%d_seed_%d"%(mbeta, mrandom_seed))
	mask_config = cfg.config.Config64()
	mask_config.beta_value = mbeta
	mask_config.random_seed = mrandom_seed
	mask_handler = ModelHandler(mask_model_path, mask_config)

	if mis_train:
		train_path = "train_both"
	else:
		train_path = "comp_only"
	base_path = os.path.join(rootpath, "pretrain_mask", train_path, "cbeta_%d_seed_%d_clof_%d"%(cbeta, crandom_seed, clof))
	comp_config = cfg.config.Config256()
	comp_config.beta_value = cbeta
	comp_config.mask_latent_of_focus = clof #int(input("please enter latent of focus: ")) # pause for analysis
	comp_config.random_seed = crandom_seed

	# run model without combining (this will stop the activation of model). Model activation include loading previous models.
	# this will only create directories.
	mask_config.is_train=mis_train
	model_handler = DualModelHandler(base_path, mask_config, comp_config, train_new=False, is_combine_models=False)
	
	# copy the previously trained mask file
	shutil.copy(mask_handler.model_save_file, model_handler.mask_mh.model_save_file)

	# activate the models, save the model configurations, and start training.
	model_handler.setup_combination_model()
	model_handler.save()
	model_handler.train()

def run_mask(beta=30, random_seed=1, rootpath = "exp"):
	# we train it here so we keep these parameters
	
	mask_model_path = os.path.join(rootpath, "mask_only/beta_%d_seed_%d"%(beta, random_seed))
	print("running %s"%mask_model_path)
	mask_config = cfg.config.Config64()
	mask_config.beta_value = beta
	mask_config.random_seed = random_seed
	mask_handler = ModelHandler(mask_model_path, mask_config,train_new=True)
	mask_handler.save()
	mask_handler.train()

def run_masks():
	betas = [20,25,30]
	random_seed = [1,10,25,30]
	rootpath = "exp2"
	args = [(b,r,rootpath) for b in betas for r in random_seed]
	for k in args:
		run_mask(*k)

def run_comp():
	mbeta = 25
	mrandom_seed = 1
	mis_train = [True, False]
	cbeta = 500
	clof = 7
	crandom_seed = 1
	rootpath = "exp2"

	args = [(mbeta, mrandom_seed, i, cbeta, clof, crandom_seed, rootpath) for i in mis_train]
	with multiprocessing.Pool(2) as pool:
		pool.starmap(pretrain_experiment, args)

if __name__ == '__main__':
	run_masks()

	#randomize_mask_step_experiment()