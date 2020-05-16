import copy
import core.config as cfg
import shutil
import time
import multiprocessing
from core.model.handler import ModelHandler, DualModelHandler

def randomize_mask_step_experiment():
	# use pretrained mask
	mask_model_path = "exp/mask_only/beta_30"
	mask_config = cfg.config.Config64()
	mask_config.beta_value = 30
	mask_handler = ModelHandler(mask_model_path, mask_config)


	base_path = "exp/train_both/randomized_mask_step"
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

def pretrain_experiment():
	# we train it here so we keep these parameters
	mask_model_path = "exp/mask_only/beta_30"
	mask_config = cfg.config.Config64()
	mask_config.beta_value = 30
	mask_handler = ModelHandler(mask_model_path, mask_config)
	#mask_handler.save()
	#mask_handler.train()

	base_path = "exp/test"
	# use mask config above
	#mask_config = cfg.config.Config64()
	#mask_config.beta_value = 30
	comp_config = cfg.config.Config256()
	comp_config.beta_value = 500
	comp_config.mask_latent_of_focus = 8#int(input("please enter latent of focus: ")) # pause for analysis


	# run model without combining (this will stop the activation of model). Model activation include loading previous models.
	# this will only create directories.
	mask_config.is_train=False
	model_handler = DualModelHandler(base_path, mask_config, comp_config, train_new=False, is_combine_models=False)
	
	# copy the previously trained mask file
	shutil.copy(mask_handler.model_save_file, model_handler.mask_mh.model_save_file)

	# activate the models, save the model configurations, and start training.
	model_handler.setup_combination_model()
	model_handler.save()
	model_handler.train()

def run_masks(beta=30, random_seed=1):
	# we train it here so we keep these parameters
	mask_model_path = "exp/mask_only/beta_%d_seed_%d"%(beta, random_seed)
	print("running %s"%mask_model_path)
	mask_config = cfg.config.Config64()
	mask_config.beta_value = beta
	mask_config.random_seed = random_seed
	mask_handler = ModelHandler(mask_model_path, mask_config)
	mask_handler.save()
	mask_handler.train()

if __name__ == '__main__':

	betas = [20,25,30]
	random_seed = [1,10,25,30]
	
	args = [(b,r) for b in betas for r in random_seed]
	with multiprocessing.Pool(5) as pool:
		pool.starmap(run_masks, args)

	#randomize_mask_step_experiment()