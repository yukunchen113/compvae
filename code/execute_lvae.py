import os
from collections import OrderedDict
from utilities.model_multiprocess import train_wrapper, run_models, ParallelProcessBase
import numpy as np
import utilities.hparams as hp
import pathlib
import sys

def create_model(base_path, model_size, **kw):
	import core.config as cfg 
	from core.model.handler import LadderModelHandler
	import tensorflow as tf
	print("Running %s"%base_path)
	config = cfg.config.Config()
	for k,v in kw.items():
		setattr(config, k, v)

	config.optimizer = tf.keras.optimizers.Adamax(learning_rate=0.0003)

	if model_size:
		config_processing = cfg.addition.make_lvae_large
	else:
		config_processing = cfg.addition.make_lvae_small
	modhand = LadderModelHandler(config=config, base_path=base_path, config_processing=config_processing)
	return modhand

def run_training(base_path, gpu_num=0, **kw):
	#os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
	modhand = create_model(base_path=base_path, **kw)
	modhand.save()
	train_wrapper(modhand.train)

def get_run_parameters():
	##################
	# set parameters #
	##################
	parameters = OrderedDict(
		random_seed = [1,5,10], 
		num_latents = [12],
		beta=[1,5,10],
		model_size=[0],
		is_sample=[False],
		)
	# num_latent_layers = 3
	# parameters['hparam_schedule'] = [
	# 	hp.network.StepTrigger(
	# 		num_layers=num_latent_layers,
	# 		alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
	# 		layerhps = [
	# 			hp.layers.NoOp(beta=20),
	# 			hp.layers.NoOp(beta=20),
	# 			hp.layers.NoOp(beta=20),
	# 		]
	# 	),
	# 	hp.network.StepTrigger(
	# 		num_layers=num_latent_layers,
	# 		alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
	# 		layerhps = [
	# 			hp.layers.NoOp(beta=10),
	# 			hp.layers.NoOp(beta=10),
	# 			hp.layers.NoOp(beta=10),
	# 		]
	# 	),
	# 	hp.network.StepTrigger(
	# 		num_layers=num_latent_layers,
	# 		alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
	# 		layerhps = [
	# 			hp.layers.NoOp(beta=1),
	# 			hp.layers.NoOp(beta=1),
	# 			hp.layers.NoOp(beta=1),
	# 		]
	# 	),
	# ]
	


	if not "COMPVAE_EXPERIMENT_BASEPATH" in os.environ:
		base_path = os.getcwd()
	else:
		base_path=os.environ["COMPVAE_EXPERIMENT_BASEPATH"]
	base_path = os.path.join(base_path,"experiments/lvae/celeba/base")
	
	return parameters,base_path


class ParallelProcess(ParallelProcessBase):
	@classmethod
	def run_training(cls,*ar,**kw):
		return run_training(*ar,**kw)

import sys
def main():
	args=sys.argv
	if len(args)>1:
		ParallelProcess.execute(args[1])
	else:
		parameters, base_path =get_run_parameters()
		sub_folder = ["random_seed","num_latents"] # parameters for subfolders
		execute_file = os.path.basename(__file__)#"execute.py"

		parallel_run = ParallelProcess(
			execute_file=execute_file,
			max_concurrent_procs_per_gpu=3,
			num_gpu=2)
		
		run_models(
			parallel_run=parallel_run,
			parameters=parameters,
			base_path=base_path,
			sub_folder=sub_folder,
			source_dir=os.path.relpath(pathlib.Path(__file__).parent.absolute())	
			)


if __name__ == '__main__':
	main()
