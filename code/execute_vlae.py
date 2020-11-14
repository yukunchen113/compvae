import os
from collections import OrderedDict
from utilities.model_multiprocess import train_wrapper, run_models, ParallelProcessBase
import numpy as np
import utilities.hparams as hp
import pathlib
import sys

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

def get_run_parameters():
	##################
	# set parameters #
	##################
	parameters = OrderedDict(
		num_latents = [8],
		beta = ["annealed"], # will be overwritten
		#beta = [400,350,300,250,175,150,125,100],
		random_seed = [1,5,10], #Using BetaVAE
		model_size=[0],
		)
	#parameters['hparam_schedule'] = [lambda step: ep.hparam_schedule_alpha_beta(step, 4, final_beta=j) for j in parameters["beta"]]
	#parameters['hparam_schedule'] = [
	#	lambda step: ep.hparam_schedule_alpha3(step,3)]
	###########################################################
	# WARNING: hparam_schedule_alpha_beta3 is last layer only #
	###########################################################
	num_latent_layers=3

	parameters['hparam_schedule'] = [
		hp.network.CondKLDTrigger(
			num_layers=num_latent_layers,
			num_child=2, 
			routing_start_step=2000,
			#mask_start_step=np.inf,
			layerhps = [
				None,
				hp.layers.SpecifiedBetaHold(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=5000, 
					start_step=5000, converge_beta=400, kld_detection_threshold=2),
				hp.layers.SpecifiedBetaHold(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=5000, 
					start_step=5000, converge_beta=400, kld_detection_threshold=2),
			]
		),


	]

	if not "COMPVAE_EXPERIMENT_BASEPATH" in os.environ:
		base_path = os.getcwd()
	else:
		base_path=os.environ["COMPVAE_EXPERIMENT_BASEPATH"]
	base_path = os.path.join(base_path,"experiments/shapes3d/multilayer/conditioning_residual/")
	
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
		sub_folder = ["beta", "random_seed","num_latents"] # parameters for subfolders
		execute_file = os.path.basename(__file__)#"execute.py"

		parallel_run = ParallelProcess(
			execute_file=execute_file,
			max_concurrent_procs_per_gpu=5,
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
