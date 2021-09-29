# training loop
# include this in copy code
import os
import hsr.model.vae as md
from hsr.save import TrainSaver, ModelSaver, PickleSaver
import hsr.dataset as ds
from hsr.train import Trainer
from hsr.visualize import lvae_traversal, vlae_traversal
from hsr.utils.multiprocess import ParallelRun
import hsr.utils as ut
import tensorflow as tf
import copy
import shutil
import hsr.utils.hparams as hp
import custom_architecture
import custom_architecture_large
import custom_architecture_larger
import custom_architecture_larger2
import custom_architecture_larger3
import metrics as mt
import time
import sys
# shared parameters
experimental_path = "experiments/"
random_seeds = [
	dict(
		path=lambda x: os.path.join(x,"random_seed_1"),
		random_seed=lambda x:1
		),
	dict(
		path=lambda x: os.path.join(x,"random_seed_10"),
		random_seed=lambda x:10
		), 
	dict(
		path=lambda x: os.path.join(x,"random_seed_15"),
		random_seed=lambda x:15
		), 
	dict(
		path=lambda x: os.path.join(x,"random_seed_30"),
		random_seed=lambda x:30
		), 
	dict(
		path=lambda x: os.path.join(x,"random_seed_60"),
		random_seed=lambda x:60
		), 
	dict(
		path=lambda x: os.path.join(x,"random_seed_100"),
		random_seed=lambda x:100
		),
	dict(
		path=lambda x: os.path.join(x,"random_seed_300"),
		random_seed=lambda x:300
		),
	]
def get_dataset_config():
	datasets = [
		ds.HierShapesBoxhead(use_preloaded=True),
		ds.HierShapesBoxheadSimple(use_preloaded=True),
		ds.HierShapesBoxheadSimple2(use_preloaded=True)
		]
	dataset_config = [
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "hiershapes/boxhead_07"),
			dataset = lambda x: datasets[0],
			),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "hiershapes/boxheadsimple"), 
			dataset = lambda x: datasets[1],
			),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "hiershapes/boxheadsimple2"), 
			dataset = lambda x: datasets[2])
		]
	return datasets, dataset_config

def mix_params(base, additionals):
	out = []
	for i in additionals:
		for b in base:
			b=copy.deepcopy(b)
			for k,v in i.items():
				b[k] = v(b[k])
			out.append(b)
	return out


#############
# Execution #
#############
def get_large_ladder_jobs():
	#######################
	# Base Parallel Setup #
	#######################
	path = copy.deepcopy(experimental_path)
	jobs_path = os.path.join(path, "jobs")
	base_kw = dict(
		path = path,

		# dataset params
		dataset_random_seed=1,
		train_dataset_params = dict(batch_size=32),
		dataset=None,

		# model params
		random_seed = 1,
		model_params = dict(beta=1,num_latents=6,gamma=1),
		model_processing = [],
		Model = custom_architecture.LVAE,

		# training params
		training_params = dict(loss_func=ut.loss.ImageBCE(),optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0003)),
		hparam_scheduler = None, # initialized hparam_scheduler
		visualized_images = [1,5],
		traversal=None,
		)
	###################
	# Create the Jobs #
	###################
	# base jobs
	jobs = [base_kw]
	#for j in jobs: print("base",j["path"])

	# base model
	base_models = [
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "lvae_larger2/"),
			Model = lambda x: custom_architecture_larger2.LVAE,
			traversal = lambda x: lvae_traversal,
			),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "vlae_larger2/"),
			Model = lambda x: custom_architecture_larger2.VLAE,
			traversal = lambda x: vlae_traversal,
			),
		]
	jobs = mix_params(jobs, base_models)	

	# dataset
	datasets, dataset_config = get_dataset_config()
	jobs = mix_params(jobs, dataset_config)
	#for j in jobs: print("dataset",j["path"])

	# add models
	models = [
		dict(
			path=lambda x: os.path.join(x,"base/num_latents_4"),
			model_params=lambda x: {**x,**dict(num_latents=4)},
			),
		]
	jobs = mix_params(jobs, models)
	models = [
		dict(
			path=lambda x: os.path.join(x,"base/gamma_1"),
			model_params=lambda x: {**x,**dict(gamma=1)},
			),
		dict(
			path=lambda x: os.path.join(x,"base/gamma_5"),
			model_params=lambda x: {**x,**dict(gamma=5)},
			),
		]
	jobs = mix_params(jobs, models)


	# add hyperparameters
	hyperparameters = [
		dict(path=lambda x: os.path.join(x,f"beta_{1}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=1), hp.layers.NoOp(beta=1),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=1, start_step=0)]),
			),
		dict(path=lambda x: os.path.join(x,f"beta_{5}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=5), hp.layers.NoOp(beta=5),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=5, start_step=0)]),
			), 
		dict(path=lambda x: os.path.join(x,f"beta_{10}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=10), hp.layers.NoOp(beta=10),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=10, start_step=0)]),
			), 
		dict(path=lambda x: os.path.join(x,f"beta_{15}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=15), hp.layers.NoOp(beta=15),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=15, start_step=0)]),
			), 
		dict(path=lambda x: os.path.join(x,f"beta_{20}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=20), hp.layers.NoOp(beta=20),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=20, start_step=0)]),
			),
		]
	jobs = mix_params(jobs, hyperparameters)
	#for j in jobs: print("models",j["path"])

	# add random seeds
	jobs = mix_params(jobs, random_seeds)

	return datasets, jobs

def get_large_single_jobs():
	#######################
	# Base Parallel Setup #
	#######################
	path = copy.deepcopy(experimental_path)
	jobs_path = os.path.join(path, "jobs")
	base_kw = dict(
		path = path,

		# dataset params
		dataset_random_seed=1,
		train_dataset_params = dict(batch_size=32),
		dataset=None,

		# model params
		random_seed = 1,
		model_params = dict(beta=1,num_latents=12,gamma=1),
		model_processing = [],
		Model = custom_architecture.BetaVAE,

		# training params
		training_params = dict(loss_func=ut.loss.ImageBCE(),optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0003)),
		hparam_scheduler = None, # initialized hparam_scheduler
		visualized_images = [1,5],
		traversal=None,
		)
	###################
	# Create the Jobs #
	###################
	# base jobs
	jobs = [base_kw]

	# base model
	base_models = [
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "betavae_larger2/"),
			Model = lambda x: custom_architecture_larger2.BetaVAE,
			traversal = lambda x: vlae_traversal,
			),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "betatcvae_larger2/"),
			Model = lambda x: custom_architecture_larger2.BetaTCVAE,
			traversal = lambda x: vlae_traversal,
			),
		]
	jobs = mix_params(jobs, base_models)	

	# dataset
	datasets,dataset_config = get_dataset_config()
	jobs = mix_params(jobs, dataset_config)

	# add models
	models = [
		dict(
			path=lambda x: os.path.join(x,"base/num_latents_12"),
			model_params=lambda x: {**x,**dict(num_latents=12)},
			),
		]
	jobs = mix_params(jobs, models)

	# add hparams
	hparams = [
		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{1}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=1, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{1}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=1)])),

		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{5}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=5, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{5}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=5)])),
		
		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{10}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=10, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{10}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=10)])),

		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{15}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=15, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{15}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=15)])),

		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{20}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=20, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{20}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=20)])),
		]
	jobs = mix_params(jobs, hparams)
	#for j in jobs: print("hparams",j["path"])

	# add random seeds
	jobs = mix_params(jobs, random_seeds)
	#for j in jobs: print("random seeds",j["path"])

	return datasets, jobs

def get_ladder_jobs():
	#######################
	# Base Parallel Setup #
	#######################
	path = copy.deepcopy(experimental_path)
	jobs_path = os.path.join(path, "jobs")
	base_kw = dict(
		path = path,

		# dataset params
		dataset_random_seed=1,
		train_dataset_params = dict(batch_size=32),
		dataset=None,

		# model params
		random_seed = 1,
		model_params = dict(beta=1,num_latents=6,gamma=1),
		model_processing = [],
		Model = custom_architecture.LVAE,

		# training params
		training_params = dict(loss_func=ut.loss.ImageBCE(),optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0003)),
		hparam_scheduler = None, # initialized hparam_scheduler
		visualized_images = [1,5],
		traversal=None,
		)
	###################
	# Create the Jobs #
	###################
	# base jobs
	jobs = [base_kw]
	#for j in jobs: print("base",j["path"])

	# base model
	base_models = [
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "vlae/"),
			Model = lambda x: custom_architecture_large.VLAE,
			traversal = lambda x: vlae_traversal,
			),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "lvae/"),
			Model = lambda x: custom_architecture_large.LVAE,
			traversal = lambda x: lvae_traversal,
			),
		]
	jobs = mix_params(jobs, base_models)	

	# dataset
	datasets, dataset_config = get_dataset_config()
	jobs = mix_params(jobs, dataset_config)
	#for j in jobs: print("dataset",j["path"])

	# add models
	models = [
		dict(
			path=lambda x: os.path.join(x,"base/num_latents_4"),
			model_params=lambda x: {**x,**dict(num_latents=4)},
			),
		]
	jobs = mix_params(jobs, models)
	#for j in jobs: print("models",j["path"])

	# add hparams, I tried to put this into a for loop but it doesn't work, perhaps because of reference and the local function?
	hparams = [
		dict(path=lambda x: os.path.join(x,f"beta_{1}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=1), hp.layers.NoOp(beta=1),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=1, start_step=0)]),
			),
		dict(path=lambda x: os.path.join(x,f"beta_{5}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=5), hp.layers.NoOp(beta=5),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=5, start_step=0)]),
			), 
		dict(path=lambda x: os.path.join(x,f"beta_{10}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=10), hp.layers.NoOp(beta=10),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=10, start_step=0)]),
			), 
		dict(path=lambda x: os.path.join(x,f"beta_{15}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=15), hp.layers.NoOp(beta=15),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=15, start_step=0)]),
			), 
		dict(path=lambda x: os.path.join(x,f"beta_{20}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=20), hp.layers.NoOp(beta=20),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=20, start_step=0)]),
			), 
		dict(path=lambda x: os.path.join(x,f"beta_{30}_annealed/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [hp.layers.NoOp(beta=30), hp.layers.NoOp(beta=30),
						hp.layers.LinearBeta(duration=5000, start_val=150, final_val=30, start_step=0)]),
			), 
		]

	jobs = mix_params(jobs, hparams)

	# add random seeds
	jobs = mix_params(jobs, random_seeds)

	return datasets, jobs

def get_single_jobs():
	#######################
	# Base Parallel Setup #
	#######################
	path = copy.deepcopy(experimental_path)
	jobs_path = os.path.join(path, "jobs")
	base_kw = dict(
		path = path,

		# dataset params
		dataset_random_seed=1,
		train_dataset_params = dict(batch_size=32),
		dataset=None,

		# model params
		random_seed = 1,
		model_params = dict(beta=1,num_latents=12,gamma=1),
		model_processing = [],
		Model = custom_architecture.BetaVAE,

		# training params
		training_params = dict(loss_func=ut.loss.ImageBCE(),optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0003)),
		hparam_scheduler = None, # initialized hparam_scheduler
		visualized_images = [1,5],
		traversal=None,
		)
	###################
	# Create the Jobs #
	###################
	# base jobs
	jobs = [base_kw]

	# base model
	base_models = [
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "betavae/"),
			Model = lambda x: custom_architecture_large.BetaVAE,
			traversal = lambda x: vlae_traversal,
			),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "betatcvae/"),
			Model = lambda x: custom_architecture_large.BetaTCVAE,
			traversal = lambda x: vlae_traversal,
			),
		]
	jobs = mix_params(jobs, base_models)	

	# dataset
	datasets,dataset_config = get_dataset_config()
	jobs = mix_params(jobs, dataset_config)

	# add models
	models = [
		dict(
			path=lambda x: os.path.join(x,"base/num_latents_12"),
			model_params=lambda x: {**x,**dict(num_latents=12)},
			),
		]
	jobs = mix_params(jobs, models)

	# add hparams
	hparams = [
		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{1}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=1, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{1}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=1)])),

		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{5}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=5, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{5}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=5)])),
		
		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{10}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=10, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{10}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=10)])),

		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{15}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=15, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{15}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=15)])),

		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{20}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=20, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{20}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=20)])),

		# for a single beta, run the annealed and normal version
		dict(path=lambda x: os.path.join(x,f"beta_{30}_annealed/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [hp.layers.LinearBeta(duration=10000, start_val=200, final_val=30, start_step=0)])),
		dict(path=lambda x: os.path.join(x,f"beta_{30}/"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1, layerhps = [hp.layers.NoOp(beta=30)])),
		]
	jobs = mix_params(jobs, hparams)
	#for j in jobs: print("hparams",j["path"])

	# add random seeds
	jobs = mix_params(jobs, random_seeds)
	#for j in jobs: print("random seeds",j["path"])

	return datasets, jobs

################
# Used Objects #
################
def run_training(path, dataset, train_dataset_params, random_seed, model_params, Model, training_params, traversal, 
	model_processing=[],dataset_random_seed=1,
	hparam_scheduler=None, final_step=200000, modelsavestep=500, imagesavestep=1000, visualized_images=[0], metricsavestep=10000):
	##############
	# Initialize #
	##############
	# get save path
	datasetsavepath = os.path.join(path, "dataset.pickle")
	modelsavepath = os.path.join(path, "model")
	trainsavepath = os.path.join(path, "train")
	imagesavepath = os.path.join(path, "images")
	misavepath = os.path.join(path, "mutual_info_sample")
	irssavepath = os.path.join(path, "irs_sample")
	for i in [modelsavepath,trainsavepath,imagesavepath,misavepath,irssavepath]: 
		if not os.path.exists(i): os.makedirs(i)
	
	# save latest running of this script into train
	current_filepath = os.path.abspath(__file__)
	trainfiledst = os.path.join(trainsavepath, os.path.basename(__file__))
	shutil.copyfile(current_filepath, trainfiledst)
	with open(trainfiledst,"a") as f:
		f.write(f"\n#args: {sys.argv}")

	# create data
	if os.path.exists(datasetsavepath): os.remove(datasetsavepath)
	datasetsaver = PickleSaver(datasetsavepath)
	dset = datasetsaver.load()
	if dset is None:
		tf.random.set_seed(dataset_random_seed)
		datasetsaver.save(dataset)
	else:
		dataset = dset
	train_data = dataset.train(**train_dataset_params)
	test_data = dataset.test()

	# create model
	tf.random.set_seed(random_seed)
	modelsaver = ModelSaver(modelsavepath)
	model = modelsaver.load()
	if model is None:
		Model = modelsaver(Model, model_processing=model_processing)
		model = Model(**model_params)

	# create training 
	trainsaver = TrainSaver(trainsavepath)
	trainer,train_params = trainsaver.load(model=model)
	if trainer is None: 
		trainer = trainsaver(Trainer)(model=model,**training_params)

	################
	# Run Training #
	################
	step = -1 if train_params is None else train_params["step"]
	hparam_scheduler = hparam_scheduler if train_params is None or not "hparam_scheduler" in train_params else train_params["hparam_scheduler"]
	for data,_ in train_data:
		if step>=final_step: break
		# preprocess
		data = dataset.preprocess(data)
		hparams = {} if hparam_scheduler is None else hparam_scheduler(step=step, model=model)

		# train
		step = trainer(step=step, inputs=data, hparams=hparams)
		
		# save model weights and training progress
		if not step%modelsavestep:
			trainsaver.save(step=step, hparam_scheduler=hparam_scheduler)
			modelsaver.save(model)

		# logging 
		if not step%imagesavestep:
			traversal(
				model,
				dataset.preprocess(test_data[visualized_images]),
				min_value=-3,max_value=3,num_steps=30,
				return_traversal_object=True,
				hparam_obj=hparam_scheduler,
				is_sample=False).save_gif(os.path.join(imagesavepath, "%d.gif"%step))
		if not step%metricsavestep:
			pass
			#mt.run_test_batch(model, os.path.join(misavepath, "%d.npz"%step))
			#mt.save_mi_lvae(model, os.path.join(misavepath, "%d.png"%step))
			#mt.save_irs(model, path=irssavepath, filename="%d.png"%step)
	print("running MI")
	if not os.path.exists(os.path.join(misavepath, "final.npz")): mt.run_test_batch(model, os.path.join(misavepath, "final.npz"))
	if hasattr(dataset, "close"): dataset.close()

