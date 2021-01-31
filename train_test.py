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
import metrics as mt
import time
#############
# Execution #
#############
def train_parallel_ladder(job_num=None):
	#######################
	# Base Parallel Setup #
	#######################
	path = "experiments_test/"
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
	def mix_params(base, additionals):
		out = []
		for i in additionals:
			for b in base:
				b=copy.deepcopy(b)
				for k,v in i.items():
					b[k] = v(b[k])
				out.append(b)
		return out

	# base jobs
	jobs = [base_kw]
	#for j in jobs: print("base",j["path"])

	# base model
	base_models = [
		# dict( # test hiershapes
		# 	path = lambda x: os.path.join(x, "vlae_custom_test/"),
		# 	Model = lambda x: custom_architecture.VLAE,
		# 	traversal = lambda x: vlae_traversal,
		# 	),
		# dict( # test hiershapes
		# 	path = lambda x: os.path.join(x, "lvae_custom_5000_step_test/"),
		# 	Model = lambda x: custom_architecture.LVAE,
		# 	traversal = lambda x: lvae_traversal,
		# 	),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "lvae_custom_large_5000_step_annealed_top_gamma_1/"),
			Model = lambda x: custom_architecture_large.LVAE,
			traversal = lambda x: lvae_traversal,
			),
		]
	jobs = mix_params(jobs, base_models)	

	# dataset
	dataset = [
		#dict( # test celeba dataset
		#	path = lambda x: os.path.join(x, "celeba"),
		#	dataset = lambda x: ds.CelebA(),
		#	),
		# dict( # test shapes3d dataset
		# 	path = lambda x: os.path.join(x, "shapes3d"),
		# 	dataset = lambda x: ds.Shapes3D(),
		# 	),
		# dict( # test hiershapes
		# 	path = lambda x: os.path.join(x, "hiershapes/boxhead_07"),
		# 	dataset = lambda x: ds.HierShapesBoxhead(port=65333),
		# 	),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "hiershapes/boxheadsimple"), 
			dataset = lambda x: ds.HierShapesBoxheadSimple(port=65334),
			),
		# dict( # test hiershapes
		# 	path = lambda x: os.path.join(x, "hiershapes/boxhead_no_hierarchy_color"),
		# 	dataset = lambda x: ds.HierShapesBoxheadNoHier(port=65335), # this port should be different from other dataset ports
		# 	),
		]
	jobs = mix_params(jobs, dataset)
	#for j in jobs: print("dataset",j["path"])

	# add models
	models = [
		dict(
			path=lambda x: os.path.join(x,"base/num_latents_6"),
			model_params=lambda x: {**x,**dict(num_latents=6)},
			),
		# dict(
		# 	path=lambda x: os.path.join(x,"base/num_latents_6"),
		# 	model_params=lambda x: {**x,**dict(num_latents=6)},
		# 	),
		# dict(
		# 	path=lambda x: os.path.join(x,"subspace/num_latents_4"),
		# 	model_processing=lambda x: [md.LatentSubspaceLVAE([[0,0,2,2],[0,0,2,2]])],	
		# 	model_params=lambda x: {**x,**dict(num_latents=4)},
		# 	),
		#dict(
		#	path=lambda x: os.path.join(x,"snorm/num_latents_6"),
		#	model_processing=lambda x: [md.StandardNormalBetaLVAE()],	
		#	), 
		#dict(
		##	path=lambda x: os.path.join(x,"snorm/num_latents_9"),
		#	model_processing=lambda x: [md.StandardNormalBetaLVAE()],	
		#	model_params=lambda x: {**x,**dict(num_latents=9)},
		#	), 
		# dict(
		# 	path=lambda x: os.path.join(x,"subspace/num_latents_12"),
		# 	Model=lambda x: subspace(mask(x)),
		# 	model_params=lambda x: {**x,**dict(num_latents=12)},
		# 	)
		]
	jobs = mix_params(jobs, models)
	#for j in jobs: print("models",j["path"])

	# add hparams
	hparams = [
		dict(path=lambda x: os.path.join(x,"beta_15/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=3,
					alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
					layerhps = [
						hp.layers.NoOp(beta=15),
						hp.layers.NoOp(beta=15),
						hp.layers.LinearBeta(duration=5000, start_val=75, final_val=15, start_step=0),
					]
				),
			),
		# dict(path=lambda x: os.path.join(x,"beta_20/alpha_scheduled"),
		# 	hparam_scheduler= lambda x: hp.network.StepTrigger(
		# 			num_layers=3,
		# 			alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000},
		# 			layerhps = [
		# 				hp.layers.NoOp(beta=20),
		# 				hp.layers.NoOp(beta=20),
		# 				hp.layers.LinearBeta(duration=5000, start_val=75, final_val=20, start_step=0),
		# 			]
		# 		),
		# 	),
		]
	jobs = mix_params(jobs, hparams)
	#for j in jobs: print("hparams",j["path"])

	# add random seeds
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
			path=lambda x: os.path.join(x,"random_seed_100"),
			random_seed=lambda x:100
			),
		]
	jobs = mix_params(jobs, random_seeds)
	#for j in jobs: print("random seeds",j["path"])

	#######################
	# Submit and Run Jobs #
	#######################
	if job_num is None: 
		parallel = ParallelRun(exec_file=os.path.abspath(__file__), job_path=jobs_path, num_gpu=2, max_concurrent_procs_per_gpu=1)
		parallel(*[str(i) for i in range(len(jobs))])
	else:
		run_training(**jobs[job_num])

def train_parallel_single(job_num=None):
	time.sleep(2)
	#######################
	# Base Parallel Setup #
	#######################
	path = "experiments_test/"
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
	def mix_params(base, additionals):
		out = []
		for i in additionals:
			for b in base:
				b=copy.deepcopy(b)
				for k,v in i.items():
					b[k] = v(b[k])
				out.append(b)
		return out

	# base jobs
	jobs = [base_kw]
	#for j in jobs: print("base",j["path"])

	# base model
	base_models = [
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "betavae_custom_large/"),
			Model = lambda x: custom_architecture_large.BetaVAE,
			traversal = lambda x: vlae_traversal,
			),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "betatcvae_custom_large/"),
			Model = lambda x: custom_architecture_large.BetaTCVAE,
			traversal = lambda x: vlae_traversal,
			),
		]
	jobs = mix_params(jobs, base_models)	

	hsdataset = ds.HierShapesBoxheadSimple(port=65335)

	# dataset
	dataset = [
		#dict( # test celeba dataset
		#	path = lambda x: os.path.join(x, "celeba"),
		#	dataset = lambda x: ds.CelebA(),
		#	),
		# dict( # test shapes3d dataset
		# 	path = lambda x: os.path.join(x, "shapes3d"),
		# 	dataset = lambda x: ds.Shapes3D(),
		# 	),
		# dict( # test hiershapes
		# 	path = lambda x: os.path.join(x, "hiershapes/boxhead_07"),
		# 	dataset = lambda x: ds.HierShapesBoxhead(port=65333),
		# 	),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "hiershapes/boxheadsimple"), 
			dataset = lambda x: hsdataset,
			),
		# dict( # test hiershapes
		# 	path = lambda x: os.path.join(x, "hiershapes/boxheadsimple2"), 
		# 	dataset = lambda x: ds.HierShapesBoxheadSimple2(port=65337),
		# 	),
		]
	jobs = mix_params(jobs, dataset)
	#for j in jobs: print("dataset",j["path"])

	# add models
	models = [
		dict(
			path=lambda x: os.path.join(x,"base/num_latents_12"),
			model_params=lambda x: {**x,**dict(num_latents=12)},
			),
		]
	jobs = mix_params(jobs, models)
	#for j in jobs: print("models",j["path"])

	# add hparams
	hparams = [
		dict(path=lambda x: os.path.join(x,"beta_15_annealed_10000/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [
						hp.layers.LinearBeta(duration=10000, start_val=200, final_val=15, start_step=0),
					]
				),
			),
		dict(path=lambda x: os.path.join(x,"beta_15/alpha_scheduled"),
			hparam_scheduler= lambda x: hp.network.StepTrigger(
					num_layers=1,
					layerhps = [
						hp.layers.NoOp(beta=15),
					]
				),
			),
		]
	jobs = mix_params(jobs, hparams)
	#for j in jobs: print("hparams",j["path"])

	# add random seeds
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
			path=lambda x: os.path.join(x,"random_seed_30"),
			random_seed=lambda x:30
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
	jobs = mix_params(jobs, random_seeds)
	#for j in jobs: print("random seeds",j["path"])

	#######################
	# Submit and Run Jobs #
	#######################
	if job_num is None: 
		parallel = ParallelRun(exec_file=os.path.abspath(__file__), job_path=jobs_path, 
			num_gpu=2, max_concurrent_procs_per_gpu=3)
		parallel(*[str(i) for i in range(len(jobs))])
		hsdataset.close(is_terminate_server=True)
	else:
		run_training(**jobs[job_num])

################
# Used Objects #
################
def run_training(path, dataset, train_dataset_params, random_seed, model_params, Model, training_params, traversal, 
	model_processing=[],dataset_random_seed=1,
	hparam_scheduler=None, final_step=1000, modelsavestep=500, imagesavestep=1000, visualized_images=[0]):
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

			#mt.save_mi_lvae(model, os.path.join(misavepath, "%d.png"%step))
			#mt.save_irs(model, path=irssavepath, filename="%d.png"%step)
	if hasattr(dataset, "close"): dataset.close()

	# run metric:


import sys
if __name__ == '__main__':
	args=sys.argv
	if len(args)>1:
		#train_parallel_ladder(int(args[1]))
		train_parallel_single(int(args[1]))
	else:
		#train_parallel_ladder(None)
		train_parallel_single(None)