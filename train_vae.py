# training loop
# include this in copy code
import os
import hsr.model.vae as md
from hsr.save import TrainSaver, ModelSaver, PickleSaver
import hsr.dataset as ds
from hsr.train import Trainer
from hsr.visualize import vae_traversal
from hsr.utils.multiprocess import ParallelRun
import hsr.utils as ut
import tensorflow as tf
import copy
import shutil
import hsr.utils.hparams as hp
import custom_architecture
import custom_architecture_large
#############
# Execution #
#############

def train_parallel(job_num=None):
	#######################
	# Base Parallel Setup #
	#######################
	path = "experiments/"
	jobs_path = os.path.join(path, "jobs")
	base_kw = dict(
		path = path,

		# dataset params
		dataset_random_seed=1,
		train_dataset_params = dict(batch_size=32),
		dataset=None,

		# model params
		random_seed = 1,
		model_params = dict(beta=1,num_latents=6),
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
		# dict( # test hiershapes
		# 	path = lambda x: os.path.join(x, "vlae_custom/"),
		# 	Model = lambda x: custom_architecture.VLAE,
		# 	traversal = lambda x: vlae_traversal,
		# 	),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "betavae_custom/"),
			Model = lambda x: custom_architecture.BetaVAE,
			traversal = lambda x: vae_traversal,
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
		#dict( # test hiershapes
		#	path = lambda x: os.path.join(x, "hiershapes/boxhead_07"),
		#	dataset = lambda x: ds.HierShapesBoxhead(port=65334),
		#	),
		dict( # test hiershapes
			path = lambda x: os.path.join(x, "hiershapes/boxhead_09"),
			dataset = lambda x: ds.HierShapesBoxhead(port=65335), # this port should be different from other dataset ports
			),
		]
	jobs = mix_params(jobs, dataset)
	#for j in jobs: print("dataset",j["path"])

	# add models
	models = [
		dict(
			path=lambda x: os.path.join(x,"base/num_latents_12"),
			model_params=lambda x: {**x,**dict(beta=10,num_latents=12)},
			),
		dict(
			path=lambda x: os.path.join(x,"base/num_latents_12"),
			model_params=lambda x: {**x,**dict(beta=15,num_latents=12)},
			),
		dict(
			path=lambda x: os.path.join(x,"base/num_latents_12"),
			model_params=lambda x: {**x,**dict(beta=20,num_latents=12)},
			),
		dict(
			path=lambda x: os.path.join(x,"base/num_latents_12"),
			model_params=lambda x: {**x,**dict(beta=30,num_latents=12)},
			),
		]
	jobs = mix_params(jobs, models)
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
		parallel = ParallelRun(exec_file=os.path.abspath(__file__), job_path=jobs_path, num_gpu=2, max_concurrent_procs_per_gpu=3)
		parallel(*[str(i) for i in range(len(jobs))])
	else:
		run_training(**jobs[job_num])

################
# Used Objects #
################
def run_training(path, dataset, train_dataset_params, random_seed, model_params, Model, training_params, traversal, 
	model_processing=[],dataset_random_seed=1,
	hparam_scheduler=None, final_step=200000, modelsavestep=1000, imagesavestep=1000, visualized_images=[0]):
	##############
	# Initialize #
	##############
	# get save path
	datasetsavepath = os.path.join(path, "dataset.pickle")
	modelsavepath = os.path.join(path, "model")
	trainsavepath = os.path.join(path, "train")
	imagesavepath = os.path.join(path, "images")
	for i in [modelsavepath,trainsavepath,imagesavepath]: 
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
				min_value=-2.5,max_value=2.5,num_steps=30,
				return_traversal_object=True,
				hparam_obj=hparam_scheduler,
				is_sample=False).save_gif(os.path.join(imagesavepath, "%d.gif"%step))
	if hasattr(dataset, "close"): dataset.close()

import sys
if __name__ == '__main__':
	args=sys.argv
	if len(args)>1:
		train_parallel(int(args[1]))
	else:
		train_parallel(None)