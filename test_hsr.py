import hsr.dataset as ds
def test_dataset():
	dataset = ds.CelebA()
	print(dataset.preprocess(dataset.test()).shape)
	print(dataset.preprocess(dataset.train(32)).shape)

# test model save and load
# test saver
import os
from hsr.model.vae import LVAE,VLAE
from hsr.save import InitSaver, ModelWeightSaver, ModelSaver
from hsr.utils.regular import cprint
def test_model_save():
	savedir = "dev/model_save_load/"
	if not os.path.exists(savedir): os.makedirs(savedir)

	modelsaver = ModelSaver(savedir)

	dataset = ds.CelebA()
	data = dataset.preprocess(dataset.test()[:32])

	Model = modelsaver(LVAE) # wrap LVAE so it saves initialization parameters
	model = Model()
	out = model(data)
	print(out.shape)
	modelsaver.save(model)

	model = modelsaver.load()
	out = model(data)
	print(out.shape)

import custom_architecture
def test_vlae():
	dataset = ds.HierShapesBoxhead(False)
	data = dataset.preprocess(dataset.test()[:32])
	model = custom_architecture.VLAE(num_latents = 10)
	for i in range(2):
		out = model(data)
	print(out.shape)

import numpy as np
import tensorflow as tf
import sys
import hsr.metrics as mt
def test_metrics(model_num=None):
	######################
	# Get Model and Data #
	######################
	path = "experiments/"
	paths = []
	for base,folders,files in os.walk(path):
		if "model" in folders:
			paths.append(os.path.join(base,"model"))
	paths.sort()
	if model_num is None:
		for i,p in enumerate(paths): print(i,":",p)
		return None
	path = paths[model_num]
	cprint.blue("selected:", path)

	# dataset params #
	tf.random.set_seed(1)
	#dataset = ds.Shapes3D()
	dataset = ds.HierShapesBoxhead(use_server=False)
	#dataset = ds.CelebA()

	# create model #
	modelsaver = ModelSaver(path)
	model = modelsaver.load()
	assert not model is None, f"No model found in {path}"

	return model, dataset




if __name__ == '__main__':
	args=sys.argv
	if len(args)>1:
		model_num=int(args[1])
	else:
		model_num=None
	test_metrics(model_num)
