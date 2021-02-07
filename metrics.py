import hsr.dataset as ds
import os
from hsr.model.vae import LVAE,VLAE
from hsr.save import InitSaver, ModelWeightSaver, ModelSaver
from hsr.utils.regular import cprint
import numpy as np
import tensorflow as tf
import sys
import hsr.metrics as mt
import matplotlib.pyplot as plt
import hiershapes as hs
from multiprocessing import Process
import seaborn as sns
import dill
import sklearn
np.set_printoptions(suppress=True)

class MultiProcessBatch(hs.dataset.MultiProcessBatch):
	"""run different parameters per batch
	"""
	def get_batch_multiproc(self, plotter, parameters):
		labels = [parameters() for _ in range(self.batch_size)]
		images, labels = self.get_images(labels, plotter=plotter, return_labels=True)
		self.queue.put((images, labels))
	
	def start_procs(self, parameters):
		self.processes = []
		for i,params in enumerate(parameters): 
			process = Process(target=self.get_batch_multiproc, kwargs=dict(plotter=self.plotter[i], parameters=params))
			process.start()
			self.processes.append(process)

	def __call__(self, parameters):
		self.start_procs(parameters)
		data = []# get the data from the queue first to prevent lock when joining
		for _ in self.processes:
			data.append(self.queue.get_queue(True))
		for proc in self.processes:
			proc.join()
		return data

def get_model(model_num=None):
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

	# create model #
	modelsaver = ModelSaver(path)
	model = modelsaver.load()
	assert not model is None, f"No model found in {path}"
	return model

def get_irs(model, irs_dset, batch_size):
	# prepare data for evaluation
	irs_set = []
	for igroup in irs_dset.igroups:
		latent_set = []
		for i in range(irs_dset.num_i):
			latents = []
			images = irs_dset(igroup, i, return_labels=False)
			images = np.array_split(images,np.ceil(images.shape[0]/batch_size),axis=0)
			for imgbatch in images:
				lat = model.encoder(irs_dset.dataset.preprocess(imgbatch), use_mean=True)
				lat = np.concatenate([l[1] for l in lat],axis=-1) # format latents to (batch_size, total num latents)
				latents.append(lat)
			latent_set.append(np.concatenate(latents,0)) # keep the latent set
			print(f"Finished running latents {i+1}/{irs_dset.num_i} for group {igroup}\t\t\r",end = "")
		irs_set.append((igroup, latent_set))
	
	# evaluate
	max_latents = np.concatenate([i for j in irs_set for i in j[1]],0) # normalizer
	assert len(max_latents.shape) == 2
	max_latents = np.max(irs_dset.irs.distance_func(max_latents,np.mean(max_latents,axis=0,keepdims=True)),axis=0)
	group_labels,scores = [],[]
	for igroup, latent_set in irs_set:
		score = irs_dset.irs(latent_set, max_latents)
		scores.append(score)
		group_labels.append(igroup)
	scores = np.asarray(scores)
	return scores

def mutual_information(latents):
	assert len(latents.shape) == 2
	mi_map = np.zeros((latents.shape[1], latents.shape[1]))
	for i in range(mi_map.shape[0]):
		for j in range(mi_map.shape[1]):
			#mi_map[i,j] = sklearn.metrics.mutual_info_score(latents[:,i], latents[:,j])
			#mi_map[i,j] = sklearn.metrics.adjusted_mutual_info_score(latents[:,i], latents[:,j])
			mi_map[i,j] = sklearn.metrics.normalized_mutual_info_score(latents[:,i], latents[:,j])
	return mi_map

def histogram_discretize(target, num_bins=20):
	"""Discretization based on histograms.
	Code is from disentanglement_lib
	"""
	discretized = np.zeros_like(target)
	for i in range(target.shape[1]):
		discretized[:, i] = np.digitize(target[:, i], np.histogram(
			target[:, i], num_bins)[1][:-1])
	return discretized

def save_mi_lvae(model, savepath=None, num_batches=200, batch_size=100):
	# accumulate datapoints
	print("accumulating datapoints")
	latents=[]
	for step in range(num_batches):
		np.random.seed()
		# sample from priors 
		prior_space=[None for _ in range(len(model.ladder_params)+1)]
		prior_space[-1] = list([np.random.uniform(-3,3,size=(batch_size,model.num_latents)), None, None])
		try:
			prior_space[:-1] = model.decoder(None, latent_space=prior_space, return_ls=True, use_mean=False)[1][:-1]	
		except TypeError:
			return
		# format latents
		lat = np.concatenate([i[0] for i in prior_space],-1) # getting the samples
		latents.append(lat)
		print("saving mi metric: ",step+1, end="\r")
	latents = np.concatenate(latents, 0)
	latents = histogram_discretize(latents)

	# get mutual information
	print("getting MI")
	mi_map_latents = mutual_information(latents)

	# plot map
	print("plotting")
	plt.clf()
	cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
	ax = sns.heatmap(
		mi_map_latents, 
		mask = np.eye(*mi_map_latents.shape, dtype=bool),
		cmap=cmap)
	if savepath is None:
		plt.show()
	else:
		plt.savefig(savepath)

class IRSHiershapesDataset:
	def __init__(self, name):
		path=f"dev/irs_cache/{name}.npz"
		assert path.endswith(".npz"), "must specify .npz file"
		self.irs = mt.InterventionalRobustnessScore()
		self.name = name
		self.path = path
		self.pickle_path = f"dev/irs_cache/{name}.pickle"

		# dynamic
		self.num_i = None
		self.num_j = None
		self.data = None
		self.dataset = None
		self.igroups = None

	def create(self, num_i_per_proc, num_proc, num_j, igroups, dataset):
		assert not os.path.exists(self.path)
		dirname = os.path.dirname(self.path)
		if not os.path.exists(dirname): os.makedirs(dirname)
		# use custom multiprocess
		scene, parameters = dataset.get_scene_parameters()
		dset = MultiProcessBatch(scene=scene, num_proc=num_proc, prefetch=num_proc)
		dset.batch(num_j)
		irsdataset = {}
		for igroup in igroups:
			# generate data
			_, base_parameters = dataset.get_scene_parameters(is_filter_val=False)   
			data = []
			for i in range(num_i_per_proc): # num I - for a consistent I set create random J set
				# get data
				parameters = []
				for _ in range(num_proc):
					sample_i = base_parameters() # get consistent I set
					_, param = dataset.get_scene_parameters( # consistent I set, random J set label generator
						{k:v for k,v in sample_i.items() if k in igroup}, is_filter_val=True)
					parameters.append(param)

				# generate data
				data += dset(parameters) # list of (imagebatch, labelbatch) tuples
				print(f"Generated latents {i+1}/{num_i_per_proc} for group {igroup}\r",end = "")
			irsdataset[tuple(igroup)] = data
		np.savez_compressed(self.path, num_i=num_i_per_proc*num_proc, num_j=num_j, data=irsdataset, igroups=igroups)
		with open(self.pickle_path, "wb") as f:
			dill.dump(dataset, f)
		self.load()

	def load(self):
		data = np.load(self.path, allow_pickle=True)
		self.num_i = data["num_i"].item()
		self.num_j = data["num_j"].item()
		self.data = data["data"].item()
		self.igroups = list(data["igroups"])
		with open(self.pickle_path, "rb") as f:
			self.dataset = dill.load(f)

	def __getstate__(self):
		self.num_i = None
		self.num_j = None
		self.data = None
		self.dataset = None
		self.igroups = None
		return self.__dict__

	# save igroup as tuple
	def __call__(self, igroup, i, return_labels=False):
		assert os.path.exists(self.path), "path doesn't exist"
		data = self.data[tuple(igroup)][i]
		if not return_labels: return data[0]
		return data

def main():
	for model_num in range(163,171): 
		model = get_model(model_num) # get model
		if model is None: return
		#save_irs(model, iscreate=True)
		#dataset = ds.HierShapesBoxhead(use_server=False, use_pool=False)
		#save_irs(model, dataset=dataset, path="dev/irs_test/%d"%model_num, iscreate=True, name_extension="")
		dataset = ds.HierShapesBoxheadSimple(use_server=False, use_pool=False)
		save_irs(model, dataset=dataset, path="dev/irs_test/%d"%model_num, iscreate=True, name_extension="_simple")

def save_irs(model, dataset, path=None, filename="image.png", iscreate=False, name_extension=""):
	igroups = dict(
		hierarchical_eval = [
			["color","_overall_eye_color","eye_color"],
			["scale"],
			["azimuth"],
			["_wall_color"],
			["_floor_color"]],
		independent_eval = [
			["color"],
			["_overall_eye_color"],
			["eye_color"],
			["scale"],
			["azimuth"],
			["_wall_color"],
			["_floor_color"]])
	# get dataset
	names = ["hierarchical_eval", "independent_eval"]

	for name in names:
		igroup = igroups[name]
		irs_dset = IRSHiershapesDataset(name = name+name_extension)
		if not os.path.exists(irs_dset.path):
			assert iscreate, f"{name+name_extension} not created"
			irs_dset.create(
				num_i_per_proc=10, num_proc=10, num_j=100,
				igroups=igroup,
				dataset=dataset
				)
		else:
			irs_dset.load()

		scores = get_irs(model, irs_dset, 100)
		plt.clf()
		cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
		ax = sns.heatmap(scores, cmap=cmap)
		if path is None: 
			plt.show()
		else:
			filedir = os.path.join(path, name)
			if not os.path.exists(filedir): os.makedirs(filedir)
			plt.savefig(os.path.join(filedir, filename))


if __name__ == '__main__':
	main()