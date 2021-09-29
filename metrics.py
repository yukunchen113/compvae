import hsr.dataset as ds
import os
from hsr.model.vae import LVAE,VLAE
from hsr.utils.loss import ImageBCE, kld_loss_reduction_numpy
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

def process_labels(labels):
	new_labels = []
	label_names = ["color", "scale", "ec", 'ec1', 'ec2', 'ec3', 'ec4', 'floor', 'wall', 'azimuth']
	for params in labels:
		overall_ec = np.mean(params["eye_color"]) if not "_overall_eye_color" in params else params["_overall_eye_color"]
		params = [params["color"]]+[params["scale"][0]]+[overall_ec]+list(params["eye_color"])+list(params["bg_color"])+[params["azimuth"]]
		new_labels.append(params)
	new_labels = np.asarray(new_labels)
	return new_labels, label_names

def get_model_paths(basepath = "experiments/", identifier = "model"):
	paths = []
	for base,folders,files in os.walk(basepath):
		if identifier in folders and not "archived" in base:
			paths.append(os.path.join(base, identifier))
	paths.sort()
	return paths

def get_model(model_num=None, return_path=False, basepath="experiments/"):
	######################
	# Get Model and Data #
	######################
	paths = get_model_paths(basepath = basepath)
	if model_num is None:
		for i,p in enumerate(paths): print(i,":",p)
		return None
	path = paths[model_num]
	cprint.blue("selected:", path)

	# create model #
	modelsaver = ModelSaver(path)
	model = modelsaver.load()
	assert not model is None, f"No model found in {path}"
	if return_path: return model, path
	return model

def get_irs(model, irs_dset, batch_size):
	# prepare data for evaluation
	irs_set = []
	for igroup in irs_dset.igroups:
		latent_set = []
		labels_set = []
		for i in range(irs_dset.num_i):
			latents = []
			images, labels = irs_dset(igroup, i, return_labels=True)
			labels_set.append(process_labels(labels)[0])
			images = np.array_split(images,np.ceil(images.shape[0]/batch_size),axis=0)
			for imgbatch in images:
				lat = model.encoder(irs_dset.dataset.preprocess(imgbatch), use_mean=True)
				lat = np.concatenate([l[1] for l in lat],axis=-1) # format latents to (batch_size, total num latents)
				latents.append(lat)
			latent_set.append(np.concatenate(latents,0)) # keep the latent set
			print(f"Finished running latents {i+1}/{irs_dset.num_i} for group {igroup}\t\t\r",end = "")
		irs_set.append((igroup, latent_set, labels_set))
	
	# evaluate
	cum_latents = np.concatenate([i for j in irs_set for i in j[1]],0) # used for scaling EMPIDA to IRS, these are all the latents
	cum_labels = np.concatenate([i for j in irs_set for i in j[2]], 0) # used for other forms of testing.

	assert len(cum_latents.shape) == 2
	cum_latents_mean = np.mean(cum_latents,axis=0,keepdims=True)
	max_latents = np.max(irs_dset.irs.distance_func(cum_latents,cum_latents_mean),axis=0)

	# get irs scores
	#cprint.green(f"max latents: {max_latents}")
	group_labels,scores = [],[]
	for igroup, latent_set, _ in irs_set:
		score = irs_dset.irs(latent_set, max_latents)
		scores.append(score)
		group_labels.append(igroup)
	scores = np.asarray(scores)
	return scores

def discrete_entropy(ys):
	"""Compute discrete mutual information. 
	- should calculate discrete entropy
	- code modified from: https://github.com/google-research/disentanglement_lib/blob/86a644d4ed35c771560dc3360756363d35477357/disentanglement_lib/evaluation/metrics/utils.py#L126
	"""
	num_factors = ys.shape[1]
	h = np.zeros(num_factors)
	for j in range(num_factors):
		h[j] = sklearn.metrics.mutual_info_score(ys[:,j], ys[:,j])
	return h

def mutual_information(latents, labels=None):
	if labels is None: labels = latents
	assert len(latents.shape) == 2
	mi_map = np.zeros((latents.shape[1], labels.shape[1]))
	for i in range(mi_map.shape[0]):
		for j in range(mi_map.shape[1]):
			mi_map[i,j] = sklearn.metrics.mutual_info_score(latents[:,i], labels[:,j])
			#mi_map[i,j] = sklearn.metrics.adjusted_mutual_info_score(latents[:,i], labels[:,j])
			#mi_map[i,j] = sklearn.metrics.normalized_mutual_info_score(latents[:,i], labels[:,j])
	# normalize wrt labels:
	mi_map=mi_map/discrete_entropy(labels).reshape(1,-1)
	return mi_map

def histogram_discretize(target, num_bins=30):
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

	# get mutual information
	print("getting MI")
	latents = histogram_discretize(latents)
	mi_map_latents = mutual_information(latents, latents)

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

def run_irs():
	paths = get_model_paths()
	for p in paths: print(p)
	scoresdir = "dev/scores/"
	if not os.path.exists(scoresdir): os.makedirs(scoresdir)
	total_scores = {}
	for model_num, path in list(enumerate(paths)):
		model = get_model(model_num) # get model
		if model is None: return
		if "boxhead_07/" in path:
			dataset = ds.HierShapesBoxhead(use_server=False, use_pool=False)
			scores = save_irs(model, dataset=dataset, path="dev/scores/heatmaps/", filename=f"_{model_num}.png", iscreate=True, name_extension="")
		if "boxheadsimple/" in path:
			dataset = ds.HierShapesBoxheadSimple(use_server=False, use_pool=False)
			scores = save_irs(model, dataset=dataset, path="dev/scores/heatmaps/", filename=f"_{model_num}.png", iscreate=True, name_extension="_simple")
		if "boxheadsimple2/" in path:
			dataset = ds.HierShapesBoxheadSimple2(use_server=False, use_pool=False)
			scores = save_irs(model, dataset=dataset, path="dev/scores/heatmaps/", filename=f"_{model_num}.png", iscreate=True, name_extension="_simple2")
		total_scores[path] = scores
	np.savez_compressed(os.path.join(scoresdir,"scores"), total_scores=total_scores)

def save_irs(model, dataset, path=None, filename=".png", iscreate=False, name_extension=""):
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
	irs_scores = {}
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
			dataset.terminate()
		else:
			irs_dset.load()

		scores = get_irs(model, irs_dset, 100)
		if not filename is None:
			plt.clf()
			cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
			ax = sns.heatmap(scores, cmap=cmap)
			if path is None: 
				plt.show()
			else:
				if not os.path.exists(path): os.makedirs(path)
				filedir = os.path.join(path, name+name_extension+filename)
				plt.savefig(filedir)
		irs_scores[name] = scores
	return irs_scores

def get_dataset(path):
	if "boxheadsimple/" in path: 
		dataset = ds.HierShapesBoxheadSimple(use_preloaded=True)
	elif "boxheadsimple2/" in path: 
		dataset = ds.HierShapesBoxheadSimple2(use_preloaded=True)
	elif "boxhead_07/" in path: 
		dataset = ds.HierShapesBoxhead(use_preloaded=True)
	else:
		dataset = None
	return dataset


def run_test_batch(model, path): 
	if "boxheadsimple/" in path: 
		dataset = ds.HierShapesBoxheadSimple(use_preloaded=True)
	elif "boxheadsimple2/" in path: 
		dataset = ds.HierShapesBoxheadSimple2(use_preloaded=True)
	elif "boxhead_07/" in path: 
		dataset = ds.HierShapesBoxhead(use_preloaded=True)
	else:
		dataset = None
	num_batches = 10000//64
	labels,latents = [],[]
	for step,data in enumerate(dataset.train(64)):
		img, lab = data
		lab = process_labels(lab)[0]
		labels.append(lab)
		latents.append(np.concatenate([l[1] for l in model.encoder(dataset.preprocess(img),use_mean=False)],axis=-1))
		if step>=num_batches:break
		print(f"{step}/{num_batches}\r", end="")
	latents = np.concatenate(latents,0)
	labels = np.concatenate(labels,0)
	np.savez_compressed(path, latents=latents, labels=labels, path=path)

def mutual_information(latents, labels=None, conditional=None, normalization_type="labels"):
	if labels is None: labels = latents
	assert len(latents.shape) == 2
	mi_map = np.zeros((latents.shape[1], labels.shape[1]))
	for i in range(mi_map.shape[0]):
		for j in range(mi_map.shape[1]):
			if conditional is None:
				if normalization_type=="all": 
					mi_map[i,j] = sklearn.metrics.normalized_mutual_info_score(latents[:,i], labels[:,j])
				else:
					mi_map[i,j] = sklearn.metrics.mutual_info_score(latents[:,i], labels[:,j])
				# mi_map[i,j] = sklearn.metrics.adjusted_mutual_info_score(latents[:,i], labels[:,j])
			else:
				mi_map[i,j] = drv.information_mutual_conditional(latents[:,i], labels[:,j], conditional)
	# normalize wrt labels:
	if not conditional is None or normalization_type=="labels": mi_map=mi_map/discrete_entropy(labels).reshape(1,-1)
	if not conditional is None or normalization_type=="latents": mi_map=mi_map/discrete_entropy(labels).reshape(1,-1)
	return mi_map

def overall_disentanglement(mi_score, independent_factors):
	# we pick out the representative latent by finding the latent that contains the max mutual information with the factor
	# we then find MIG with this to find how much the factor is represented by this code
	# we also need to find if this latent is the sole item that is contained in that latent
	# originally MIG didn't need to do this, but we do as we ignore the MIG for hierarchical structure groups
	# we remove the dependents since we want to have the "child exists" as seperate metric
	nondepmi = np.concatenate((mi_score[:,0:2], np.max(mi_score[:,2:7],-1).reshape(-1,1), mi_score[:,7:]),-1)
	sorted_latent = np.sort(nondepmi,0)
	mig = np.mean((sorted_latent[-1,:]-sorted_latent[-2,:])[independent_factors]) # in MIG we subtract to account for multiple latents that contain the same information about a factor

	representative_latents = nondepmi[(np.argsort(nondepmi,0)[-1], np.arange(nondepmi.shape[-1]))]
	max_information = nondepmi[(np.argsort(nondepmi,0)[-1],)]
	max_information = np.sort(max_information*np.logical_not(np.eye(*max_information.shape)))[:,-1]
	latent_information = np.maximum(representative_latents-max_information,0)
	latent_information = np.mean(latent_information)

	return np.sqrt(np.sum(np.square([mig,latent_information])))/np.sqrt(2)

def parent_disentanglement(mi_score, parent_factor, child_factors):
	parent_exists = mi_score[:,parent_factor]
	other_factors = np.max(mi_score[:,[i for i in range(mi_score.shape[-1]) if not i == parent_factor]],-1)
	parent_exists_idx = parent_exists>other_factors
	if not np.sum(parent_exists_idx): return 0
	mi_score = mi_score[parent_exists_idx]
	parent_score = sorted(mi_score[:,parent_factor])
	if len(parent_score)==1: return parent_score[-1]
	parent_score = parent_score[-1]-parent_score[-2]
	return parent_score

def child_existence(mi_score, parent_factor, child_factors):
	# Gets child existence by keeping the ones that have greater information about the child than the parent
	# Will return the max information that exists about a child in these latents.
	child_exists = np.max(mi_score[:,child_factors],-1)
	other_factors = np.max(mi_score[:,[i for i in range(mi_score.shape[-1]) if not i in child_factors]],-1)
	child_exists_idx = child_exists>other_factors
	if not np.sum(child_exists_idx): return 0
	child_exists = child_exists[child_exists_idx]
	child_exists_score = np.maximum(np.max(child_exists),0)
	return child_exists_score

def child_disentanglement(mi_score, parent_factor, child_factors):
	# disentanglement/prominence of child factors
	# MIG doesn't account for if many factors are in one latent, 
	# This usually isn't a problem in independent datasets
	# but for us, where partial information could happen due to details not being measured,
	# it may cause a problem 
	#	- as entangled partial are represented one latent and the rest of the information is not represented
	# so, we must measure the max element of columns and then rows
	# also though the entropy of a child factor is an upper bound on the information, 
	# the entropy also includes parent information which the model may or may not require in it's latent representation
	child_exists = np.max(mi_score[:,child_factors],-1)
	other_factors = np.max(mi_score[:,[i for i in range(mi_score.shape[-1]) if not i in child_factors]],-1)
	child_exists_idx = child_exists>other_factors
	if not np.sum(child_exists_idx): return 0
	child_dis_score_1 = np.sort(mi_score[child_exists_idx][:,child_factors],0)
	if len(child_dis_score_1)>2:
		child_dis_score_1 = np.mean(child_dis_score_1[-1]-child_dis_score_1[-2])
	else:
		child_dis_score_1 = np.mean(child_dis_score_1[-1])

	# latents that don't contain any information might bias this measure
	child_mi = mi_score[child_exists_idx][:,child_factors]
	representative_latents = child_mi[(np.argsort(child_mi,0)[-1], np.arange(child_mi.shape[-1]))]

	max_information = child_mi[(np.argsort(child_mi,0)[-1],)]
	max_information = np.sort(max_information*np.logical_not(np.eye(*max_information.shape)))[:,-1]
	child_dis_score_2 = np.maximum(representative_latents-max_information,0)
	child_dis_score_2 = np.mean(child_dis_score_2)

	child_dis_score = np.sqrt(np.sum(np.square([child_dis_score_1,child_dis_score_2])))/np.sqrt(2)
	return child_dis_score

def run_mi_metric(model_data, results_storage=None): # put in local scope for jupyter
	stored_metrics = {
			"path":[],
			"mi_score":[],
			"disentangled_non_hierarchy":[],
			"parent_score":[],
			"child_exists_score":[],
			"child_dis_score":[]}
	for modelnum, data in model_data.items():
		latents, labels, path = data
		print(f"Running Model {modelnum}")
		stored_metrics["path"].append(path)
		latents,labels = histogram_discretize(latents), histogram_discretize(labels),
		
		mi_score = np.transpose(mutual_information(latents, labels))
		stored_metrics["mi_score"].append(mi_score)

		mi_score = mutual_information(latents, labels)
		disentangled_non_hierarchy = overall_disentanglement(mi_score, independent_factors=[0,1,3,4,5])
		stored_metrics["disentangled_non_hierarchy"].append(disentangled_non_hierarchy)

		parent_score = parent_disentanglement(mi_score, 2, child_factors=[3,4,5,6])
		stored_metrics["parent_score"].append(parent_score)

		child_exists_score = child_existence(mi_score, parent_factor=2, child_factors=[3,4,5,6])
		stored_metrics["child_exists_score"].append(child_exists_score)

		child_dis_score=child_disentanglement(mi_score,parent_factor=2, child_factors=[3,4,5,6])
		stored_metrics["child_dis_score"].append(child_dis_score)
		

		if results_storage is None:
			cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
			print(mi_score)
			print(disentangled_non_hierarchy)
			print(parent_score)
			print(child_exists_score)
			print(child_dis_score)
			ax = sns.heatmap(mi_score, cmap=cmap, yticklabels=[
				"cube color", "scale", "overall inscribed color", 'inscribed color 1', 'inscribed color 2', 'inscribed color 3', 'inscribed color 4', 
				'floor color', 'wall color', 'azimuth'])
			plt.title("MI Heatmap")
			plt.xlabel("latents")
			plt.ylabel("labels")
			plt.show()
	if not results_storage is None: np.savez_compressed(results_storage, **stored_metrics)

##############################################################
# all dci code is from, or modified from disentanglement lib #
##############################################################
from sklearn import ensemble
import scipy
from multiprocessing import Pool

class DCI:
	def get_factor_gbt(self, factor):
		model = ensemble.GradientBoostingRegressor(verbose=0)
		model.fit(self.latents, self.labels[:,factor])
		importance_matrix_factor = np.abs(model.feature_importances_)
		return model, importance_matrix_factor
	
	def get_trained_gbts(self, latents, labels):
		self.latents, self.labels = latents, labels
		num_codes, num_factors = latents.shape[-1], labels.shape[-1]
		importance_matrix = np.zeros(shape=[num_codes, num_factors])
		models = []

		with Pool(10) as p: out = p.map(self.get_factor_gbt, list(range(num_factors)))

		for i,gbt_ret in enumerate(out):
			model, importance_matrix_factor = gbt_ret
			models.append(model)
			importance_matrix[:,i] = importance_matrix_factor
		return importance_matrix, models
	# def __call__(self, gbts, latents, labels):
	# 	test_loss = []
	# 	for i in range(labels.shape[-1]):
	# 		test_loss.append(np.mean(model.predict(labels) == latents[:, i]))
	# 	return np.mean(test_loss)

def disentanglement_per_code(importance_matrix):
	"""Compute disentanglement score of each code."""
	# importance_matrix is of shape [num_codes, num_factors].
	return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
								  base=importance_matrix.shape[1])

def disentanglement(importance_matrix):
	"""Compute the disentanglement score of the representation."""
	per_code = disentanglement_per_code(importance_matrix)
	if np.abs(importance_matrix.sum())<1e-5:
		importance_matrix = np.ones_like(importance_matrix)
	code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
	return np.sum(per_code*code_importance)

def completeness_per_factor(importance_matrix):
	"""Compute completeness of each factor."""
	# importance_matrix is of shape [num_codes, num_factors].
	return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
								  base=importance_matrix.shape[0])

def completeness(importance_matrix):
	""""Compute completeness of the representation."""
	per_factor = completeness_per_factor(importance_matrix)
	if np.abs(importance_matrix.sum())<1e-5:
		importance_matrix = np.ones_like(importance_matrix)
	factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
	return np.sum(per_factor*factor_importance)
import time
def run_dci_metric(model_data, results_storage=None): # put in local scope for jupyter
	stored_metrics = {"path":[], "importance_matrix":[],
			"overall_disentanglement":[],"overall_completeness":[],"child_disentanglement":[],
			"child_completeness":[], "elbo_loss":[]}
	start_time = time.time()
	for modelnum, data in model_data.items():
		latents, labels, path, elbo_loss = data
		start_time = time.time()
		stored_metrics["path"].append(path)
		dci = DCI()
		importance_matrix, models = dci.get_trained_gbts(latents, labels)
		stored_metrics["importance_matrix"].append(np.transpose(importance_matrix))

		child_factors = [3,4,5,6]
		parent_factor = [2]
		overall_factors = [i for i in range(importance_matrix.shape[-1]) if not i in child_factors]
		child_latents = np.max(importance_matrix[:,overall_factors],axis=-1)<=np.max(importance_matrix[:,child_factors],axis=-1)
		overall_latents = np.logical_not(child_latents)


		if sum(overall_factors):
			overall_disentanglement = disentanglement(importance_matrix[overall_latents][:, overall_factors])
			overall_completeness = completeness(importance_matrix[overall_latents][:, overall_factors])
		else:
			overall_disentanglement = 0
			overall_completeness = 0

		if sum(child_latents):
			child_disentanglement = disentanglement(importance_matrix[child_latents][:, child_factors])
			child_completeness = completeness(importance_matrix[child_latents][:, child_factors])
		else:
			child_disentanglement = 0
			child_completeness = 0

		stored_metrics["overall_disentanglement"].append(overall_disentanglement)
		stored_metrics["overall_completeness"].append(overall_completeness)
		stored_metrics["child_disentanglement"].append(child_disentanglement)
		stored_metrics["child_completeness"].append(child_completeness)
		stored_metrics["elbo_loss"].append(elbo_loss)

		print(f"Ran Model {modelnum}/{len(model_data)}, {np.round(time.time()-start_time,5)} seconds",
			"overall disentanglement: ",np.round(overall_disentanglement,4),
			"overall completeness: ",np.round(overall_completeness,4),
			"child factor disentanglement: ",np.round(child_disentanglement,4),
			"child factor completeness: ",np.round(child_completeness,4),
			"elbo_loss: ", np.round(elbo_loss,4),
			)

		if results_storage is None:
			cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=False, as_cmap=True)
			print("overall disentanglement: ",overall_disentanglement)
			print("overall completeness: ",overall_completeness)
			print("child factor disentanglement: ",child_disentanglement)
			print("child factor completeness: ",child_completeness)
			print("elbo_loss: ",elbo_loss)
			ax = sns.heatmap(mi_score, cmap=cmap, yticklabels=[
				"cube color", "scale", "overall inscribed color", 'inscribed color 1', 'inscribed color 2', 'inscribed color 3', 'inscribed color 4', 
				'floor color', 'wall color', 'azimuth'])
			plt.title("Importance Matrix")
			plt.xlabel("latents")
			plt.ylabel("labels")
			plt.show()
	if not results_storage is None: np.savez_compressed(results_storage, **stored_metrics)

# metric data from pregenerated files
import os

def process_latents(latents, batch_size, get_mean=True):
	assert latents.shape[0] == batch_size
	assert latents.shape[1] == 3
	latents = latents[:,0+int(get_mean)].reshape(latents.shape[0], -1)
	return latents

def get_metric_data(experiments="experiments/"):
	final_files = []
	datafolder, datafile = 'run_set', 'sample.npz'
	for path,folders,files in list(os.walk(experiments)):
		if path.endswith(datafolder) and datafile in files:
			final_files.append(os.path.join(path,datafile))
	model_data = {}
	print("Gathering data...")
	for i,file in enumerate(final_files):
		data = np.load(file)
		model_data[i] = process_latents(data["mean_latents"], len(data["labels"])), data["labels"], data["path"], get_elbo_loss(data["true_images"], data["reconstructed_sample"], data["kld"])
		print(f"Progress: {i+1}/{len(final_files)} \r", end="")
	return model_data

def get_elbo_loss(true, reconstruction, kld):
	lossfunc = ImageBCE()
	elbo_loss = lossfunc.numpy(true, reconstruction)+kld_loss_reduction_numpy(np.sum(kld,1))
	return elbo_loss

def get_model_dataset(model_num):
	model,path = get_model(model_num=model_num, return_path=True, basepath=os.environ["HSR_MODEL_DIR"])
	sample_path = os.path.dirname(os.path.relpath(path, os.environ["HSR_MODEL_DIR"]))+"/"
	dataset = get_dataset(path)
	return model, dataset, path

def run_model(num, modeldir=None, storagedir=None, finished_step=200000, samples = 10000):
	if modeldir is None: modeldir = os.environ["HSR_MODEL_DIR"]
	if storagedir is None: storagedir = os.environ["HSR_RESULTS_DIR"]
	modelpaths = get_model_paths(modeldir)
	path = modelpaths[num]

	savepathdir = os.path.join(storagedir,os.path.relpath(path, os.environ["HSR_MODEL_DIR"]))
	savepathdir = os.path.join(os.path.dirname(savepathdir), "run_set")
	savepath = os.path.join(savepathdir, "sample.npz")
	if os.path.exists(savepath) or np.load(os.path.join(os.path.dirname(path), "train", "train_progress.npz"), allow_pickle=True)["step"]<finished_step: return
	if not os.path.exists(savepathdir): os.makedirs(savepathdir)
	
	model, dataset, pathcheck = get_model_dataset(num)
	assert path==pathcheck, "path mismatch"


	# get latents, labels, origninal image, reconstructed image, path
	num_batches = samples//64
	labels,latents,mean_latents,reconstructed_sample,reconstructed_mean,true_images,kld,kld_mean = [],[],[],[],[],[],[],[]
	for step,(img, lab) in enumerate(dataset.train(64)):
		img = dataset.preprocess(img)
		lab = process_labels(lab)[0]

		# provide the same data across models
		# save the "test" set into (images, latents) into hdf5 file
		# get mean reco+latents+kl, sampled reco+latents+kl, save into hdf5 file



		# collect information
		# latent shapes are changed 
		# from: [batch, layer num, (s,m,std), batchsize, num latents] -> 
		# to: [batch x batchsize, (s,m,std), layer nums, num latents]
		true_images.append(img)
		labels.append(lab)
		
		reconstructed_sample.append(model(img, use_mean=False))
		latents.append(np.transpose(model.latent_space, (2,1,0,3)))
		kld.append(np.transpose(model.past_kld, (1,0,2)))
		
		reconstructed_mean.append(model(img, use_mean=True))
		mean_latents.append(np.transpose(model.latent_space, (2,1,0,3)))
		kld_mean.append(np.transpose(model.past_kld, (1,0,2)))

		if step>=num_batches:break
		print(f"modelnum {num}/{len(modelpaths)}: {step+1}/{num_batches}\r", end="")
	
	true_images = np.concatenate(true_images,0)
	labels = np.concatenate(labels,0)
	
	reconstructed_sample = np.concatenate(reconstructed_sample,0)
	latents = np.concatenate(latents,0)
	kld = np.concatenate(kld,0)
	
	reconstructed_mean = np.concatenate(reconstructed_mean,0)
	mean_latents = np.concatenate(mean_latents,0)
	kld_mean = np.concatenate(kld_mean,0)
	
	np.savez_compressed(savepath, path=savepath, 
		true_images=true_images, 
		labels=labels, 

		reconstructed_sample=reconstructed_sample, 
		latents=latents, 
		kld=kld,
		
		reconstructed_mean=reconstructed_mean, 
		mean_latents=mean_latents, 
		kld_mean=kld_mean,
		)

def run_models(modeldir, storagedir, finished_step=200000, samples = 10000):
	print("Running...")
	modelpaths = get_model_paths(modeldir)
	for num, path in enumerate(modelpaths):
		base = os.path.dirname(path)
		# savepath
		savepath = os.path.join(storagedir,os.path.relpath(path, os.environ["HSR_MODEL_DIR"]))
		savepath = os.path.join(os.path.dirname(savepath), "run_set")
		savepath = os.path.join(savepath, "sample.npz")
		#if not os.path.exists(savepath) and np.load(os.path.join(base, "train", "train_progress.npz"), allow_pickle=True)["step"]>=finished_step:
		run_model(num=num, modeldir=modeldir, storagedir=storagedir, finished_step=finished_step, samples=samples)

import subprocess
def run_metrics():
	process = ["python3.7", "metrics.py", "0"]
	#process = ["ls", "-l"]
	run = subprocess.run(process)
	while run.returncode:
		time.sleep(0.5)
		run = subprocess.run(process)
	print("Finished")

import sys
if __name__ == '__main__':
	# run_mi_metric(get_metric_data(), "results.npz") # mi metric
	run_dci_metric(get_metric_data(os.environ["HSR_RESULTS_DIR"]), "results_dci_with_elbo.npz") # Dci metic
	# if len(sys.argv)>1:
		# run_models(os.environ["HSR_MODEL_DIR"], os.environ["HSR_RESULTS_DIR"])
	# else:
	# 	run_metrics()