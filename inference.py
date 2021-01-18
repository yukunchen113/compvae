import os
from hsr.save import ModelSaver
import hsr.dataset as ds
from hsr.visualize import lvae_traversal, LVAETraversal
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from hsr.utils.regular import cprint
import shutil
from disentangle import visualize as vs
def inference_lvae(model_num=None):
	# Select Model Number #
	path = "experiments/"
	paths = []
	for base,folders,files in os.walk(path):
		if "model" in folders:
			paths.append(os.path.join(base,"model"))
	paths.sort()
	if model_num is None:
		for i,p in enumerate(paths): print(i,":",p)
		exit()
	path = paths[model_num]
	cprint.blue("selected:", path)

	# dataset params #
	tf.random.set_seed(1)
	#dataset = ds.Shapes3D()
	dataset = ds.HierShapesBoxhead(use_server=False)
	#dataset = ds.CelebA()
	test_data = dataset.test()

	# create model #
	modelsaver = ModelSaver(path)
	model = modelsaver.load()
	assert not model is None, f"No model found in {path}"
	
	# Run Inference #
	#out = model(dataset.preprocess(test_data[:32]))

	###################################
	# See Effects of Top Layer Latent #
	###################################
	'''
	inputs = dataset.preprocess(test_data[:32])
	stack_vertically = lambda x: np.concatenate(x,axis=0)
	stack_horizontally = lambda x: np.concatenate(x,axis=1)
	# latent space samples
	latent_space = model.encoder(inputs)

	# paramaters
	reconstruct_top_n = 1 # clear lower layers
	latent_nums = 4

	# setup space to only keep last layer
	latent_space[:-reconstruct_top_n]=[None for _ in latent_space[:-reconstruct_top_n]]
	
	for latent_num in range(latent_nums):
		latent_means = [None if i is None else i[1].numpy().copy() for i in latent_space]
		space = []

		top_latent_sample = np.linspace(-3, 3, 30)

		for i in top_latent_sample:
			# change a specified latent
			latent_means[-reconstruct_top_n][:,latent_num] = i # changing the mean as we are using that

			# run decoder prior (just make all except last latents space samples None)
			rec,lspace= model.decoder(latent_means, return_ls=True, use_mean=True) #fill the other layers
			
			# see effects of layer directly below the selected one
			space.append(lspace[-(reconstruct_top_n+1)][1].numpy()) # get the mean

		space = np.transpose(space, (1,0,2))
		nlatent = np.cumsum(np.ones_like(space),-1)-1

		# plotting
		space = space - np.mean(space,axis=1,keepdims=True) # center at 0
		#space = space.reshape(space.shape[0],-1)
		#nlatent = nlatent.reshape(nlatent.shape[0],-1)

		#plt.ylim(-0.5,0.5)
		for lnum in range(space.shape[-1]):
			plt.subplot(1,space.shape[-1],lnum+1)
			plt.title(f"top latent:{latent_num},lower latent:{lnum}")
			for batchnum in range(len(space)):
				plt.scatter(top_latent_sample,space[batchnum,:,lnum])
		plt.show()
	'''
	########################################################################
	# See Reconstruction Effects of Varying Uncorrelated Top Layer Latents # 
	########################################################################
	inputs = dataset.preprocess(test_data[:32])
	
	analysis = LVAELayerDependenceAnalysis("dev_analysis/test", True)
	analysis.make_dependence_traversal(model, inputs[:2])
	analysis.get_traversal_ranges(model,inputs)
	exit()
	stack_vertically = lambda x: np.concatenate(x,axis=0)
	stack_horizontally = lambda x: np.concatenate(x,axis=1)
	# latent space samples
	latent_space, encoder_outputs = model.encoder(inputs, is_return_encoder_outputs=True)

	# paramaters
	reconstruct_top_n = 1 # clear lower layers
	num_upper_layer_latents = latent_space[-1][0].shape[-1]

	# setup space to only keep last layer
	latent_space[:-reconstruct_top_n]=[None for _ in latent_space[:-reconstruct_top_n]]
	for latent_num in range(num_upper_layer_latents):
		latent_means = [None if i is None else i[1].numpy().copy() for i in latent_space]
		space = []
		recs = []
		top_latent_sample = np.linspace(-3, 3, 30)

		for i in top_latent_sample:
			# change a specified latent
			latent_means[-reconstruct_top_n][:,latent_num] = i # changing the mean as we are using that

			# run decoder prior (just make all except last latents space samples None)
			rec,lspace=model.decoder(latent_means, return_ls=True, use_mean=True, encoder_outputs=encoder_outputs) #fill the other layers
			
			# see effects of layer directly below the selected one
			space.append(lspace[-(reconstruct_top_n+1)][1].numpy()) # get the mean
			recs.append(rec.numpy())
		recs = np.transpose(recs, (1,0,2,3,4))
		space = np.transpose(space, (1,0,2))
		nlatent = np.cumsum(np.ones_like(space),-1)-1

		# plot reconstructions
		#image = []
		#for batchnum in range(len(inputs)):
		#	image.append(stack_horizontally([inputs[batchnum]]+list(recs[batchnum])))
		#plt.imshow(stack_vertically(image[:10]))
		#plt.show()

		'''
		# plot latents spaces
		#space = space - np.mean(space,axis=1,keepdims=True) # center at 0
		#space = space.reshape(space.shape[0],-1)
		#nlatent = nlatent.reshape(nlatent.shape[0],-1)

		for lnum in range(space.shape[-1]):
			plt.subplot(1,space.shape[-1],lnum+1)
			#plt.ylim(-0.5,0.5)
			#plt.ylim(-20,20)
			plt.title(f"top latent:{latent_num},lower latent:{lnum}")
			for batchnum in list(range(len(space)))[:5]:
				plt.scatter(top_latent_sample,space[batchnum,:,lnum])
		plt.show()
		#'''

		# plot latents space ranges
		#space = space.reshape(space.shape[0],-1)
		#nlatent = nlatent.reshape(nlatent.shape[0],-1)

		for lnum in range(space.shape[-1]):
			plt.subplot(1,space.shape[-1],lnum+1)
			#plt.ylim(-0.5,0.5)
			#plt.ylim(-20,20)
			plt.xlim(0,60)
			plt.title(f"top latent:{latent_num},lower latent:{lnum}")
			plt.hist(np.amax(space,axis=1)[...,lnum]-np.amin(space,axis=1)[...,lnum])
		plt.show()		

class LVAELayerDependenceAnalysis:

	def __init__(self, path, overwrite=False):
		if not overwrite: assert not os.path.exists(path)
		if os.path.exists(path): shutil.rmtree(path)
		os.makedirs(path)
		self.path = path
		self.Traversal = LVAETraversal

	def traverse_with_encoder_information(self,model,images,traversal_layer,trav_range):
		latent_space, encoder_outputs = model.encoder(images, is_return_encoder_outputs=True)
		latent_space[:traversal_layer]=[None for _ in range(len(latent_space[:traversal_layer]))]
		recs = []
		for latent_num in range(latent_space[traversal_layer][0].shape[-1]):
			latent_means = [None if i is None else i[1].numpy().copy() for i in latent_space]
			top_latent_sample = np.linspace(*trav_range)
			recs.append([])
			for i in top_latent_sample:
				latent_means[traversal_layer][:,latent_num] = i # changing the mean as we are using that
				rec,lspace=model.decoder(latent_means, return_ls=True, use_mean=True, encoder_outputs=encoder_outputs) #fill the other layers
				recs[-1].append(rec.numpy())
		recs = np.asarray(recs)# shape of (num top layer latents, num steps trav, N,W,H,C)
		recs = np.transpose(recs, (1,2,0,3,4,5))
		recs = recs.reshape([recs.shape[0],recs.shape[1]*recs.shape[2]]+list(recs.shape)[-3:])
		return recs

	def make_dependence_traversal(self,model,images,layer_nums=[-1,-2], trav_range=[-3,3,30]):
		assert len(images.shape) == 4, "must be batch of images"
		assert len(layer_nums)==2, "must specify layer nums where the lower layer will be evaluated for changes in top layer"
		assert layer_nums[0]>layer_nums[1], "must specify layer nums where the lower layer will be evaluated for changes in top layer"
		stack_vertically = lambda x: np.concatenate(x,axis=-3)
		stack_horizontally = lambda x: np.concatenate(x,axis=-2)
		# traverse lower layer regularly and traverse top layer no encoder info
		traverse = self.Traversal(model, inputs=images)
		traverse.traverse_complete_latent_space(*trav_range)
		traverse.create_samples()
		inputs,*layer_trav = traverse.samples_list # samples in the shape of (num trav steps, num latents, )

		# traverse top layer encoder info (for lower layer only)
		layer_trav = [layer_trav[i] for i in layer_nums]
		layer_trav.append(self.traverse_with_encoder_information(model,images=images,traversal_layer=layer_nums[0],trav_range=trav_range))
		
		# stack layers
		layer_trav = stack_vertically([inputs]+layer_trav) # stack the layers
		layer_trav = np.transpose(layer_trav,(1,0,2,3,4))
		layer_trav = stack_horizontally(layer_trav)
		vs.create_gif(layer_trav, os.path.join(self.path,"dependence_traversal.gif"))
		return layer_trav

	def get_traversal_ranges(self,model,images):
		assert len(images.shape) == 4, "must be batch of images"

'''
def inference_lvae(model_num=None):
	# Select Model Number #
	path = "experiments/"
	paths = []
	for base,folders,files in os.walk(path):
		if "model" in folders:
			paths.append(os.path.join(base,"model"))
	paths.sort()
	if model_num is None:
		for i,p in enumerate(paths): print(i,":",p)
		exit()
	path = paths[model_num]
	cprint.blue("selected:", path)

	# dataset params #
	tf.random.set_seed(1)
	dataset = ds.HierShapesBoxhead(use_server=False)
	#dataset = ds.CelebA()
	test_data = dataset.test()
	test_data = dataset.preprocess(test_data)
	# create model #
	modelsaver = ModelSaver(path)
	model = modelsaver.load()
	images = model(test_data)
	for i in zip(test_data, images):
		plt.imshow(np.concatenate(i, axis=-2))
		plt.show()
'''
import sys
if __name__ == '__main__':
	args=sys.argv
	if len(args)>1:
		inference_lvae(int(args[1]))
	else:
		inference_lvae(None)
