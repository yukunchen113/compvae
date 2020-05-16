import tensorflow as tf
import time
from model import ModelHandler, DualModelHandler
from utilities import Mask
import config as cfg 
import sys
import os
import matplotlib.pyplot as plt
import utilities
import numpy as np
import utils as ut
import imageio
import shutil
def show_model(path):
	modelhandler = DualModelHandler(base_path=path, train_new=False, load_model=True)
	inputs_test = modelhandler.comp_mh.config.inputs_test[1:3]
	model = modelhandler.mask_mh
	inputs_test = model.config.preprocessing(inputs_test)
	"""
	model.model(inputs_test)[:,:,:,:3]

	# model kl evaluation
	_, mu, sig = model.model.get_latent_space()
	print(np.average(mu,0))
	print(np.average(sig,0))
	kldiv = ut.tf_custom.loss.kl_divergence_with_normal(mu, sig)
	avg_kl = np.average(kldiv, 0)
	print(avg_kl>1/3*np.amax(avg_kl))
	"""
	traversal = utilities.mask_traversal(
		model.model,
		inputs_test,
		return_traversal_object=True, 
		is_visualizable=False,
		num_steps=30,
		min_value=-3,
		max_value=3
		)
	gif_path = "test.gif"
	traversal.save_gif(gif_path, 3)


	#plt.imshow(generated)
	#plt.show()
	"""
	import matplotlib.animation as animation

	fig = plt.figure()

	ims = []
	for i in generated[:-1,3]:
		im = 
		ims.append([im])
	ims=ims+ims[::-1]
	ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
									repeat_delay=0)

	# ani.save('dynamic_images.mp4')

	plt.show()
	#"""

	"""
	mask_obj = Mask(model.model, 1)
	mask_obj(inputs_test)
	generated = mask_obj.view_mask_traversals(inputs_test)

	plt.imshow(generated)
	plt.show()
	#"""
if __name__ == '__main__':
	path = sys.argv[1]
	show_model(path)