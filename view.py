import tensorflow as tf
import time
from core.model.handler import ModelHandler, DualModelHandler
from utilities.mask import Mask, mask_traversal
import core.config.config as cfg 
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
import imageio
import shutil
import execute


def show_model(modelhandler, gif_path, lof=None, num_images=4):
	inputs_test = modelhandler.config.inputs_test[1:1+num_images]
	inputs_test = modelhandler.config.preprocessing(inputs_test)

	traversal = mask_traversal(
		modelhandler.model,
		inputs_test,
		return_traversal_object=True, 
		is_visualizable=False,
		num_steps=30,
		min_value=-3,
		max_value=3
		)
	traversal.save_gif(gif_path,lof)

def show_model_given_args():
	model_path = sys.argv[1]
	lof = None if len(sys.argv) < 2 else int(sys.argv[2])
	gif_path = "test.gif"
	modelhandler = ModelHandler(base_path=model_path, train_new=False, load_model=True)
	show_model(modelhandler, gif_path, lof)


class GifCreator:
	def __init__(self, gif_folder="test/test1"):
		self.gif_folder = gif_folder
		self.desc_path = "%s/gif_descriptions.md"%gif_folder
		self.create_gif_paths()
		self.gif_num = 0

	def create_gif_paths(self):
		# will overwrite gifs paths
		if os.path.exists(self.gif_folder):
			a = input("Overwrite?")
			if not "y" in a:
				exit()
			shutil.rmtree(self.gif_folder)
		os.makedirs(self.gif_folder)

	def __call__(self, modelhandler, lof=None, num_images=2):
		gif_path = os.path.join(self.gif_folder, "model_%d.gif"%(self.gif_num))
		show_model(modelhandler, gif_path, lof, num_images)
		self.gif_num+=1

		with open(self.desc_path, "a") as f:
			f.write("%s: %s\n"%(gif_path, modelhandler.base_path))


def main():
	gcreate = GifCreator("test/test7")
	
	mbeta = [10, 25, 50]
	mrandom_seed = 1
	mis_train = [True, False]
	cbeta = 600
	clof = 4
	crandom_seed = 1
	randmask = [True, False]
	rootpath = "exp3_test"
	is_tcvae = [True,False]
	is_pretrain = True
	args = [[b, mrandom_seed, it, cbeta, clof, crandom_seed, tc, rm, rootpath, is_pretrain, False] for tc in is_tcvae for it in mis_train for rm in randmask for b in mbeta]

	for a in args:
		modelhandler = execute.pretrain_experiment(*a)
		for i,mhand in enumerate([modelhandler.mask_mh, modelhandler.comp_mh]):
			if os.path.exists(mhand.model_save_file):
				if not i:
					gcreate(mhand, lof=modelhandler.comp_config.mask_latent_of_focus, num_images=4)
				else:
					gcreate(mhand)
if __name__ == '__main__':
	main()