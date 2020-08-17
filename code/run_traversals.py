import tensorflow as tf
import time
from core.model.handler import ModelHandler, DualModelHandler, ProVLAEModelHandler 
from utilities.vlae_method import vlae_traversal
import core.config.config as cfg 
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
import shutil
import execute


def show_model(modelhandler, gif_path, images=[1,5]):
	inputs_test = modelhandler.config.inputs_test[images]
	inputs_test = modelhandler.config.preprocessing(inputs_test)

	traversal = vlae_traversal(
		modelhandler.model,
		inputs_test,
		return_traversal_object=True, 
		is_visualizable=False,
		num_steps=30,
		min_value=-3,
		max_value=3
		)
	traversal.save_gif(gif_path)

def show_model_given_args():
	model_path = sys.argv[1]
	lof = None if len(sys.argv) < 2 else int(sys.argv[2])
	gif_path = "test.gif"
	modelhandler = ModelHandler(base_path=model_path, train_new=False, load_model=True)
	show_model(modelhandler, gif_path, lof)


class GifCreator:
	def __init__(self, gif_folder="test/gif_test", is_overwrite=False):
		self.gif_folder = gif_folder
		self.desc_path = "%s/gif_descriptions.md"%gif_folder
		self.create_gif_paths(is_overwrite=is_overwrite)
		self.gif_num = 0

	def create_gif_paths(self, is_overwrite=False):
		# will overwrite gifs paths
		if os.path.exists(self.gif_folder):
			if not is_overwrite:
				a = input("Overwrite?")
				if not "y" in a:
					exit()
			shutil.rmtree(self.gif_folder)
		os.makedirs(self.gif_folder)

	def __call__(self, modelhandler, images=[1,5], desc=""):
		gifname = "model_%d.gif"%(self.gif_num)
		gif_path = os.path.join(self.gif_folder, gifname)
		show_model(modelhandler, gif_path, images)
		self.gif_num+=1

		with open(self.desc_path, "a") as f:
			f.write("%s: %s\n"%(gifname, desc))

def run_all_models():
	gcreate = GifCreator("test/gif_test_1", is_overwrite=True)
	
	base_path = "experiments"
	paths = [os.path.dirname(i[0]) for i in os.walk(base_path) if "model_setup" in i[0]]
	for path in paths:
		#try:
		modelhandler = ProVLAEModelHandler(path)
		
		# model description
		desc_list = ["beta", "random_seed", "gamma", "num_latents", "latent_connections"]
		desc_list = ["\n\t%s = %s"%(i, getattr(modelhandler.config, i)) for i in desc_list]
		desc_list.append("\n\t%s = %s"%("experiment_path", path))
		desc = "".join(desc_list)

		# create gif
		gcreate(modelhandler, desc=desc)
		#except:
		#	pass

if __name__ == '__main__':
	run_all_models()