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


def show_model(model_path, gif_path, lof=None):
	modelhandler = ModelHandler(base_path=model_path, train_new=False, load_model=True)
	inputs_test = modelhandler.config.inputs_test[1:5]
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
	show_model(model_path, gif_path, lof)

def main():
	# load models
	betas = [10,15,30,40,50]
	random_seed = [1,10,25,30]
	rootpath = "exp2"
	args = [(b, r, rootpath, False) for b in betas for r in random_seed]
	
	# iterate through each model
	for a in args:
		model_handler = execute.run_mask(*a)



if __name__ == '__main__':
	main()