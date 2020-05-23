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


def show_model(modelhandler, gif_path, lof=None):
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
	modelhandler = ModelHandler(base_path=model_path, train_new=False, load_model=True)
	show_model(modelhandler, gif_path, lof)

def main():
	# load models
	mbeta = [10, 25, 50]
	mrandom_seed = 1
	mis_train = [True, False]
	cbeta = 600
	clof = 4
	crandom_seed = 1
	randmask = [True, False]
	rootpath = "exp2"
	is_tcvae = [True,False]
	is_pretrain = True
	args = [[b, mrandom_seed, it, cbeta, clof, crandom_seed, tc, rm, rootpath, is_pretrain, False] for tc in is_tcvae for it in mis_train for rm in randmask for b in mbeta]
	



	base_dir = "test2"
	if os.path.exists(base_dir):
		a = input("Overwrite?")
		if not "y" in a:
			exit()
		shutil.rmtree(base_dir)
	os.makedirs(base_dir)

	# iterate through each model
	with open("%s/desc.md"%base_dir, "w") as f:
		for i,a in enumerate(args):
			gif_path = "%s/model_%d.gif"%(base_dir, i)
			model_handler = execute.pretrain_experiment(*a)
			show_model(model_handler, gif_path, latent_of_focus)

			be, rs, _, tc, _ = a
			tc ="Beta-TCVAE" if tc else "BetaVAE"
			f.write("%s: beta = %d, random_seed=%d, model_type=%s\n"%(
				gif_path, be, rs, tc))

if __name__ == '__main__':
	main()