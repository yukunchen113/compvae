"""This file holds experiments that are performed, to achieve an understanding outlined in the readme file.
"""
import tensorflow as tf
import numpy as np
import commonly_used_objects as cuo
import matplotlib.pyplot as plt
import os

def main():
	experiment_base_path = "exp_1"
	for beta_value in [1,2,5,10,15,20,30,50,75,100]:
		experiment_path = os.path.join(experiment_base_path, "beta_%d"%beta_value)
		os.makedirs(experiment_path, exist_ok=True)

		# initialize model and dataset objects
		image_dir = os.path.join(experiment_path, cuo.image_dir)
		model_setup_dir = os.path.join(experiment_path, cuo.model_setup_dir)
		model_save_file = os.path.join(experiment_path, cuo.model_save_file)

		dataset_manager, dataset = cuo.dataset_manager, cuo.dataset
		model = cuo.get_model(beta_value) # beta shouldn't matter here, since no training
		preprocessing = cuo.preprocessing

		inputs_test = cuo.inputs_test
		inputs_test = preprocessing(inputs_test)
		print(inputs_test[:3].shape)
		model(inputs_test[:3]) # arbitrary 4D call for building model to load weights
		model.load_weights(model_save_file)
		test_generation(model, inputs_test)


def test_generation(model, inputs_test):
	generated = cuo.image_traversal(model, inputs_test[20:21])
	plt.imshow(generated)
	plt.show()

def test_reconstruction():
	"""Success
	"""
	a = model(inputs_test[:2])
	plt.imshow(a[0])
	plt.show()

if __name__ == '__main__':
	main()