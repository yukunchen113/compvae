"""This file holds experiments that are performed, to achieve an understanding outlined in the readme file.
"""
import tensorflow as tf
import numpy as np
import utils as ut
import commonly_used_objects as cuo
import matplotlib.pyplot as plt
import os

class MaskedTraversal(ut.visualize.Traversal):
	def create_samples(self, is_interweave=True):
		"""Will keep last the same
		"""
		super().create_samples()

		# get mask
		generated =self.samples
		im_n = 0
		batch_num = 0
		g0 = generated[:-1] # for 1.1.1
		g1 = generated[1:]
		g = np.abs(g0 - g1)
		g = g>0.005
		masked_g0 = np.where(g, g0, 0)
		if not is_interweave:
			self.samples[:-1] = masked_g0
			"""
			# this is to view the effect of the mask
			diff = np.concatenate((g, g0, masked_g0), -3)
			print(diff.shape)
			diff = np.concatenate(diff, -2)
			plt.imshow(diff)
			plt.show()
			"""
		else:
			# set mask to samples
			s_shape = self.samples.shape
			self.samples = np.expand_dims(self.samples,0)
			self.samples = np.concatenate((self.samples, self.samples),0)
			self.samples[0,:-1] = masked_g0
			self.samples = self.samples.transpose((1,2,0,3,4,5))
			self.samples = self.samples.reshape((s_shape[0], -1, *s_shape[2:]))

			# change inputs
			i_shape = self.inputs.shape
			self.inputs = np.expand_dims(self.inputs,0)
			self.inputs = np.concatenate((self.inputs, self.inputs),0)
			self.inputs = self.inputs.transpose((1,0,2,3,4))
			self.inputs = self.inputs.reshape((-1, *i_shape[-3:]))

def main():
	experiment_base_path = "exp_1"
	for beta_value in [1,2,5,10,15,20,30,50,75,100]:
		if not beta_value in [1, 30]: # for some reason, model 1 must be loaded first
			continue
		experiment_path = os.path.join(experiment_base_path, "beta_%d"%beta_value)

		# initialize model and dataset objects
		image_dir = os.path.join(experiment_path, cuo.image_dir)
		model_setup_dir = os.path.join(experiment_path, cuo.model_setup_dir)
		model_save_file = os.path.join(experiment_path, cuo.model_save_file)

		dataset_manager, dataset = cuo.dataset_manager, cuo.dataset
		model = cuo.get_model(beta_value) # beta shouldn't matter here, since no training
		preprocessing = cuo.preprocessing

		inputs_test = cuo.inputs_test
		inputs_test = preprocessing(inputs_test)
		model(inputs_test[:3]) # arbitrary 4D call for building model to load weights
		model.load_weights(model_save_file)
	
	generated = cuo.image_traversal(model,
		inputs_test[19:20], 
		min_value=0, max_value=0, 
		num_steps=15, is_visualizable=True, Traversal=MaskedTraversal)
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