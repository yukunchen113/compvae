import tensorflow as tf
import time
from model import ModelHandler
from utilities import Mask
import config as cfg 
import sys
import os
import matplotlib.pyplot as plt
def train_models(path):
	modelhandler = ModelHandler(base_path=path, train_new=False, load_model=True)
	inputs_test = modelhandler.config.inputs_test[:2]
	
	inputs_test = modelhandler.config.preprocessing(inputs_test)
	
	#"""
	mask_obj = Mask(modelhandler.model, 1)
	mask_obj(inputs_test)
	generated = mask_obj.view_mask_traversals(inputs_test)

	plt.imshow(generated)
	plt.show()
	#"""
if __name__ == '__main__':
	path = sys.argv[1]
	train_models(path)