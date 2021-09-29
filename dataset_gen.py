import hsr.dataset as ds
import numpy as np
import os
import time
import h5py
import sys
import pickle
def gen_datasets(folder, dataset, num_steps=None, batch_size=128, capacity=5000):
	st = time.time()
	storage = ds.HDF5Dataset(folder, capacity=capacity)
	dataset = dataset.train(batch_size)
	for step, data in enumerate(dataset):
		data, labels = data
		if not num_steps is None and step>num_steps: break
		storage.save(data=data, labels=labels)
		if not step%200: print(step, data.shape, time.time()-st)
	storage.close()
		
def main():
	num_proc = 24
	num_batches = 40
	capacity = 20
	datasets = [
		("dataset/boxheadsimple2", ds.HierShapesBoxheadSimple2),
		("dataset/boxheadsimple", ds.HierShapesBoxheadSimple),
		("dataset/boxhead_07", ds.HierShapesBoxhead)]
	if len(sys.argv)>=2: datasets = [datasets[int(sys.argv[1])]]
	print("running: ", datasets)
	for folder, dataset in datasets:
		dataset = dataset(use_server=False, use_pool=False, run_once=False, num_proc=num_proc, 
			prefetch=num_proc, pool_size=None, port=None)
		gen_datasets(folder, dataset, num_batches, capacity=capacity)
		dataset.dataset_manager.terminate()
		print(f"Done {folder}")

def load_data():
	folder = "dataset/boxheadsimple2"
	storage = ds.HDF5Dataset(folder)
	storage.batch(32)
	for i,data in enumerate(storage):
		images, labels = data
		print(i, images.shape, len(labels[0]))
import matplotlib.pyplot as plt
def load_hsr_dataset():
	dataset = ds.HierShapesBoxheadSimple2(use_preloaded=True)
	trainset = dataset.train(32)
	for images, labels in trainset:
		print(images.shape)

if __name__ == '__main__':
	load_data()

# - 12: 36.37/24
# - 24: better.