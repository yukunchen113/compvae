import hiershapes as hs
import numpy as np
import os
import shutil
import pyvista as pv
import open3d as o3d
import vtk
import colorsys
import copy
import inspect
import dill
import socket
import sys
import time

from hiershapes.base import BaseProj, CubeProj, SphereProj, CylinderProj, ShapeProj, Base3D, Shape3D, Node
from hiershapes.view import Background, Camera, Lighting
from hiershapes.scene import Pyramid, BoxHead, BoxHeadCentralEye
import hiershapes.scene as sc
from hiershapes.dataset import Batch
from hiershapes.utils import Parameters

import hiershapes.utils as ut 
import hiershapes.dataset as ds
from multiprocessing import Process, Queue
import queue

# test basic
def cubic_grid(resolution=2):
	x = np.linspace(-1,1,resolution)
	y = np.linspace(-1,1,resolution)
	x,y = np.meshgrid(x,y)
	z = np.ones_like(x)
	face = np.transpose(np.vstack((x.reshape(-1),y.reshape(-1),z.reshape(-1))))
	faces = []
	for i in range(3): # 3 dimensions
		for sign in [-1,1]:
			f = face.copy()
			f[:,-1] = f[:,-1]*sign

			switch_rows = [0,1]
			switch_rows.insert(i,2)
			f = f[:,switch_rows]

			faces.append(f)
	return np.vstack(faces)

def plot_basic_shapes():
	cube = CubeProj()
	sphere = SphereProj()
	cylinder = CylinderProj()

	x = np.linspace(-3,3,20)
	y = np.linspace(-3,3,20)
	z = np.linspace(-3,3,20)
	x,y,z = np.meshgrid(x,y,z)
	data = np.transpose(np.vstack((x.reshape(-1),y.reshape(-1),z.reshape(-1))))

	cube_y = cube(data)
	sphere_y = sphere(cube_y)
	cylinder_y = cylinder(data)

	p = pv.Plotter(shape=(1,3))
	p.subplot(0,0)
	p.add_mesh(pv.PolyData(cube_y).delaunay_3d(), color="g", 
		#show_edges=True
		)
	p.subplot(0,1)
	p.add_mesh(pv.PolyData(sphere_y).delaunay_3d(), color="y", 
		#show_edges=True
		)
	p.subplot(0,2)
	p.add_mesh(pv.PolyData(cylinder_y).delaunay_3d(), color="b", 
		#show_edges=True
		)
	p.show()

def plot_combo_shapes():
	# color
	color = ut.HSVToRGB()

	##########
	# Shapes #
	##########
	base_length = 1
	mid_length = 0.75
	top_length = 0.5

	# top
	def make_rectangle(cloud):
		cloud = cloud*np.asarray([[top_length,top_length,0.3]])
		return cloud
	top = Shape3D(1, color=color(0.5), resolution=20)
	top.add_postprocess(make_rectangle, "cloud")
	top.add_postprocess(lambda cloud: cloud+np.asarray([[0,0,0.3]]), "cloud") # shift to 0
	
	# mid
	def make_rectangle(cloud):
		cloud = cloud*np.asarray([[mid_length,mid_length,0.3]])
		return cloud
	mid = Shape3D(1, color=color(0.1), resolution=20)
	mid.add_postprocess(make_rectangle, "cloud")
	mid.add_postprocess(lambda cloud: cloud+np.asarray([[0,0,0.3]]), "cloud") # shift to 0
	
	# base
	def make_rectangle(cloud):
		cloud = cloud*np.asarray([[base_length,base_length,0.3]])
		return cloud
	base = Shape3D(1, color=color(0.2), resolution=20)
	base.add_postprocess(make_rectangle, "cloud")
	base.add_postprocess(lambda cloud: cloud+np.asarray([[0,0,0.3]]), "cloud") # shift to 0

	# make tree
	top = Node(top)
	mid = Node(mid)
	base = Node(base)
	top.add_postprocess(lambda cloud: cloud+np.asarray([[0,0,0.6]]), "cloud") # shift child above
	mid.add_child(top)
	mid.add_postprocess(lambda cloud: cloud+np.asarray([[0,0,0.6]]), "cloud") # shift child above
	base.add_child(mid)

	# can apply processing that affects all descendents
	def apply_scaling(cloud):
		return cloud*np.asarray([[1,1,1]])
	base.add_postprocess(apply_scaling, "cloud")

	###########
	# Plotter #
	###########
	camera = Camera(15, 0, np.pi/2-1/180*np.pi)

	p = pv.Plotter()
	for mesh in base(): p.add_mesh(**mesh)
	p.camera_position = camera(theta=0,translate=[0,0,2])

	##############
	# Background #
	##############
	background = Background(
		wall_color=color(0.2), 
		floor_color=color(0.7),
		wall_faces=[0,2,3])
	wall, floor, ceil = background()
	p.add_mesh(**wall)
	p.add_mesh(**floor)
	p.set_background("royalblue")

	############
	# Lighting #
	############
	lightingobj = Lighting()
	p = lightingobj(p) # lighting

	p.show()

def plot_joint_shape():
	shapeobj = Shape3D(0, resolution=10)
	mesh = shapeobj()
	p = pv.Plotter()
	p.add_mesh(mesh, color="fff000")
	p.show()

# test scene
import matplotlib.pyplot as plt
def plot_pyramid_scene():
	pyramid = Pyramid()
	
	# label
	# parameters = dict( # set_parameters depends on this
	# 	# object
	# 	color=[1,1,1],
	# 	length=[1,1,1],
	# 	scale=2,
	# 	shape=1,
		
	# 	# view
	# 	azimuth=np.pi/6,
	# 	bg_color=[1,1],
	# 	)
	# plt.imshow(pyramid(**parameters))
	# plt.show()

	# randomized batch
	batch = Batch(pyramid, set_kw=dict(shape=1))
	batch.batch(32)
	for images,labels in batch:
		print(images.shape), len(labels)
	# image
	plt.imshow(images[10])
	plt.show()

def plot_boxhead_raw():
	# color
	color = ut.HSVToRGB()

	##########
	# Shapes #
	##########
	base_length = 1
	mid_length = 2/7*base_length
	top_length = 1/2*mid_length


	# mids
	def make_rectangle(cloud):
		cloud = cloud*np.asarray([[base_length,mid_length,mid_length]])
		return cloud
	right = Shape3D(1, color=color(0.1), resolution=20)
	right.add_postprocess(make_rectangle, "vertices")
	right = Node(right, is_show=False)
	right.add_postprocess(lambda cloud: cloud+np.asarray([[base_length/2,3/7*base_length,3/7*base_length]]), "vertices") # shift to 0
	
	def make_rectangle(cloud):
		cloud = cloud*np.asarray([[base_length,mid_length,mid_length]])
		return cloud
	left = Shape3D(1, color=color(0.1), resolution=20)
	left.add_postprocess(make_rectangle, "vertices")
	left = Node(left, is_show=False)
	left.add_postprocess(lambda cloud: cloud+np.asarray([[base_length/2,-3/7*base_length,3/7*base_length]]), "vertices") # shift to 0

	# base
	def make_rectangle(cloud):
		cloud = cloud*np.asarray([[base_length,base_length,base_length]])
		return cloud
	base = Shape3D(1, color=color(0.2), resolution=20)
	base.add_postprocess(make_rectangle, "vertices")

	# make tree
	base = Node(base)
	base.add_child(right)
	base.add_child(left)
	base.add_postprocess(lambda cloud: cloud+np.asarray([[0,0,base_length]]), "vertices") # shift to 0
	def subtract(mesh):
		return mesh.boolean_difference(left.shape3d.mesh)
	base.shape3d.add_postprocess(subtract, "mesh") # shift to 0

	# can apply processing that affects all descendents
	def apply_scaling(cloud):
		return cloud*np.asarray([[1,1,1]])
	base.add_postprocess(apply_scaling, "vertices")

	###########
	# Plotter #
	###########
	p = pv.Plotter()
	for mesh in base(): 
		p.add_mesh(**mesh)

	##############
	# Background #
	##############
	camera = Camera(12, 0, np.pi/2-1/180*np.pi)
	background = Background(
		wall_color=color(0.2), 
		floor_color=color(0.7),
		wall_faces=[0,2,3])
	#lightingobj = Lighting()
	wall, floor, ceil = background()
	p.add_mesh(**wall)
	p.add_mesh(**floor)
	p.set_background("royalblue")
	p.camera_position = camera(theta=0,translate=[0,0,2])
	#p = lightingobj(p) # lighting

	p.show()

def plot_boxhead_scene():
	#boxhead = sc.BoxHead(eyes=[0])
	boxhead = sc.BoxHeadCentralEye()
	# parameters
	parameters = Parameters()
	def head():
		np.random.seed()
		out = {}
		out["color"] = hs.utils.quantized_uniform(*boxhead.parameters["color"][0])
		scale = hs.utils.quantized_uniform(0.75,1.25)
		out["scale"] = np.asarray([scale, scale, scale])
		return out
	parameters.add_parameters("head", head)

	def eyes(color, **kw):
		np.random.seed()
		out = {}
		eye_color = hs.utils.quantized_normal(0,0.1,n_quantized=5)+color
		out["eye_color"] = np.mod(hs.utils.quantized_uniform(-0.1, 0.1, size=4,n_quantized=5)+eye_color, 1)	
		return out
	parameters.add_parameters("eyes", eyes, ["head"])

	def view():
		np.random.seed()
		out = {}
		out["bg_color"] = hs.utils.quantized_uniform(*boxhead.parameters["bg_color"][0],size=boxhead.parameters["bg_color"][1])
		out["azimuth"] = hs.utils.quantized_uniform(*boxhead.parameters["azimuth"][0],size=boxhead.parameters["azimuth"][1])
		return out
	parameters.add_parameters("view", view)

	while True:
		image = boxhead(**parameters())
		plt.imshow(image)
		plt.show()

def get_boxhead_dataset():
	boxhead = BoxHeadCentralEye()
	# parameters
	parameters = Parameters()
	def head():
		np.random.seed()
		out = {}
		out["color"] = np.random.uniform(*boxhead.parameters["color"][0])
		length = np.random.uniform(0.75,1)
		scale = np.random.uniform(1,2)
		out["scale"] = np.asarray([length, length, 1])*np.asarray([scale, scale, scale])
		return out
	parameters.add_parameters("head", head)

	def eyes(color, **kw):
		np.random.seed()
		out = {}
		general_color = np.random.normal(0, 0.025)+color
		out["eye_color"] = np.clip(general_color+np.random.normal(0,0.025,size=4), *boxhead.parameters["eye_color"][0])
		return out
	parameters.add_parameters("eyes", eyes, ["head"])

	def view():
		np.random.seed()
		out = {}
		out["bg_color"] = np.random.uniform(*boxhead.parameters["bg_color"][0],size=boxhead.parameters["bg_color"][1])
		out["azimuth"] = np.random.uniform(*boxhead.parameters["azimuth"][0],size=boxhead.parameters["azimuth"][1])
		return out
	parameters.add_parameters("view", view)
	return boxhead, parameters

def plot_boxhead_dataset():
	boxhead, parameters = get_boxhead_dataset()
	
	dataset = ds.Batch(scene=boxhead,randomize_parameters_func=parameters)
	batch = dataset.batch(32)
	timerobj = ut.Timer()
	for images, labels in batch:
		timerobj("Got Batch...")
		print(images.shape, len(labels))
		#plt.imshow(images[0])
		#plt.show()

def run_ServerBatch_boxhead():
	boxhead = BoxHead(eyes=[0])
	# parameters
	parameters = Parameters()
	def head():
		np.random.seed()
		out = {}
		out["color"] = np.random.uniform(*boxhead.parameters["color"][0])
		scale = np.random.uniform(1,2)
		out["scale"] = np.asarray([scale, scale, scale])
		return out
	parameters.add_parameters("head", head)

	def eyes(color, **kw):
		np.random.seed()
		out = {}
		general_color = np.random.normal(0, 0.025)+color
		out["eye_color"] = np.clip(general_color+np.random.normal(0,0.025,size=4), *boxhead.parameters["eye_color"][0])
		return out
	parameters.add_parameters("eyes", eyes, ["head"])

	def view():
		np.random.seed()
		out = {}
		out["bg_color"] = np.random.uniform(*boxhead.parameters["bg_color"][0],size=boxhead.parameters["bg_color"][1])
		out["azimuth"] = np.random.uniform(*boxhead.parameters["azimuth"][0],size=boxhead.parameters["azimuth"][1])
		return out
	parameters.add_parameters("view", view)
	server = ds.Server(scene=boxhead, randomize_parameters_func=parameters, num_proc=5, prefetch=5, 
		retrieval_batch_size=32)
	server()

def run_ServerClientBatch_boxhead():
	boxhead = BoxHead(eyes=[0])
	# parameters
	parameters = Parameters()
	def head():
		np.random.seed()
		out = {}
		out["color"] = np.random.uniform(*boxhead.parameters["color"][0])
		scale = np.random.uniform(1,2)
		out["scale"] = np.asarray([scale, scale, scale])
		return out
	parameters.add_parameters("head", head)

	def eyes(color, **kw):
		np.random.seed()
		out = {}
		general_color = np.random.normal(0, 0.025)+color
		out["eye_color"] = np.clip(general_color+np.random.normal(0,0.025,size=4), *boxhead.parameters["eye_color"][0])
		return out
	parameters.add_parameters("eyes", eyes, ["head"])

	def view():
		np.random.seed()
		out = {}
		out["bg_color"] = np.random.uniform(*boxhead.parameters["bg_color"][0],size=boxhead.parameters["bg_color"][1])
		out["azimuth"] = np.random.uniform(*boxhead.parameters["azimuth"][0],size=boxhead.parameters["azimuth"][1])
		return out
	parameters.add_parameters("view", view)
	server = ds.ServerClient(scene=boxhead, randomize_parameters_func=parameters, num_proc=10, prefetch=10,
		pool_size=5, 
		retrieval_batch_size=100)
	client = server().batch(32)

	used_labels = []
	used_images = None
	for images, labels in client: # for basic checking, not optimized at all.
		print("STARTED CHECK")
		# check label pool
		label_pool = []
		for i in client.pool.label_pool:
			label_pool+=i
		new_labels = []
		for i in label_pool:
			is_same = False
			for j in used_labels:
				if ut.Compare()(i,j):
					is_same = True
					continue
			if not is_same:
				new_labels.append(i)
		used_labels+=new_labels
		
		# check image pool:
		num_new_images = 0
		new_images = np.concatenate(client.pool.image_pool, axis=0)
		if used_images is None:
			used_images = new_images
			num_new_images += len(new_images)
		else:
			test = None
			for i in used_images:
				a = np.all(np.abs(np.expand_dims(i,0)-new_images)<1e-4,(1,2,3))
				if test is None: 
					test = a
				else:
					test = np.logical_or(test, a)
			used_images = np.concatenate((used_images, new_images),0)
			num_new_images+=np.sum(np.logical_not(test).astype(int))



		print(len(new_labels), num_new_images)

# broadcasting server for multimodel training using generator.
def run_server():
	server = ut.Server()
	while True:
		requests = server.get_requests()
		mapping = {"random":np.random.uniform(0,1), "one":1}
		requests = {k:mapping[v] for k,v in requests.items()}
		if requests: print(requests)
		time.sleep(5)
		server.send_data(requests)
		time.sleep(0.2)

def run_client():
	client = ut.Client()
	print(client("random"))
	print(client("one"))

def run_server_client(client, server):
	if int(sys.argv[1]):
		print("Running Client")
		client()
	else:
		print("Running Server")
		server()

def run_batch():
	pyramid = Pyramid()
	dataset = ds.Batch(scene=pyramid,set_kw=dict(shape=1))
	dataset.batch(32)
	for i, data in enumerate(dataset):
		images, labels = data
		print(images.shape, len(labels))
		if i >= 5:
			break
	with open("dev_analysis/save.pickle", "wb") as f: 
		dill.dump(dataset, f)
	with open("dev_analysis/save.pickle", "rb") as f: 
		dataset = dill.load(f)

	for i, data in enumerate(dataset):
		images, labels = data
		print(f"Got Data {i}, {images.shape}, {len(labels)}")
		if i >= 5:
			break

def run_MultiProcessBatch():
	pyramid = Pyramid()
	dataset = ds.MultiProcessBatch(scene=pyramid, num_proc=5, prefetch=5, set_kw=dict(shape=1))
	test = dict( # set_parameters depends on this
		# object
		color=[1,1,1],
		length=[1,1,1],
		scale=2,
		shape=1,
		
		# view
		azimuth=np.pi/6,
		bg_color=[1,1],
		)
	images, parameters = dataset.get_images([test], return_labels=True)
	dataset.batch(32)
	for i, data in enumerate(dataset):
		images, labels = data
		print(f"Got Data {i}, {images.shape}, {len(labels)}")

def run_parallel_batch():
	pyramid = Pyramid()
	dataset = ds.ParallelBatch(scene=pyramid, num_proc=5, prefetch=5, retrieval_batch_size=32,set_kw=dict(shape=1))
	test = dict( # set_parameters depends on this
		# object
		color=[1,1,1],
		length=[1,1,1],
		scale=2,
		shape=1,
		
		# view
		azimuth=np.pi/6,
		bg_color=[1,1],
		)
	images, parameters = dataset.get_images([test], return_labels=True)
	print(images.shape)
	dataset.batch(32)
	for i, data in enumerate(dataset):
		images, labels = data
		print(f"Got Data {i}, {images.shape}, {len(labels)}")
		if i >= 5:
			break
	with open("dev_analysis/save.pickle", "wb") as f: 
		dill.dump(dataset, f)
	with open("dev_analysis/save.pickle", "rb") as f: 
		dataset = dill.load(f)

	for i, data in enumerate(dataset):
		images, labels = data
		print(f"Got Data {i}, {images.shape}, {len(labels)}")
		if i >= 5:
			break

def run_ServerBatch():
	pyramid = Pyramid()
	server = ds.Server(scene=pyramid, num_proc=5, prefetch=5, 
		retrieval_batch_size=32,set_kw=dict(shape=1))
	server()

def run_ClientBatch():
	dataset = ds.Client(prefetch=5)
	# test = dict( # set_parameters depends on this
	# 	# object
	# 	color=[1,1,1],
	# 	length=[1,1,1],
	# 	scale=2,
	# 	shape=1,
		
	# 	# view
	# 	azimuth=np.pi/6,
	# 	bg_color=[1,1],
	# 	)
	#images, parameters = dataset.get_images([test], return_labels=True)
	#print(images.shape)
	dataset.batch(32)
	for i, data in enumerate(dataset):
		images, labels = data
		print(images.shape)

def run_multiclient_server():
	# this is to mimic the effect of multiple clients running for a server
	# sees if client allows for termination and for second client
	scene, parameters = get_boxhead_dataset()
	
	servers, clients = [],[]
	port = 65349
	# run server client 1
	server = hs.dataset.ServerClient(scene=scene, randomize_parameters_func=parameters, 
		num_proc=10, prefetch=2, pool_size=2, port=port, retrieval_batch_size=100)
	client = server()
	servers.append(server)
	clients.append(client)
	assert not server.server is None

	# run server client 2
	server = hs.dataset.ServerClient(scene=scene, randomize_parameters_func=parameters, 
		num_proc=10, prefetch=2, pool_size=2, port=port, retrieval_batch_size=100)
	client = server()
	servers.append(server)
	clients.append(client)
	assert server.server is None

	# run server client 3
	server = hs.dataset.ServerClient(scene=scene, randomize_parameters_func=parameters, 
		num_proc=10, prefetch=2, pool_size=2, port=port, retrieval_batch_size=100)
	client = server()
	servers.append(server)
	clients.append(client)
	assert server.server is None


	for client in clients:
		client.batch(10)
		images, labels = client()
		print(images.shape)
	for i, client in enumerate(clients): 
		client.close()
		print("terminated client",i)
	clients = []

	# run server client 4
	server = hs.dataset.ServerClient(scene=scene, randomize_parameters_func=parameters, 
		num_proc=10, prefetch=2, pool_size=2, port=port, retrieval_batch_size=100)
	client = server()
	servers.append(server)
	clients.append(client)
	assert server.server is None

	for client in clients:
		client.batch(10)
		images, labels = client()
		print(images.shape)
	for i, client in enumerate(clients): 
		client.close()
		print("terminated client",i)

	for i,server in enumerate(servers): 
		server.close()
		print("terminated server",i)



if __name__ == '__main__':
	#plot_combo_shapes()
	#plot_joint_shape()
	#plot_pyramid_scene()
	#plot_boxhead_scene()
	run_multiclient_server()
	#plot_boxhead_dataset()
	#run_ServerClientBatch_boxhead()
	#run_server_client(run_ClientBatch, run_ServerBatch_boxhead)
	#plot_test()
	#run_server_client(run_client, run_server)
	#run_batch()
	#run_MultiProcessBatch()
	#run_parallel_batch()

	#run_server_client(run_ClientBatch, run_ServerBatch)