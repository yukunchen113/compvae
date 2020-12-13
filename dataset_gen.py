import hiershapes as hs
import numpy as np
import os
import pyvista as pv
import open3d as o3d
import vtk
import colorsys
import copy
import inspect

from hiershapes.base import BaseProj, CubeProj, SphereProj, CylinderProj, HueToColor, ShapeProj, Base3D, Shape3D, Node
from hiershapes.view import Background, Camera, Lighting
from hiershapes.scene import Pyramid


import threading
import queue




class Batch():
	def __init__(self, scene, set_kw={}, prefetch=0, batch_size=None):
		self.scene = scene
		self.batch_size = batch_size
		self.set_kw = set_kw
		self.prefetch = prefetch
		self.threads = []
		self.queue = None

	def batch(self,batch_size):
		self.batch_size = batch_size

	def get_batch(self, savetoqueue=False):
		images, labels = [], []
		for i in range(self.batch_size):
			parameters = self.scene.randomize_parameters(**self.set_kw)
			labels.append(parameters)
			images.append(self.scene(**parameters))
		if savetoqueue:
			self.queue.put((np.asarray(images), labels))
		else:
			return np.asarray(images), labels
	
	def __call__(self):
		if self.prefetch:
			for i in self.threads:
				if not i.is_alive(): 
					i.join()
			self.threads = [i for i in self.threads if i.is_alive()]
			if self.queue is None:
				self.queue = queue.Queue(self.prefetch)
				for i in range(self.prefetch): 
					thread = threading.Thread(target=self.get_batch, kwargs={"savetoqueue":True})
					thread.start()
					self.threads.append(thread)
				images, labels = self.queue.get()
			else:
				images, labels = self.queue.get()
				thread = threading.Thread(target=self.get_batch, kwargs={"savetoqueue":True})
				thread.start()
				self.threads.append(thread)
			return images, labels
		else:
			return self.get_batch()

	def __iter__(self):
		return self
	def __next__(self):
		return self.__call__()

# test
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
	color = HueToColor()

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

import matplotlib.pyplot as plt
def plot_scene():
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
	batch = Batch(pyramid, set_kw=dict(shape=1), prefetch=0)
	batch.batch(100)
	images, labels = batch()
	
	# image
	plt.imshow(images[0])
	plt.show()

def plot_test():
	import vtk

	# Use a sphere
	import pyvista
	from pyvista import examples

	mesh = Shape3D(0, resolution=10)()

	colors = vtk.vtkNamedColors()
	colors.SetColor('HighNoonSun', [255, 255, 251, 255])  # Color temp. 5400°K
	colors.SetColor('100W Tungsten', [255, 214, 170, 255])  # Color temp. 2850°K

	# light1 = vtk.vtkLight()
	# light1.SetFocalPoint(0, 0, 0)
	# light1.SetPosition(0, 1, 0.2)
	# light1.SetColor(colors.GetColor3d('HighNoonSun'))
	# light1.SetIntensity(0.3)
	#renderer.AddLight(light1)

	light2 = vtk.vtkLight()
	light2.SetFocalPoint(0, 0, 0)
	light2.SetPosition(-0.5, 1, 0.5)
	light2.SetColor(colors.GetColor3d('100W Tungsten'))
	light2.SetIntensity(1.0)
	#renderer.AddLight(light2)

	# Add a box on the bottom
	bounds = mesh.GetBounds()
	rnge = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
	expand = 1.0
	THICKNESS = rnge[2] * 0.1
	center = ((bounds[1] + bounds[0]) / 2.0, bounds[2] + THICKNESS / 2.0, (bounds[5] + bounds[4]) / 2.0)
	xlen = bounds[1] - bounds[0] + (rnge[0] * expand)
	ylen = THICKNESS
	zlen = bounds[5] - bounds[4] + (rnge[2] * expand)
	plane_mesh = pyvista.Cube(center, xlen, ylen, zlen)

	# create the plotter
	pl = pyvista.Plotter()
	pl.renderer.RemoveAllLights()
	#pl.renderer.AddLight(light1)
	pl.renderer.AddLight(light2)
	pl.add_mesh(mesh, ambient=0.2, diffuse=0.5, specular=0.51, specular_power=30,
				smooth_shading=True, color='orange')
	pl.add_mesh(plane_mesh)

	shadows = vtk.vtkShadowMapPass()
	seq = vtk.vtkSequencePass()

	passes = vtk.vtkRenderPassCollection()
	passes.AddItem(shadows.GetShadowMapBakerPass())
	passes.AddItem(shadows)
	seq.SetPasses(passes)

	# Tell the renderer to use our render pass pipeline
	cameraP = vtk.vtkCameraPass()
	cameraP.SetDelegatePass(seq)
	pl.renderer.SetPass(cameraP)

	# nice camera position
	cpos = [(0.10533537264201895, 0.28584795035272126, 0.3472861003034224),
			(-0.028657675440026627, 0.060039973117803645, -0.094230396877531),
			(-0.07280203079138521, 0.8967386829419564, -0.4365313262850401)]

	pl.camera_position = cpos
	pl.show()
	print(pl.camera_position)


if __name__ == '__main__':
	#plot_combo_shapes()
	#plot_joint_shape()
	plot_scene()
	#plot_test()

