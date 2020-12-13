"""Prebuilt scenes
"""
import numpy as np
import os
import pyvista as pv
from hiershapes.base import HueToColor, Shape3D, Node
from hiershapes.view import Background, Camera, Lighting
import copy

# scene
class Pyramid:
	def __init__(self, num_blocks=3, shape_cache_path="dev_tmp/", image_size = [64,64]):
		if not os.path.exists(shape_cache_path): os.makedirs(shape_cache_path)
		self.shape_cache_path = shape_cache_path
		self.color = HueToColor()
		self.base_block_kw = dict(
			length=1, color=0.5, height=0.3, shape=0.5, resolution=20, #object parameters
			) #default parameters
		
		###################
		# Parameter Specs #
		###################
		self.parameters = dict( # set_parameters depends on this
			# object
			color=([0,1], (num_blocks,)),
			length=([0.5,1], (num_blocks,)), 
			scale=([1,2], ()), 
			shape=([0,2], ()),
			
			# view
			azimuth=([-np.pi/6,np.pi/6], ()),
			bg_color=([0,1], (2,))
			) # parameters and ranges
		self.num_blocks = num_blocks
		self.image_size = image_size
		
		# default parameters
		self.pyramid = [dict() for _ in range(num_blocks)]
		self.object_transformations = dict()
		self.bg_kw = dict(wall_faces=[0,2,3])
		self.camera_kw = dict(theta=0,translate=[0,0,2])

	def check_inbetween(self, item, item_range, item_shape):
		if item_shape: assert np.asarray(item).shape == item_shape
		assert np.all(np.logical_and(item_range[0] <= np.asarray(item), np.asarray(item) <= item_range[1])), f"{item} not in range {item_range} and length {item_length}"

	def randomize_parameters(self,**kw):
		parameters = {}
		for k,v in self.parameters.items():
			if k in kw:
				parameters[k] = kw[k]
			else:
				parameters[k] = np.random.uniform(*v[0],size=v[1])
		return parameters

	def set_parameters(self, **kw):
		for k in kw.keys(): assert k in self.parameters.keys()
		for k,v in kw.items(): self.check_inbetween(v,*self.parameters[k])
		pyramid = copy.deepcopy(self.pyramid)
		object_transformations = copy.deepcopy(self.object_transformations)
		bg_kw = copy.deepcopy(self.bg_kw)
		camera_kw = copy.deepcopy(self.camera_kw)
		
		# pyramid
		if "length" in kw: kw["length"] = np.cumprod(kw["length"][::-1])[::-1]
		for k in ["color", "length", "shape"]:
			if not k in kw: continue
			for i in range(len(pyramid)):
				if len(np.asarray(kw[k]).shape):
					pyramid[i][k] = kw[k][i]
				else:
					pyramid[i][k] = kw[k]

		# pyramid transformations
		object_transformations["scale"] = kw["scale"]

		# view
		camera_kw["theta"]=kw["azimuth"]

		for k,i in zip(["floor_color","wall_color"],kw["bg_color"]): bg_kw[k] = self.color(i)
		return pyramid, object_transformations, camera_kw, bg_kw

	def create_block(self, length, color, height, shape, resolution=20, base_block=None):
		# make block
		def make_rectangle(cloud):
			cloud = cloud*np.asarray([[length,length,height]])
			return cloud
		block = Shape3D(shape_val=shape, path=self.shape_cache_path, color=self.color(color), resolution=resolution)
		block.add_postprocess(make_rectangle, "vertices")
		block.add_postprocess(lambda cloud: cloud+np.asarray([[0,0,height]]), "vertices") # shift to 0

		# make hierarchy
		block = Node(block)
		if not base_block is None:
			base_block.add_postprocess(lambda cloud: cloud+np.asarray([[0,0,height*2-0.01*height]]), "vertices")
			block.add_child(base_block)
		base_block = block
		return block

	def __call__(self, **parameters):
		#from disentangle.general.tools import Timer
		#timerobj = Timer()
		#timerobj("Start")
		pyramid, object_transformations, camera_kw, bg_kw = self.set_parameters(**parameters)
		#timerobj("Set Parameters")
		##########
		# Shapes #
		##########
		blocks = []
		base_block = None
		for kw in pyramid:
			base_block = self.create_block(**{**self.base_block_kw,**kw},base_block=base_block)
			blocks.append(base_block)
		#timerobj("Got Object")

		# scaling/transformations
		def scale(vertices):
			scale_val = object_transformations["scale"]
			return vertices*np.asarray([[scale_val,scale_val,scale_val]])
		blocks[-1].add_postprocess(scale,"vertices")
		#timerobj("Transformed Object")
		###########
		# Plotter #
		###########
		camera = Camera(12, 0, np.pi/2-1/180*np.pi)
		lightingobj = Lighting()
		background = Background(**bg_kw)

		#timerobj("Got View")
		###########
		# Plotter #
		###########
		plotter = pv.Plotter(off_screen=True)
		for mesh in blocks[-1](): plotter.add_mesh(**mesh)
		plotter.camera_position = camera(**camera_kw)
		wall, floor, ceil = background()
		plotter.add_mesh(**wall)
		plotter.add_mesh(**floor)
		plotter.set_background("royalblue")
		plotter = lightingobj(plotter) # lighting
		#timerobj("Plotted View")
		image = plotter.screenshot(None,window_size=self.image_size,return_img=True)
		#timerobj("Screenshot")
		return image
