"""Prebuilt scenes
"""
import numpy as np
import os
import pyvista as pv
from hiershapes.base import Shape3D, Node
from hiershapes.view import Background, Camera, Lighting
import hiershapes.utils as ut
import copy

class Scene:
	# base class
	def check_inbetween(self, item, item_range, item_shape):
		if item_shape: assert np.asarray(item).shape == item_shape
		assert np.all(np.logical_and(item_range[0] <= np.asarray(item), np.asarray(item) <= item_range[1])), f"{item} not in range {item_range} and not of shape {item_shape}"

	def randomize_parameters(self,**kw):
		assert not self.parameters is None
		np.random.seed()
		parameters = {}
		for k,v in self.parameters.items():
			if k in kw:
				parameters[k] = kw[k]
			else:
				parameters[k] = np.random.uniform(*v[0],size=v[1])
		return parameters

class Pyramid(Scene):
	def __init__(self, num_blocks=3, shape_cache_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"premade_shapes/pyramid"), image_size = [64,64]):
		if not os.path.exists(shape_cache_path): os.makedirs(shape_cache_path)
		self.shape_cache_path = shape_cache_path
		self.color = ut.HLSToRGB()
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

	def create_block(self, length, color, height, shape, resolution=20, child_block=None):
		# make block
		def scale(cloud):
			cloud = cloud*np.asarray([[length,length,height]])
			return cloud
		block = Shape3D(shape_val=shape, path=self.shape_cache_path, color=self.color(color), resolution=resolution)
		block.add_postprocess(scale, "vertices")
		block.add_postprocess(lambda cloud: cloud+np.asarray([[0,0,height]]), "vertices") # shift to 0

		# make hierarchy
		block = Node(block)
		if not child_block is None:
			child_block.add_postprocess(lambda cloud: cloud+np.asarray([[0,0,height*2-0.01*height]]), "vertices")
			block.add_child(child_block)
		child_block = block
		return block

	def __call__(self,plotter=None,**parameters):
		pyramid, object_transformations, camera_kw, bg_kw = self.set_parameters(**parameters)
		##########
		# Shapes #
		##########
		blocks = []
		child_block = None
		for kw in pyramid:
			child_block = self.create_block(**{**self.base_block_kw,**kw},child_block=child_block)
			blocks.append(child_block)

		# scaling/transformations
		def scale(vertices):
			scale_val = object_transformations["scale"]
			return vertices*np.asarray([[scale_val,scale_val,scale_val]])
		blocks[-1].add_postprocess(scale,"vertices")

		if plotter is None: plotter = pv.Plotter(off_screen=True)
		for mesh in blocks[-1](): plotter.add_mesh(**mesh)

		###########
		# Plotter #
		###########
		camera = Camera(12, 0, np.pi/2-1/180*np.pi)
		lightingobj = Lighting()
		background = Background(**bg_kw)
		plotter.camera_position = camera(**camera_kw)
		wall, floor, ceil = background()
		plotter.add_mesh(**wall)
		plotter.add_mesh(**floor)
		plotter.set_background("royalblue")
		#plotter = lightingobj(plotter) # lighting
		#timerobj("Plotted View")
		image = plotter.screenshot(None,window_size=self.image_size,return_img=True)
		#timerobj("Screenshot")
		#del plotter
		return image

class BoxHead(Scene):
	def __init__(self, eyes=[0,1], # 0 is left eye, 1 is right eye
		shape_cache_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"premade_shapes/box_head"), image_size = [64,64]):
		if not os.path.exists(shape_cache_path): os.makedirs(shape_cache_path)
		self.shape_cache_path = shape_cache_path
		self.color = ut.HLSToRGB()
		self.image_size = image_size
		self.eyes = eyes
		self.parameters = dict( # set_parameters depends on this
			# object
			color=([0,1], ()),
			eye_color=([0,1], (4,)),
			scale=([0.5,2], (3,)), 
			
			# view
			bg_color=([0,1], (2,)),
			azimuth=([-np.pi/6,np.pi/6], ()),
			)

		# constants
		self.shape = 1
		self.length = 1	
		self.mid_length = 2/7*self.length
		self.top_length = 1/2*self.mid_length

	def create_block(self, length, depth, height, color, shape, translate=[0,0,0], scale=[1,1,1], resolution=20, is_show=True):
		# make block
		def scale_vert(cloud):
			cloud = cloud*np.asarray([[depth,length,height]])
			return cloud
		block = Shape3D(shape_val=shape, path=self.shape_cache_path, color=color, resolution=resolution)
		block.add_postprocess(scale_vert, "vertices")

		# make hierarchy
		block = Node(block, is_show=is_show)
		block.add_postprocess(lambda cloud: cloud*np.asarray([scale])+np.asarray([translate]), "vertices")
		return block

	def create_eyes(self, length, mid_length, shape):
		eyes = []
		for i in self.eyes:
			assert 0<=i<2
			i = i*2-1
			eye = self.create_block(
				length=mid_length, depth=length, height=mid_length, 
				color=self.color(0.1), # color doesn't matter as it won't be shown
				shape=shape, 
				translate=[length/1.9,i*3/7*length,3/7*length], is_show=False)
			eyes.append(eye)
		return eyes

	def create_iris(self, color, eye_color, length, mid_length, top_length, shape):
		iris = []
		for i,ec in enumerate(eye_color):
			i,j = np.floor(i/2)*2-1, i%2*2-1
			iris.append(self.create_block(
				length=top_length, depth=length/2, height=top_length, 
				color=self.color(ec), shape=shape, 
				translate=[0,i*1/2*mid_length,j*1/2*mid_length]))
		return iris

	def create_box_head(self, color, eye_color):


		# head
		base = self.create_block(
			length=self.length, depth=self.length, height=1, 
			color=self.color(color), shape=self.shape, 
			translate=[0,0,self.length])
		
		# eyes
		eyes = self.create_eyes(self.length, self.mid_length, self.shape)
		for eye in eyes: base.add_child(eye)
		#base = ut.subtract_mesh(base, [i.shape3d for i in eyes])

		# iris
		for eye in eyes:
			iris = self.create_iris(color=color, eye_color=eye_color, length=self.length, 
				mid_length=self.mid_length, top_length=self.top_length, shape=self.shape)
			for i in iris: eye.add_child(i)
		return base

	def __call__(self,plotter=None,**parameters):
		##########
		# Shapes #
		##########
		for k,v in parameters.items():
			assert k in self.parameters, f"unavailable key: '{k}', available factors: {list(self.parameters.keys())}"
			self.check_inbetween(v,*self.parameters[k])
		# parameters
		color = parameters["color"]
		eye_color = parameters["eye_color"]
		scale = parameters["scale"]
		floor_color = parameters["bg_color"][0]
		wall_color = parameters["bg_color"][1]
		azimuth = parameters["azimuth"]

		boxhead = self.create_box_head(color=color, eye_color=eye_color)

		# other transformations
		boxhead.add_postprocess(lambda vertices: vertices*scale, "vertices")

		if plotter is None: 
			plotter = pv.Plotter(off_screen=True)
			#plotter = pv.Plotter()
		boxhead_meshes = boxhead()
		for mesh in boxhead_meshes: plotter.add_mesh(**mesh)
		
		##############
		# Background #
		##############
		camera = Camera(12, 0, np.pi/2-1/180*np.pi)
		plotter.camera_position = camera(theta=azimuth,translate=[0,0,2])
		background = Background(wall_color=self.color(wall_color), floor_color=self.color(floor_color), wall_faces=[0,2,3])
		wall, floor, ceil = background()
		plotter.add_mesh(**wall)
		plotter.add_mesh(**floor)
		plotter.set_background("royalblue")
		#plotter.show()
		#exit()
		image = plotter.screenshot(None,window_size=self.image_size,return_img=True)
		return image

class BoxHeadCentralEye(BoxHead):
	def __init__(self, shape_cache_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),"premade_shapes/box_head"), image_size = [64,64]):
		super().__init__(eyes=[0], shape_cache_path=shape_cache_path, image_size=image_size)
		self.length = 1	
		self.mid_length = 3/7*self.length
		self.top_length = 1/2*self.mid_length

	def create_eyes(self, length, mid_length, shape):
		eyes = []
		for i in self.eyes:
			assert 0<=i<2
			i = i*2-1
			eye = self.create_block(
				length=mid_length, depth=length, height=mid_length, 
				color=self.color(0.1), # color doesn't matter as it won't be shown
				shape=shape, 
				translate=[length/1.9,0,0], is_show=False)
			eyes.append(eye)
		return eyes
