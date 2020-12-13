import numpy as np
import os
import pyvista as pv
import open3d as o3d
import colorsys
import copy
import inspect

class BaseProj:
	def remove_invalid_points(self, points):
		return points
	def call(self, points):
		return points
	def __call__(self, points, return_original=False):
		assert points.shape[-1]==3
		orig_points = self.remove_invalid_points(points)
		points = self.call(orig_points)
		if return_original: return points, orig_points
		return points	

class CubeProj(BaseProj):
	def __init__(self):
		self.ignore = np.asarray([[0,0,0]])
	def remove_invalid_points(self, points):
		for i in self.ignore:
			points = points[np.all(points!=i,axis=-1)]
		return points
	def call(self, points):
		points = points/np.amax(np.abs(points),axis=-1,keepdims=True)
		return points

class SphereProj(BaseProj):
	def __init__(self):
		self.ignore = np.asarray([[0,0,0]])
	def remove_invalid_points(self, points):
		for i in self.ignore:
			points = points[np.all(points!=i,axis=-1)]
		return points
	def call(self, points):
		points = points/np.sqrt(np.sum(np.square(points),axis=-1,keepdims=True))
		return points

class CylinderProj(BaseProj):
	def __init__(self):
		self.ignore = np.asarray([[0,0,0]])
		self.cube = CubeProj()
	def remove_invalid_points(self, points):
		for i in self.ignore:
			points = points[np.all(points!=i,axis=-1)]
		return points
	def call(self, points):
		points = self.cube(points)
		face = np.argmax(np.abs(points),-1)==points.shape[-1]-1
		points[face,:-1] = points[face,:-1]*np.sqrt(1-0.5*np.square(np.flip(points[face,:-1],-1)))
		points[~face,:-1] = points[~face,:-1]/np.sqrt(np.sum(np.square(points[~face,:-1]),axis=-1,keepdims=True))
		return points

class HueToColor:
	def __init__(self, percent_saturation=100, percent_value=100):
		self.saturation = percent_saturation/100
		self.value = percent_value/100
	def __call__(self, hue):
		assert hue <= 1 and hue >= 0, "hue must be between 0 and 1"
		return colorsys.hsv_to_rgb(hue,self.saturation,self.value)

class ShapeProj(BaseProj):
	def __init__(self, shape_val):
		self.shape_val = shape_val
		self.shapes = [SphereProj(), CubeProj(), CylinderProj()]
		assert int(self.shape_val) <= len(self.shapes)-1

	def call(self, points):
		points = self.shapes[(np.floor(self.shape_val)).astype(int)](points)
		if np.floor(self.shape_val)+1 != len(self.shapes): 
			p1, p0 = self.shapes[(np.floor(self.shape_val)+1).astype(int)](points,return_original=True)
			points = (p1 - p0)*(self.shape_val-np.floor(self.shape_val)) + p0
		return points

class Base3D:
	def __init__(self, shapeobj, color=[0.5,0.5,0.5], resolution=2, normals_consistent_tangent_plane=20, num_smooth_interations=3, poisson_depth=9):
		self.shapeobj = shapeobj
		self.resolution = resolution
		self.normals_consistent_tangent_plane = normals_consistent_tangent_plane
		self.num_smooth_interations = num_smooth_interations
		self.poisson_depth = poisson_depth
		self.color = color
		self._postprocess = {"cloud":None,"vertices":None,"triangles":None,"color":None}
		self._vertices, self._triangles = None, None

	def add_postprocess(self, postprocess, process_type):
		old = self._postprocess[process_type]
		if old is None:
			new = postprocess
		elif callable(old):
			new = [old, postprocess]
		elif type(old) == list or type(old) == tuple:
			new = list(old)+[postprocess]
		else:
			raise Exception("Unknown old process type", type(old), type(postprocess))
		self._postprocess[process_type] = new

	def postprocess(self, item, process_type):
		process = self._postprocess[process_type]
		if process is None:
			pass
		elif callable(process):
			item = process(item)
		elif type(process)==list or type(process)==tuple:
			for proc in process:
				item = proc(item)
		else:
			raise Exception("Unknown process type", type(process))
		return item

	@property
	def vertices(self):
		return copy.deepcopy(self._vertices)

	@property
	def triangles(self):
		return self._triangles
	
	@property
	def mesh(self):
		assert not self.vertices is None, "Not initialized"
		assert not self.triangles is None, "Error, vertices are defined but not triangles, call this class again to set both"
		vertices = self.postprocess(self.vertices, "vertices")
		triangles = self.postprocess(self.triangles, "triangles")
		triangles = np.concatenate((np.sum(np.ones_like(triangles),axis=-1,keepdims=True),triangles), axis=-1) # format to pyvista format
		mesh = pv.PolyData(vertices, triangles)
		return mesh

	def cubic_grid(self, return_list_of_faces=False):
		x = np.linspace(-1,1,self.resolution)
		y = np.linspace(-1,1,self.resolution)
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
		if not return_list_of_faces: faces = np.vstack(faces)
		return faces

	def make_surface(self, data):
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(data)
		pcd.estimate_normals()
		pcd.orient_normals_consistent_tangent_plane(self.normals_consistent_tangent_plane)
		mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=self.poisson_depth)
		if self.num_smooth_interations: mesh = mesh.filter_smooth_simple(number_of_iterations=self.num_smooth_interations)
		points = np.asarray(mesh.vertices)
		faces = np.asarray(mesh.triangles)
		return points, faces

	def save(self, path):
		assert not self.vertices is None, "Not initialized"
		assert not self.triangles is None, "Error, vertices are defined but not triangles, call this class again to set both"
		np.savez_compressed(path, vertices=self.vertices, triangles=self.triangles)

	def load(self, path):
		data = np.load(path)
		self._vertices, self._triangles = data["vertices"], data["triangles"]

	def __call__(self, reset=False):
		# get point cloud of surface
		if self.vertices is None or self.triangles is None or reset:
			data = self.cubic_grid() # ~80 is good quality
			data = self.shapeobj(data)

			# get the triangles with poisson surface resonstruction
			data = self.postprocess(data, "cloud")
			self._vertices, self._triangles = self.make_surface(data)

		# polydata object
		color = self.postprocess(self.color,"color")
		return dict(mesh=self.mesh,color=color,smooth_shading=True)

class Shape3D(Base3D):
	def __init__(self, shape_val, path=None, **kw):
		shapeobj = ShapeProj(shape_val)
		self.shape_val = shape_val
		self.path = path
		super().__init__(shapeobj=shapeobj, **kw)

	@property	
	def filename(self):
		fname = f"sval-{self.shape_val}_res-{self.resolution}.npz"
		return os.path.join(self.path,fname)

	def __call__(self, *ar,**kw):
		if not self.path is None and os.path.exists(self.filename): self.load(self.filename)
		out = super().__call__(*ar,**kw)
		if not self.path is None and not os.path.exists(self.filename): self.save(self.filename)
		return out

class Node:
	def __init__(self, shape3d, is_show=True):
		self.available_postprocess_type = None
		self.shape3d = shape3d
		self.is_show = is_show
		self.check_shape(shape3d)
		self.postprocess = []
		self.children = []

	def check_shape(self, shape3d):
		assert hasattr(shape3d, "postprocess")
		assert hasattr(shape3d, "add_postprocess")
		assert hasattr(shape3d, "_postprocess")
		if not self.available_postprocess_type is None:
			for i in self.available_postprocess_type: assert i in shape3d._postprocess, f"invalid process_type, {process_type}. Available: {self.available_postprocess_type}"

	def add_child(self, child, overwrite=False):
		assert (not child in self.children) or overwrite, "child already exists"
		# make sure the child is a Node
		assert Node in inspect.getmro(child.__class__)
		child = self.apply_postprocesses(child) # apply transformations/postprocess to new child
		self.children.append(child)

	def apply_postprocesses(self, child):
		for postprocess, process_type in self.postprocess:
			child.add_postprocess(postprocess, process_type)
		return child

	def add_postprocess(self, postprocess, process_type):
		if not self.available_postprocess_type is None: assert process_type in self.available_postprocess_type, f"invalid process_type, {process_type}. Available: {self.available_postprocess_type}"
		self.shape3d.add_postprocess(postprocess, process_type)
		# apply new postprocess to old children
		for i in range(len(self.children)):
			self.children[i].add_postprocess(postprocess, process_type)
		# add to postprocess for handling new children
		self.postprocess.append((postprocess, process_type))

	def __call__(self):
		# get all descendents (list of kwargs to pv.Plotter().plot)
		out = [] if not self.is_show else [self.shape3d()]
		for child in self.children:
			out += child()
		# return list of kwargs to pv.Plotter().plot, keep in mind self.is_show
		return out
