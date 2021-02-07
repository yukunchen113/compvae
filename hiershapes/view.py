import numpy as np
import pyvista as pv
import vtk
from hiershapes.base import CubicGrid
import copy
 
class Background:
	def __init__(self, floor_color=[0.5,0.5,0.5], wall_color=[0.5,0.5,0.5], ceil_color=[0.5,0.5,0.5], wall_faces=[0,1,2,3]):
		self.resolution = 3
		self.cubic_grid = CubicGrid(resolution=self.resolution)
		self.default_parameters = dict(
				floor_color = floor_color,
				wall_color = wall_color,
				ceil_color = ceil_color,
				wall_faces = wall_faces)

	def __call__(self, **parameters):
		parameters = {**self.default_parameters,**parameters}
		faces = self.cubic_grid(return_list_of_faces=True)
		faces = [i*np.asarray([[20,12,3]])+np.asarray([[8,0,3]]) for i in faces]
		ceiling = pv.PolyData(faces[-1]).delaunay_2d()
		floor = pv.PolyData(faces[-2]).delaunay_2d()
		walls = None
		for i,wall in enumerate(faces[:-2]):
			if not i in parameters["wall_faces"]: continue
			wall = pv.PolyData(wall).delaunay_2d()
			if walls is None:
				walls=wall
			else:
				walls+=wall
		return dict(mesh=walls, color=parameters["wall_color"]), dict(
			mesh=floor, color=parameters["floor_color"]), dict(mesh=ceiling, color=parameters["ceil_color"])

class Camera:
	"""
	Camera position in polar coordinates, focal point is origin
	"""
	def __init__(self, r, theta, phi, translate=[0,0,0]):
		# base view
		self.default_parameters = dict(r=r, theta=theta, phi=phi, translate=translate)

	def __call__(self, **parameters):
		# radians
		parameters = {**self.default_parameters,**parameters}
		r, theta, phi, translate = parameters["r"], parameters["theta"], parameters["phi"], parameters["translate"]
		position = np.asarray([r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)])+np.asarray(translate)
		focal_point = np.asarray([0,0,0])+np.asarray(translate)
		up_view = [np.cos(theta), np.sin(theta), np.cos(np.pi/2-phi)]
		view = [position, focal_point, up_view]
		return view

class Lighting:
	def __init__(self, position=[2,2,3],focal_point=[0,0,0],color = [255,214,170,255], intensity=0.3):
		self.default_parameters = dict(
			color=color,
			focal_point=focal_point,
			position=position,
			intensity=intensity)
		
	def get_light(self, color, focal_point, position, intensity):
		colors = vtk.vtkNamedColors()
		colors.SetColor('color', color)
		light = vtk.vtkLight()
		light.SetFocalPoint(*focal_point)
		light.SetPosition(*position)
		light.SetColor(colors.GetColor3d('color'))
		light.SetIntensity(intensity)
		return light

	def get_pass(self,passes=None):
		passes = vtk.vtkRenderPassCollection()
		shadows = vtk.vtkShadowMapPass()
		seq = vtk.vtkSequencePass()
		passes.AddItem(shadows.GetShadowMapBakerPass())
		passes.AddItem(shadows)
		seq.SetPasses(passes)
		cameraP = vtk.vtkCameraPass() # Tell the renderer to use our render pass pipeline
		cameraP.SetDelegatePass(seq)
		return cameraP

	def __call__(self, plt, remove_existing_lights=False, **parameters):
		parameters = {**self.default_parameters,**parameters}
		if remove_existing_lights: plt.renderer.RemoveAllLights()
		plt.renderer.AddLight(self.get_light(
			color=parameters["color"],
			focal_point=parameters["focal_point"],
			position=parameters["position"],
			intensity=parameters["intensity"] ))
		plt.renderer.SetPass(self.get_pass())
		return plt
