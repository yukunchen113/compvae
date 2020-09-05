"""
Viewing models


TBD:
- Main: experiments traversal (filepath traversal)
- Sub modules:
	- images across training
	- model access
		- traversals given data
		- Metrics between latents (for one batch)
			- KL between latents
"""

import PyQt5 as qt

import sys
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


import os
import numpy as np
import importlib.util
from utilities.standard import import_given_path, cprint
from utilities.vlae_method import vlae_traversal
from disentangle.loss import kl_divergence_between_gaussians
from collections import OrderedDict

class LimitedSizeDict(OrderedDict):
	# LRU dictionary from: https://stackoverflow.com/questions/2437617/how-to-limit-the-size-of-a-dictionary
	def __init__(self, *args, **kwds):
		self.size_limit = kwds.pop("size_limit", None)
		OrderedDict.__init__(self, *args, **kwds)
		self._check_size_limit()

	def __setitem__(self, key, value):
		OrderedDict.__setitem__(self, key, value)
		self._check_size_limit()

	def _check_size_limit(self):
		if self.size_limit is not None:
			while len(self) > self.size_limit:
				self.popitem(last=False)

class MetricsBetweenLatents:
	def __init__(self, path,codepath=None):
		# will only use the modelhandler in the path specified.
		
		# sets up path
		if codepath is None:
			codepath=os.path.join(path,"code","core","__init__.py")
		assert os.path.exists(codepath)
		self.path = path

		self.samples = None
		self.inputs = None
		self._traversal_image_num = 0
		self._model_num = -4
		self._possible_models=None
		
		# cache for recent state:
		self.traversal=LimitedSizeDict(size_limit=10)

		self._set_import(codepath)
		self.load_model()
		self.get_data()
		self.run_batch()
		self.get_traversal()


	def _set_import(self,path):
		self.core = import_given_path("core", path)

	@property
	def possible_models(self):
		if self._possible_models is None:
			self._possible_models = [os.path.dirname(i[0]) for i in os.walk(self.path) if "model_setup" in i[0]]
		return self._possible_models

	def load_model(self):
		basepath=self.possible_models[self.model_num]
		self.modelhandler = self.core.model.handler.ProVLAEModelHandler(basepath)
		cprint.blue("Model Basepath:",basepath)

	def get_data(self, images=None):
		if not images is None:
			inputs_test = self.modelhandler.config.inputs_test[images]
		else:
			inputs_test = self.modelhandler.config.inputs_test 
		inputs_test = self.modelhandler.config.preprocessing(inputs_test)
		self.inputs = inputs_test

	def run_batch(self):
		past_hp = self.modelhandler.config.hparam_schedule.past_hp
		self.recon = self.modelhandler.model(self.inputs, **past_hp)
		self.latents = np.asarray(self.modelhandler.model.latent_space)

	def get_traversal(self):
		if self.traversal_image_num in self.traversal:
			self.trav_image, self.samples = self.traversal[self.traversal_image_num]
			return

		trav_image = self.inputs[self.traversal_image_num]
		if len(trav_image.shape) < 4:
			trav_image=np.expand_dims(trav_image,0)
		self.samples = np.asarray(vlae_traversal(
					self.modelhandler.model,
					trav_image,
					min_value=-3, 
					max_value=3, 
					num_steps=30, 
					return_traversal_object=True).samples_list[1:])
		self.trav_image = trav_image
		self.traversal[self.traversal_image_num]=[self.trav_image, self.samples]
	
	def get_elements(self, layer_num=-1, latent_num=0, return_bounded=False):
		assert not self.samples is None, "self.get_traversal must be run first."
		# clip to bounds
		layer_num = np.clip(layer_num, -len(self.samples), len(self.samples)-1)
		latent_num = np.clip(latent_num, -self.samples.shape[2], self.samples.shape[2]-1)
		if return_bounded:
			return self.samples[layer_num,:,latent_num], layer_num, latent_num
		return self.samples[layer_num,:,latent_num]

	@property
	def traversal_image_num(self):
		return self._traversal_image_num
	@traversal_image_num.setter 
	def traversal_image_num(self, num):
		assert not self.inputs is None
		self._traversal_image_num=np.clip(num, 0, len(self.inputs)-1).astype(int)
		self.get_traversal()

	@property
	def model_num(self):
		return self._model_num
	@model_num.setter 
	def model_num(self, num):
		assert not self.inputs is None
		self._model_num=np.clip(num, 0, len(self.possible_models)-1).astype(int)
		self.get_traversal()

def kld_metric(first_ls, second_ls):
	kld = kl_divergence_between_gaussians(first_ls[0], first_ls[1], second_ls[0], second_ls[1])
	kld = np.mean(kld)
	return kld

class LatentElement:
	# this is used as interface between widgets
	def __init__(self):
		self.layer_val = -1
		self.latent_val = 0

#######
# GUI #
#######

class MplCanvas(FigureCanvasQTAgg):
	def __init__(self, parent=None):
		fig = Figure()
		self.axes = fig.add_subplot(111)
		super().__init__(fig)

"""
class ModelSelectionWidget(QtWidgets.QWidget):
	def __init__(self, metric_obj):
		super().__init__()
		self.metric_obj= metric_obj
		self.extra_calls = [] # any children calls that depend on changes from this.
		
		# create buttons
		self.label = QtWidgets.QLabel(self)

		next_b = QtWidgets.QPushButton('Next', self)
		next_b.clicked.connect(self.next)

		prev_b = QtWidgets.QPushButton('Prev', self)
		prev_b.clicked.connect(self.prev)

		refresh = QtWidgets.QPushButton('Refresh Test Data', self)
		refresh.clicked.connect(self.refresh_data)

		layout=QtWidgets.QHBoxLayout()
		layout.addWidget(self.label)
		layout.addWidget(prev_b)
		layout.addWidget(next_b)
		layout.addWidget(refresh)

		self.init()
		self.setLayout(layout)

	def init(self, refresh_data=False):
		self.label.setText("Model: %d %s"%list(enumerate(self.metric_obj.possible_models))[self.metric_obj.model_num])
		self.metric_obj.load_model()
		if refresh_data:
			self.metric_obj.get_data()
		self.metric_obj.run_batch()
		self.metric_obj.get_traversal()

		for func in self.extra_calls:
			func()
	@QtCore.pyqtSlot()
	def refresh_data(self):
		self.init(True)

	@QtCore.pyqtSlot()
	def prev(self):
		self.metric_obj.model_num = self.metric_obj.model_num-1
		self.init()

	@QtCore.pyqtSlot()
	def next(self):
		self.metric_obj.model_num = self.metric_obj.model_num+1
		self.init()
"""

class ImageInfoWidget(QtWidgets.QWidget):
	def __init__(self, metric_obj, img_canvas):
		super().__init__()
		self.metric_obj= metric_obj
		self.img_canvas=img_canvas
		self.extra_calls = [] # any children calls that depend on changes from this.

		# create buttons
		self.image_label = QtWidgets.QLabel(self)

		next_b = QtWidgets.QPushButton('Next', self)
		next_b.clicked.connect(self.next_image)

		prev_b = QtWidgets.QPushButton('Prev', self)
		prev_b.clicked.connect(self.prev_image)

		layout=QtWidgets.QHBoxLayout()
		layout.addWidget(self.image_label)
		layout.addWidget(prev_b)
		layout.addWidget(next_b)

		self.init_image()
		self.setLayout(layout)


	def init_image(self):
		self.img_canvas.axes.clear()
		self.img_canvas.axes.imshow(self.metric_obj.trav_image[0])
		self.img_canvas.draw()
		self.image_label.setText("Image: %d"%self.metric_obj.traversal_image_num)

		for func in self.extra_calls:
			func()

	@QtCore.pyqtSlot()
	def prev_image(self):
		self.metric_obj.traversal_image_num = self.metric_obj.traversal_image_num-1
		self.init_image()

	@QtCore.pyqtSlot()
	def next_image(self):
		self.metric_obj.traversal_image_num = self.metric_obj.traversal_image_num+1
		self.init_image()

class LatentTraversalWidget(QtWidgets.QWidget):
	def __init__(self, metric_obj, latent_element):
		super().__init__()
		self.metric_obj= metric_obj
		self.img_canvas=MplCanvas(self)
		self.extra_calls = [] # any children calls that depend on changes from this.

		# inital values
		self.latent_element = latent_element
		self.traversal_val=0

		# create buttons
		next_layer = QtWidgets.QPushButton('Next', self)
		next_layer.clicked.connect(lambda x=None: self.layer_back(is_next=True))
		prev_layer = QtWidgets.QPushButton('Prev', self)
		prev_layer.clicked.connect(lambda x=None: self.layer_back(is_next=False))
		self.layer_label =QtWidgets.QLabel(self)
		
		next_latent = QtWidgets.QPushButton('Next', self)
		next_latent.clicked.connect(lambda x=None: self.latent_back(is_next=True))
		prev_latent = QtWidgets.QPushButton('Prev', self)
		prev_latent.clicked.connect(lambda x=None: self.latent_back(is_next=False))
		self.latent_label = QtWidgets.QLabel(self)

		layer_layout = QtWidgets.QHBoxLayout()
		layer_layout.addWidget(self.layer_label)
		layer_layout.addWidget(prev_layer)
		layer_layout.addWidget(next_layer)
		
		latent_layout = QtWidgets.QHBoxLayout()
		latent_layout.addWidget(self.latent_label)
		latent_layout.addWidget(prev_latent)
		latent_layout.addWidget(next_latent)

		layout=QtWidgets.QVBoxLayout()
		layout.addLayout(layer_layout)
		layout.addLayout(latent_layout)
		layout.addWidget(self.img_canvas)

		timer = QtCore.QTimer(self)
		timer.timeout.connect(self.traverse_back)
		timer.start()

		self.init_image()
		self.setLayout(layout)

	def traverse_back(self):
		self.traversal_val=self.traversal_val+1
		self.init_image()

	def init_image(self):
		images, self.latent_element.layer_val, self.latent_element.latent_val = self.metric_obj.get_elements(
			self.latent_element.layer_val, self.latent_element.latent_val, return_bounded=True)
		self.img_canvas.axes.clear()
		images = np.vstack([images,images[::-1]])
		self.img_canvas.axes.imshow(images[self.traversal_val%len(images)])
		self.img_canvas.draw()

		#set labels
		self.layer_label.setText("Layer: %s"%self.latent_element.layer_val)
		self.latent_label.setText("Latent: %s"%self.latent_element.latent_val)
		for func in self.extra_calls:
			func()

	@QtCore.pyqtSlot()
	def layer_back(self,is_next=True):
		if is_next:
			self.latent_element.layer_val = self.latent_element.layer_val+1
		else:
			self.latent_element.layer_val = self.latent_element.layer_val-1
		self.init_image()

	@QtCore.pyqtSlot()
	def latent_back(self,is_next=True):
		if is_next:
			self.latent_element.latent_val = self.latent_element.latent_val+1
		else:
			self.latent_element.latent_val = self.latent_element.latent_val-1
		self.init_image()

class SingleKLDMetricWidget(QtWidgets.QWidget):
	def __init__(self, metric_obj, first_latent, second_latent, label_name="KLD Metric DKL(0|1): "):
		super().__init__()
		self.first_latent=first_latent
		self.second_latent=second_latent
		self.metric_obj = metric_obj
		self.label = QtWidgets.QLabel(self)
		layout=QtWidgets.QHBoxLayout()
		layout.addWidget(self.label)
		self.setLayout(layout)
		self.update_label()

	def update_label(self):
		# self.metric_obj.latents are of shape [num latent layers, [samples, mean, logvar], N, num latents]
		first_ls = self.metric_obj.latents[self.first_latent.layer_val,1:,:,self.first_latent.latent_val]
		second_ls = self.metric_obj.latents[self.second_latent.layer_val,1:,:,self.second_latent.latent_val]
		kld = kld_metric(first_ls, second_ls)
		self.label.setText("KLD Metric: %f"%kld)

class KLDMetricWidget(QtWidgets.QWidget):
	def __init__(self, metric_obj, first_latent, second_latent, first=0, second=None):
		super().__init__()
		if second is None:
			second = first+1
		self.kl=[]
		layout = QtWidgets.QHBoxLayout()
		self.kl.append(SingleKLDMetricWidget(metric_obj, first_latent, second_latent, label_name=f"KLD Metric DKL({first}|{first+1}): "))
		layout.addWidget(self.kl[-1])
		self.kl.append(SingleKLDMetricWidget(metric_obj, second_latent, first_latent, label_name=f"KLD Metric DKL({first+1}|{first}): "))
		layout.addWidget(self.kl[-1])
		self.setLayout(layout)
	def update_label(self):
		for i in self.kl:
			i.update_label()

class MetricsBetweenLatentsGUI(QtWidgets.QMainWindow):
	def __init__(self, path):
		super().__init__()

		# get data
		self.metric = MetricsBetweenLatents(path)

		# populate mainlayour with other content
		mainlayout = self.setup_gui_layout()

		# set layout 
		widget = QtWidgets.QWidget()
		widget.setLayout(mainlayout)
		self.setCentralWidget(widget)
		self.show()

	def setup_gui_layout(self):
		# create main layout
		mainlayout = QtWidgets.QVBoxLayout()

		#model_num=ModelSelectionWidget(self.metric)
		#mainlayout.addWidget(model_num)

		# display image information
		img_canvas = MplCanvas(self)
		image_info=ImageInfoWidget(self.metric, img_canvas)
		#model_num.extra_calls.append(image_info.init_image)
		mainlayout.addWidget(image_info)

		# disply image and selected latents 
		latent_elements = []
		latent_travs = []

		selected_latents=QtWidgets.QHBoxLayout()
		selected_latents.addWidget(img_canvas)

		num_view_latents = 2
		for i in range(num_view_latents): # add latents for viewing
			latent_elements.append(LatentElement())
			latent_travs.append(LatentTraversalWidget(self.metric, latent_elements[-1]))
			image_info.extra_calls.append(latent_travs[-1].init_image)
			selected_latents.addWidget(latent_travs[-1])

		mainlayout.addLayout(selected_latents)
		
		# display metrics
		metric_display=QtWidgets.QVBoxLayout()
		kld_metrics = []
		for i in range(num_view_latents-1):
			kld_metrics.append(KLDMetricWidget(self.metric,*latent_elements[i:i+2], first=i, second=i+1))
			for j in latent_travs[i:i+2]:
				j.extra_calls.append(kld_metrics[-1].update_label)
			metric_display.addWidget(kld_metrics[-1])
		mainlayout.addLayout(metric_display)

		return mainlayout


def main():
	path = "test/step_trigger_layer_introduction_lg/"
	app = QtWidgets.QApplication(sys.argv)
	gui = MetricsBetweenLatentsGUI(path)
	app.exec_()


if __name__ == '__main__':
	main()