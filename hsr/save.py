import functools
import dill
import os
import numpy as np
class Saver:
	"""
	basic saver, keeps a path and calls a save function on path
	"""
	def __init__(self,path):
		self.path = path

	def save(self, func):
		return func(self.path)

	def load(self, func):
		return func(self.path)

class InitSaver(Saver):
	"""
	Wraps a class, will pickle the initialization parameters and class at a specified path
	"""
	def __init__(self,path,ignore=[],is_return_class_kw=False):
		super().__init__(path)
		self.ignore = ignore
		self.is_return_class_kw = is_return_class_kw
	def save(self,class_obj,kw):
		removed_kw = [i for i in self.ignore if i in kw.keys()]
		kw = {k:v for k,v in kw.items() if not k in self.ignore}
		with open(self.path,"wb") as f:
			dill.dump([class_obj,kw,removed_kw],f)

	def load(self,**kw):
		if not os.path.exists(self.path): return None
		with open(self.path,"rb") as f:
			class_obj,kwargs,removed_kw = dill.load(f)
		for k in removed_kw: assert k in kw
		kwargs.update(kw)
		if self.is_return_class_kw:
			return class_obj, kwargs
		return class_obj(**kwargs)

	def __call__(self, class_obj):
		# method to save model + train class and initialization params
		class wrapper:
			def __init__(wobj,func):
				wobj.func = func
			def __get__(wobj,obj,cls):
				@functools.wraps(wobj.func)
				def new_func(*ar,**kw):
					assert not len(ar), "Saver only accepts kwarg version"
					self.save(cls,kw)
					return wobj.func(obj,**kw)
				new_func.__func__ = new_func.im_func = wobj.func
				new_func.__self__ = new_func.im_self = obj
				return new_func
		class_obj.__init__ = wrapper(class_obj.__init__)
		return class_obj

class ModelWeightSaver(Saver):
	def save(self,model):
		model.save_weights(self.path)
	def load(self,model):
		if os.path.exists(self.path): model.load_weights(self.path)

class ModelSaver(Saver):
	def __init__(self,path):
		assert os.path.isdir(path)
		super().__init__(path)
		self.weightsaver = ModelWeightSaver(os.path.join(self.path,"model_weights.h5"))
		self.modelsaver = InitSaver(os.path.join(self.path,"model_initialization.pickle"), is_return_class_kw=True)
		self.modelprocessingsaver = PickleSaver(os.path.join(self.path,"model_processing.pickle"))

	def save(self,*ar,**kw):
		"""
		Saves per step same as ModelWeightSaver
		"""
		return self.weightsaver.save(*ar,**kw)

	def apply_processing(self,model,model_processing):
		for mproc in model_processing:
			model = mproc(model)
		return model

	def load(self):
		ret = self.modelsaver.load()
		if ret is None: return None
		Model, kwargs = ret
		model_processing = self.modelprocessingsaver.load()
		if model_processing is None: model_processing = []
		Model = self.apply_processing(Model, model_processing)
		model = Model(**kwargs)
		self.weightsaver.load(model)
		return model

	def __call__(self,model,model_processing=[]):
		"""
		Saves initialization parameters, same as InitSaver
		"""
		model = self.modelsaver(model)
		self.modelprocessingsaver.save(model_processing)
		model = self.apply_processing(model, model_processing)
		return model

class NumpySaver(Saver):
	# uses numpy's savez function, wrapped here for consistency
	def save(self, **kw):
		np.savez(self.path,**kw)
	def load(self):
		if not os.path.exists(self.path): return None
		data = np.load(self.path, allow_pickle=True)
		return data
class PickleSaver(Saver):
	# uses numpy's savez function, wrapped here for consistency
	def save(self, kw):
		with open(self.path,"wb") as f:
			dill.dump(kw,f)
	def load(self):
		if not os.path.exists(self.path): return None
		with open(self.path,"rb") as f:
			data = dill.load(f)
		return data
class TrainSaver(Saver):
	def __init__(self,path):
		assert os.path.isdir(path)
		super().__init__(path)
		self.progress_saver = PickleSaver(os.path.join(self.path,"train_progress.npz"))
		self.trainsaver = InitSaver(os.path.join(self.path,"train_initialization.pickle"), ignore=["model"], is_return_class_kw=True)

	def save(self,**kw):
		"""
		Saves per step same as ModelWeightSaver
		"""
		return self.progress_saver.save(kw)

	def load(self, model):
		ret = self.trainsaver.load(model=model)
		if ret is None: return None,None
		Trainer,kwargs = ret
		trainer = Trainer(**kwargs)
		kw = self.progress_saver.load()
		return trainer, kw

	def __call__(self,*ar,**kw):
		"""
		Saves initialization parameters, same as InitSaver
		"""
		return self.trainsaver(*ar,**kw)