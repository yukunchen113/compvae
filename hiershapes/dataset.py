import pyvista as pv
from multiprocessing import Process, Value
import queue
import numpy as np
import hiershapes.utils as ut
import time
import pickle
class Batch():
	def __init__(self, scene, set_kw={}, randomize_parameters_func=None):
		self.scene = scene
		self.set_kw = set_kw
		self.batch_size = None
		self._plotter = None
		self.randomize_parameters_func = randomize_parameters_func

	@property
	def plotter(self):
		if self._plotter is None:
			self._plotter = pv.Plotter(off_screen=True)
		return self._plotter

	def __getstate__(self):
		d = dict(self.__dict__)
		d["_plotter"] = None
		return d

	def batch(self,batch_size):
		self.batch_size = batch_size
		return self

	def get_batch_labels(self, batch_size):
		labels = []
		for i in range(batch_size):
			if self.randomize_parameters_func is None: 
				parameters = self.scene.randomize_parameters(**self.set_kw)
			else:
				parameters = self.randomize_parameters_func(**self.set_kw)
			labels.append(parameters)
		return labels

	def get_images(self, parameters, plotter=None, return_labels=False):
		if plotter is None: 
			plotter = self.plotter
		images = []
		for i in parameters:
			plotter.clear() # it seems that clearing causes shadows to disappear when using xvfb
			if hasattr(plotter, "enable_lightkit"): plotter.enable_lightkit() # this is here to handle docker (which uses xvfb) which removes lightkit for some reason
			images.append(self.scene(plotter=plotter,**i))
		if return_labels:
			return np.asarray(images), parameters
		return np.asarray(images)

	def get_batch(self, batch_size):
		labels = self.get_batch_labels(batch_size=batch_size)
		return self.get_images(labels, plotter=self.plotter, return_labels=True)

	def __call__(self, batch_size=None):
		if batch_size is None: batch_size = self.batch_size
		return self.get_batch(batch_size)

	def __iter__(self):
		return self

	def __next__(self):
		return self.__call__()

class MultiProcessBatch(Batch):
	# generates randomized batches to queue multiprocessed which is then read and
	# directly sent to user through call.
	def __init__(self,*ar,num_proc=1,prefetch=1,run_once=False,**kw):
		super().__init__(*ar,**kw)

		# multiprocess
		assert num_proc >= 1
		assert prefetch >= 1
		self.prefetch = prefetch
		self.num_proc = num_proc
		self.run_once = run_once
		self.queue = ut.Queue(self.prefetch)
		self.processes = None

	@property
	def plotter(self):
		if self._plotter is None:
			self._plotter = [pv.Plotter(off_screen=True) for _ in range(self.num_proc+1)]
		return self._plotter

	def terminate(self):
		if not self.processes is None:
			self.queue.get_all_queue() # empty queue
			for proc in self.processes: 
				proc.terminate()
			for proc in self.processes:
				proc.join()
			self.processes = None


	def get_images(self, *ar, plotter=None, **kw):
		if plotter is None:
			plotter = self.plotter[-1]
		return super().get_images(*ar, plotter=plotter, **kw)
	
	def get_batch_multiproc(self, plotter):
		while True:
			labels = self.get_batch_labels(self.batch_size)
			images, labels = self.get_images(labels, plotter=plotter, return_labels=True)
			self.queue.put((images, labels))
			if self.run_once: 
				break
	
	def __getstate__(self):
		d = super().__getstate__()
		d["processes"] = None
		return d
	
	def start_procs(self):
		self.processes = []
		for i in range(self.num_proc): 
			process = Process(target=self.get_batch_multiproc, kwargs=dict(plotter=self.plotter[i]))
			process.start()
			self.processes.append(process)

	def __call__(self, get_all=False, is_blocking=True):
		if self.processes is None: self.start_procs()
		if self.run_once and not self.processes is None: 
			data = []# get the data from the queue first to prevent lock when joining
			for _ in self.processes:
				data.append(self.queue.get_queue(True))
			for proc in self.processes:
				proc.join()
			self.processes = None
			return data
		if get_all: return self.queue.get_all_queue() # this will return nothing if there is nothing in the queue
		return self.queue.get_queue(is_blocking)

	def __exit__(self):
		self.terminate()

class BatchPool:
	def __init__(self, pool_size=1, is_fill_pool=False):
		self.pool_size = pool_size
		self.is_fill_pool = is_fill_pool
		self.image_pool = None
		self.label_pool = None

	def add_data(self, images, labels):
		# stack all multiprocessed get.
		if self.image_pool is None:
			self.image_pool, self.label_pool = [None for _ in range(self.pool_size)],[None for _ in range(self.pool_size)]
		self.image_pool[1:]=self.image_pool[:-1]
		self.label_pool[1:]=self.label_pool[:-1]
		self.image_pool[0] = images
		self.label_pool[0] = labels

	def __getstate__(self):
		d = dict(self.__dict__)
		d["image_pool"] = None
		d["label_pool"] = None
		return d

	def __call__(self, batch_size, data):
		"""Fills the array with data, which should be a list of tuples of (image, labels)
		also, return a sample of the pool
		"""
		for images, labels in data: self.add_data(images, labels)
		if self.image_pool is None: return None


		image_pool = [i for i in self.image_pool if not i is None]

		if self.is_fill_pool and len(image_pool)<len(self.image_pool): 
			print(f"filling pool {ut.loading_bar(len(image_pool), len(self.image_pool))}", end = "\r")
			return None
		# pool
		image_pool = np.concatenate(image_pool, axis=0)
		label_pool = []
		for labels in self.label_pool: 
			if not labels is None: label_pool+=labels
		
		# sample from sample pool
		idx = np.random.permutation(np.arange(len(image_pool)))[:batch_size]
		images, labels = image_pool[idx], [label_pool[i] for i in idx]
		return images, labels

class ParallelBatch(Batch):
	def __init__(self,*ar,num_proc=1,prefetch=1,retrieval_batch_size=None,pool_size=None,is_fill_pool=True,**kw):
		super().__init__(*ar,**kw)
		if pool_size is None: pool_size = prefetch
		assert not "run_once" in kw, "ParallelBatch doesn't support run once."
		self._batch = MultiProcessBatch(*ar,num_proc=num_proc,prefetch=prefetch,**kw)
		self._batch.batch(retrieval_batch_size)
		self.pool = BatchPool(pool_size=pool_size,is_fill_pool=is_fill_pool)
	
	def __call__(self):
		out = None
		while out is None:
			data = self._batch(get_all=True)
			out = self.pool(batch_size=self.batch_size, data=data)
		return out

class Server(Batch):
	def __init__(self,*ar,num_proc=1,prefetch=1,retrieval_batch_size=None,port=None,verbose=True,**kw):
		super().__init__(*ar,**kw)
		assert not "run_once" in kw, "Server doesn't support run once."
		self._batch = MultiProcessBatch(*ar,num_proc=num_proc,prefetch=prefetch,**kw)
		self._batch.batch(retrieval_batch_size)
		self.server = ut.Server(port=port)
		self.verbose=verbose

	def __call__(self):
		# run the server
		accumulated_random_req = {}
		if self.verbose: timerobj = ut.Timer()
		if self.verbose: timerobj("Started Server")
		while True:
			requests = self.server.get_requests()
			out = {}
			for k,v in requests.items():
				if v == "random":
					accumulated_random_req[k] = v
				elif type(v) == list:
					try:
						out[k] = self.get_images(v)
					except Exception as e:
						out[k] = e
				else:
					out[k] = None
			data = self._batch(is_blocking=False)
			if not data is None:
				out.update({k:data for k,v in accumulated_random_req.items()})
				if self.verbose and len(accumulated_random_req): timerobj(f"New batch sent to {len(accumulated_random_req)} connections")
				accumulated_random_req = {}
			self.server.send_data(out)
	
	def __exit__(self):
		self.server.close()
		self._batch.terminate()

	def close(self):
		self.server.close()
		self._batch.terminate()

class Client:
	def __init__(self,prefetch=1,port=None,is_fill_pool=True,pool_size=None,verbose=True):
		self.verbose = verbose
		self.batch_size = None
		self.client = ut.Client(port=port)
		if pool_size is None: pool_size = prefetch
		self.pool = BatchPool(pool_size=pool_size, is_fill_pool=is_fill_pool)
		self.queue = ut.Queue(prefetch)
		self.processes = None

		self.term = Value("i",0)
		
	def set_port(self, port):
		self.client.set_port(port)

	def batch(self, batch_size):
		self.batch_size = batch_size
		return self

	def get_images(self, parameters, return_labels=False):
		images = self.client(parameters)
		if isinstance(images, Exception): raise images
		if return_labels: return images, parameters
		return images

	def monitor_server(self):
		while True:
			if self.term.value: break
			data = self.client("random")
			if self.term.value: break
			if data is None: continue
			self.queue.put(data)
			if self.verbose: print("got data...")
	def start_procs(self):
		self.processes = []
		process = Process(target=self.monitor_server)
		self.processes.append(process)
		process.start()

	def get_batch(self, batch_size):
		if self.processes is None: self.start_procs()
		out = None
		while out is None:
			data = self.queue.get_all_queue()
			out = self.pool(batch_size=batch_size, data=data)
		return out

	def close(self):
		if not self.processes is None:
			self.queue.get_all_queue()
			self.term.value = 1 # sends signal in shared mem to exit. this safely exits client
			for proc in self.processes:
				proc.join()
			self.processes = None

	def __call__(self):
		assert not self.batch_size is None, "batch_size not specified for client"
		return self.get_batch(self.batch_size)
	
	def __getstate__(self):
		d = dict(self.__dict__)
		d["processes"] = None
		d["term"] = None
		return d
	def __setstate__(self, d):
		self.__dict__ = d
		self.term = Value("i",0)
	
	def __exit__(self):
		self.close()

	def __iter__(self):
		return self

	def __next__(self):
		return self.__call__()

class ServerClient:
	def __init__(self,*ar,prefetch=1,port=None,is_fill_pool=True,pool_size=None,verbose=True,**kw):
		try:
			self.server = Server(*ar,prefetch=prefetch,port=port,verbose=verbose,**kw)
		except OSError:
			print("No Server...", end="")
			self.server = None
		time.sleep(3)
		self.client = Client(prefetch=prefetch,port=port,verbose=verbose,pool_size=pool_size,is_fill_pool=is_fill_pool)
		self.server_proc = None

	def close(self):
		# close server
		if not self.server_proc is None:
			self.server.close()
			self.server_proc.terminate()
			self.server_proc.join()

	def __call__(self):
		if not self.server is None: 
			print("Started Server...", end="")
			self.server_proc = Process(target=self.server)
			self.server_proc.start()
		print("Created Client")
		return self.client
