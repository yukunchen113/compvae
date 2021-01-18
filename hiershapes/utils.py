import socket
import pickle
import time
import multiprocessing
import numpy as np
import colorsys
import queue
################
# Server Utils #
################
class BaseSocket:
	def __init__(self, port=None):
		self.host = "localhost"
		if port is None: port = 65334
		self.port = port
		self.recv_chunk = 1024
	
	def send(self, data, conn):
		try:
			data = pickle.dumps(data)
			conn.sendall(data)
		except ConnectionResetError:
			pass

	def recv(self, conn):
		data = b''
		while True:
			chunk = conn.recv(self.recv_chunk)
			data = data+chunk
			if len(chunk)<self.recv_chunk: break
		try:
			return pickle.loads(data)
		except pickle.UnpicklingError:
			return None # error with packet, 

class Server(BaseSocket):

	"""Server that sends dataset group. 
	Will listen for connections prior to new set creation and send batch of data to all these connections.
	Connections must request again after batch is sent.
	"""
	def __init__(self, *ar, **kw):
		super().__init__(*ar,**kw)
		self.server_init()

	def server_init(self):
		self.server = socket.socket()
		self.server.setblocking(0) # so we can accept connections
		self.server.bind((self.host,self.port))
		self.server.listen()
	
	def send_data(self, data):
		for k,v in data.items():
			self.send(v,k)

	def get_requests(self):
		connections = []
		while True:
			try:
				connections.append(self.server.accept())
			except BlockingIOError:
				break
		data = {}
		for conn,addr in connections:
			try:
				data[conn] = self.recv(conn)
			except ConnectionResetError:
				pass
		time.sleep(0.5)
		#if len(connections): print(f"Received {len(connections)} clients")
		return data

	def close(self):
		self.server.shutdown(1)
		self.server.close()

class Client(BaseSocket):
	"""Connects to server to retrieve data. 
	After data is sent, must be called again to reconnect to retrieve next batch of data	
	"""
	def set_port(self, port):
		if not port is None: self.port = port

	def __call__(self, message):
		self.server = socket.socket()
		self.server.connect((self.host, self.port))
		data = self.send(message, self.server)
		data = self.recv(self.server)
		self.server.close()
		return data

class Queue:
	def __init__(self, *ar, **kw):
		self.ar = ar
		self.kw = kw
		self.queue=multiprocessing.Queue(*ar,**kw)
	
	def put(self,*ar,**kw):
		return self.queue.put(*ar,**kw)

	def get(self,*ar,**kw):
		return self.queue.get(*ar,**kw)

	def get_queue(self, is_blocking=True):
		if is_blocking: return self.queue.get()
		try:
			return self.queue.get(False)
		except queue.Empty:
			return None

	def get_all_queue(self):
		data = []
		while True:
			try:
				data.append(self.queue.get(False))
				time.sleep(0.05) # sleep time is needed to allow the queue to readjust, otherwise will say empty
			except queue.Empty:
				break
		return data

	def __getstate__(self):
		d = dict(self.__dict__)
		d["queue"] = None
		return d

	def __setstate__(self,d):
		self.__dict__ = d
		if self.queue is None:
			self.queue = multiprocessing.Queue(*self.ar, **self.kw)

###############
# Shape Utils #
###############
def subtract_mesh(block, subtracted_shape):
	def subtract(mesh):
		if type(subtracted_shape) == list:
			for submesh in subtracted_shape: mesh = mesh.boolean_difference(submesh.mesh)
		else:
			mesh = mesh.boolean_difference(submesh)
		return mesh

	subtract = Timer().funcwrap(subtract, "subtract")

	block.shape3d.add_postprocess(subtract, "mesh")
	return block

def quantized_uniform(low, high, size=(), n_quantized=10):
	np.random.seed()
	items = np.round(np.random.uniform(0,1,size=size)*n_quantized)/n_quantized
	items = items*(high-low)+low
	return items
def quantized_normal(mean, stddev, size=(), n_quantized=10):
	np.random.seed()
	items = np.round(np.random.normal(0,1,size=size)*n_quantized)/n_quantized
	items = items*stddev+mean
	return items

class HSVToRGB:
	def __init__(self, percent_saturation=100, percent_value=100):
		self.saturation = percent_saturation/100
		self.value = percent_value/100
	def __call__(self, hue, saturation=None, value=None):
		if saturation is None: saturation=self.saturation
		if value is None: value=self.value
		assert hue <= 1 and hue >= 0, "hue must be between 0 and 1"
		return colorsys.hsv_to_rgb(hue,saturation,value)

class HLSToRGB:
	def __init__(self, lightness=0.5, saturation=1):
		self.saturation = saturation
		self.lightness = lightness
	def __call__(self, hue, lightness=None, saturation=None):
		if saturation is None: saturation=self.saturation
		if lightness is None: lightness=self.lightness
		assert hue <= 1 and hue >= 0, "hue must be between 0 and 1"
		return colorsys.hls_to_rgb(hue,lightness,saturation)

class Parameters:
	def __init__(self, is_filter_val=True):
		self.keys = []
		self.items = {}
		self.is_filter_val = is_filter_val
	
	def add_parameters(self, key, func, dependences=[]):
		assert not key in self.keys, "key already exists"
		if self.keys == []: assert dependences == [], "first key must have no dependences"
		for i in dependences: assert i in self.keys, f"{i} must be specified, current specified: {self.keys}"
		self.keys.append(key)
		self.items[key] = [func, dependences]
	
	def __call__(self, **set_keys):
		ret = {}
		out = {}
		for key in self.keys:
			func, dependences = self.items[key]
			out[key] = func(**{k:v for d in dependences for k,v in out[d].items()})
			ret.update(out[key])
		for k,v in set_keys.items():
			ret[k] = v
		if self.is_filter_val: ret = {k:v for k,v in ret.items() if not k.startswith("_")}
		return ret

#################
# Regular Utils #
#################
def loading_bar(cur, total):
	# from https://github.com/yukunchen113/disentangle/blob/master/src/disentangle/general/tools.py
	fraction = cur/total
	string = "[%-20s]\t%.2f%%\t%d/%d\t\t"%("="*int(20*fraction), fraction*100, cur, total)
	return string

class Timer():
	"""
	# from https://github.com/yukunchen113/disentangle/blob/master/src/disentangle/general/tools.py
	object to keep track of time
	"""
	def __init__(self):
		self.start_time = time.time()
		self.past_time = time.time()

	def __call__(self, *args, **kwargs):
		return self.print(*args, **kwargs)

	def funcwrap(self, func, string):
		def newfunc(*ar,**kw):
			self.print(string="Started...")
			out = func(*ar,**kw)
			self.print(string=string)
			return out
		return newfunc

	def print(self, string="", return_string=False):
		new_string = "%s Past Time: %f, Total Time: %f"%(string, 
			time.time() - self.past_time, time.time() - self.start_time)
		self.past_time = time.time()
		if return_string:
			return new_string
		print(new_string)

class cprint: 
	# color codes from https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	@classmethod
	def header(cls, *ar,**kw):
		print(cls.HEADER, *ar, cls.ENDC, **kw)
	@classmethod
	def blue(cls, *ar,**kw):
		print(cls.BLUE, *ar, cls.ENDC, **kw)
	@classmethod
	def green(cls, *ar,**kw):
		print(cls.GREEN, *ar, cls.ENDC, **kw)
	@classmethod
	def warning(cls, *ar,**kw):
		print(cls.WARNING, *ar, cls.ENDC, **kw)
	@classmethod
	def fail(cls, *ar,**kw):
		print(cls.FAIL, *ar, cls.ENDC, **kw)
	@classmethod
	def bold(cls, *ar,**kw):
		print(cls.BOLD, *ar, cls.ENDC, **kw)	
	@classmethod
	def underline(cls, *ar,**kw):
		print(cls.UNDERLINE, *ar, cls.ENDC, **kw)

class Compare:
	def __init__(self,precision=1e-5, verbose=False):
		self.precision = precision
		self.verbose=verbose

	def num(self,a,b):
		if np.all(abs(a - b)<self.precision):
			return True
		return False

	def type(self,i,j):
		if type(i) == dict:
			if not self.dictionary(i,j):
				if self.verbose: cprint.fail("check_dictionary", i, j)
				return False
		elif type(i) in [list, tuple]:
			if not self.list(i,j):
				if self.verbose: cprint.fail("check_list", i, j)
				return False
		else:
			if not self.num(i,j):
				if self.verbose: cprint.fail("check_num", i, j)
				return False
		return True

	def dictionary(self,a,b):
		if not len(a) == len(b): 
			return False
		for k in a.keys():
			if not k in a: return False
		for k in a.keys():
			if not self.type(a[k],b[k]): return False
		return True

	def list(self,a, b):
		if not len(a) == len(b): return False
		for i,j in zip(a,b):
			if not self.type(i,j): return False
		return True

	def __call__(self,a,b):
		return self.type(a,b)