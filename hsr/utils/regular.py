import numpy as np 
import importlib.util
##########################
# Config and Setup Utils #
##########################

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

def import_given_path(name, path):
	spec = importlib.util.spec_from_file_location(name, path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod

class Compare:
	@classmethod
	def num(cls,a,b):
		if np.all(abs(a - b)<1e-8):
			return True
		return False

	@classmethod	
	def type(cls,i,j):
		j = type(i)(j)
		if type(i) == dict:
			if not cls.dictionary(i,j):
				cprint.fail("check_dictionary", i, j)
				return False
		elif type(i) in [list, tuple]:
			if not cls.list(i,j):
				cprint.fail("check_list", i, j)
				return False
		else:
			if not cls.num(i,j):
				cprint.fail("check_num", i, j)
				return False
		return True

	@classmethod	
	def dictionary(cls,a,b):
		if not len(a) == len(b): 
			return False
		for k in a.keys():
			if not k in a: return False
		for k in a.keys():
			if not cls.type(a[k],b[k]): return False
		return True

	@classmethod	
	def list(cls,a, b):
		if not len(a) == len(b): return False
		for i,j in zip(a,b):
			if not cls.type(i,j): return False
		return True

class GPUMemoryUsageMonitor:
	def __init__(self):
		from pynvml.smi import nvidia_smi
		self.nvsmi = nvidia_smi.getInstance()
	def get_memory_usage(self, gpu_num=0):
		"""returns amount of memory used on gpus as string
		"""
		memory_usage = self.nvsmi.DeviceQuery('memory.free, memory.total')
		mem_dict = memory_usage["gpu"][gpu_num]["fb_memory_usage"]
		return str(mem_dict["total"]-mem_dict["free"])+" "+mem_dict["unit"]