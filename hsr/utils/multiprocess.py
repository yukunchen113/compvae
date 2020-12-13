import os
import sys
from hsr.save import InitSaver
import subprocess
import copy
import dill
import time
class ParallelRun:
	def __init__(self, exec_file, job_path, num_gpu, max_concurrent_procs_per_gpu):
		self.exec_file = exec_file
		self.job_path = job_path
		if not os.path.exists(job_path): os.makedirs(job_path)
		self.num_gpu = num_gpu
		self.max_concurrent_procs_per_gpu = max_concurrent_procs_per_gpu
		self.procs = None
	def submit_job(self, path, job):
		assert path.endswith(".pickle")
		with open(os.path.join(self.job_path, path),"wb") as f:
			dill.dump([self.exec_file, job], f)

	def make_proc(self, path, envar=None):
		# get execution command
		path = os.path.join(self.job_path,path)
		assert path.endswith(".pickle") and os.path.exists(path)
		with open(path,"rb") as f:
			execution = dill.load(f)
		process = ["python%d.%d"%sys.version_info[:2]]+execution

		# get envar if not specified
		if envar is None:
			envar=os.environ.copy()

		# create executable process
		proc = subprocess.Popen(process,env=envar)
		return proc
	
	def run(self, paths=None):
		if paths is None: paths = os.listdir(self.job_path)
		paths.sort(key=lambda x: int(x.replace(".pickle","")))
		self.procs=[[] for i in range(self.num_gpu)]
		idx=0
		finished = False
		while not finished:
			for cur_gpu in range(self.num_gpu):
				if idx<len(paths) and len(self.procs[cur_gpu])<self.max_concurrent_procs_per_gpu:
					envar=os.environ.copy()
					envar["CUDA_VISIBLE_DEVICES"] = str(cur_gpu)
					self.procs[cur_gpu].append(self.make_proc(paths[idx],envar=envar))
					time.sleep(0.5)#sleep for queue creation
					idx+=1
			try:
				finished=True
				for cur_gpu in range(self.num_gpu):
					self.procs[cur_gpu]=[i for i in self.procs[cur_gpu] if i.poll() is None]
					finished = finished and len(self.procs[cur_gpu])==0 and idx >= len(paths)
			except Exception as e:
				for cur_gpu in range(self.num_gpu):
					for i in self.procs[cur_gpu]:
						while not i.poll() is None:
							i.terminate()
				raise e
		print("Finished")
	
	def __exit__(self, *ar):
		for cur_gpu in range(self.num_gpu):
			for i in self.procs[cur_gpu]:
				while not i.poll() is None:
					i.terminate()

	def submit(self, *jobs):
		paths = []
		for i,job in enumerate(jobs):
			path = str(i)+".pickle"
			self.submit_job(path, job=job)
			paths.append(path)

	def __call__(self, *jobs):
		self.submit(*jobs)
		self.run()
