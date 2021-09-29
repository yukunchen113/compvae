import train
from hsr.utils.multiprocess import ParallelRun
import time
import os
import copy 
def run_single(job_num=None):
	time.sleep(2) # this is for ports in docker
	jobs_path = os.path.join(copy.deepcopy(train.experimental_path), "jobs")
	datasets, jobs = train.get_single_jobs()	

	# Submit and Run Jobs #
	if job_num is None: 
		parallel = ParallelRun(exec_file=os.path.abspath(__file__), job_path=jobs_path, 
			num_gpu=2, max_concurrent_procs_per_gpu=4)
		parallel(*[str(i) for i in range(len(jobs))])
		for i in datasets: i.close(is_terminate_server=True)
	else:
		train.run_training(**jobs[job_num])
	raise Exception("Finished models")


import sys
if __name__ == '__main__':
	args=sys.argv
	train_parallel = run_single
	if len(args)>1:
		train_parallel(int(args[1]))
	else:
		train_parallel(None)