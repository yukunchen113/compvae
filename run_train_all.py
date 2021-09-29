import train
from hsr.utils.multiprocess import ParallelRun
import time
import os
import copy 
def run_all(job_num=-1, group_num=-1):
	# get all models (small+large+larger)
	alljobs = []
	
	_, jobs = train.get_ladder_jobs()
	alljobs.append(jobs)
	print("ladder jobs: ",len(jobs))
	
	_, jobs = train.get_single_jobs()
	alljobs.append(jobs)
	print("single jobs: ",len(jobs))

	_, jobs = train.get_large_ladder_jobs()
	alljobs.append(jobs)
	print("larger ladder jobs: ",len(jobs))

	_, jobs = train.get_large_single_jobs()
	alljobs.append(jobs)
	print("larger single jobs: ",len(jobs))

	if not group_num == -1: alljobs = [alljobs[group_num]]

	jobs = []
	for j in alljobs: jobs+=j
	print("Total: ",len(jobs))

	if job_num == -1: 
		for i,j in enumerate(jobs): print(i,j["path"])
	assert not job_num == -1, "job_num must be specified"
	# Submit and Run Jobs

	train.run_training(**jobs[job_num])
	print("Finished model")


import sys
if __name__ == '__main__':
	args=sys.argv
	train_parallel = run_all
	if len(args)>1:
		train_parallel(*[int(i) for i in args[1:]])
	else:
		train_parallel()