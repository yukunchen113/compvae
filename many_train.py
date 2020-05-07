"""Trains many models at once through subprocess for isolation
"""
import subprocess

max_num_proc = 7
max_num_large_proc = 2
mask_procs = []
procs = [] 

mask_beta_value = 30
for beta_value in [15,30]:
	for l in [1,3,5,6,8,9]:
		a= subprocess.Popen(["python3.7 train_256.py %d %d %d exp_5"%(beta_value,l,mask_beta_value)], shell=True)
		procs.append(a)
		if len(procs) >= max_num_large_proc:
			for p in procs:
				p.wait()
			procs = []
	for p in procs:
		p.wait()

for p in procs:
	p.wait()

# nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9