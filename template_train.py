import tensorflow as tf
import time
from model import ModelHandler
from exec import FileLock, num_gpu
from utilities import Mask
import config as cfg 
import sys
import os

## TBD: dynamic switching of CUDA_VISIBLE_DEVICES. Currently doesn't work and only uses one

def train_models(path, queue_file, gnum=0):
	while 1:
		gnum=gnum%num_gpu
		try:
			os.environ["CUDA_VISIBLE_DEVICES"] = str(gnum)
			modelhandler = ModelHandler(base_path=path, train_new=True)
			modelhandler.save()
			modelhandler.train()
			break
		except tf.errors.ResourceExhaustedError:
			gnum=(gnum+1)%num_gpu
			time.sleep(5*60) # wait 5 min and try again
		except tf.errors.InternalError:
			gnum=(gnum+1)%num_gpu
		except tf.errors.UnknownError:
			time.sleep(60)


	lock = FileLock(queue_file)
	lock.lock()
	paths = []
	with open(queue_file) as f:
		# load paths and remove duplicates
		lines = f.readlines()
		for i in lines:
			if not path == i.replace("\n", ""):
				paths.append(i)

	with open(queue_file, "w") as f:
		for i in paths:
			f.write(i)
	lock.unlock()

if __name__ == '__main__':
	path = sys.argv[1]
	qfile = sys.argv[2]
	gnum = int(sys.argv[3])
	train_models(path, qfile, gnum)