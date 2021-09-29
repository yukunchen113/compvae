import tensorflow as tf 
import numpy as np
#######################
# Training/Loss Utils #
#######################

# reconstruction loss
class ImageMSE(): # mean squared error
	def __init__(self, loss_process=lambda x:x):
		self.loss_process = loss_process

	def __call__(self, actu, pred):
		reduction_axis = range(1,len(actu.shape))

		# per point
		loss = tf.math.squared_difference(actu, pred)

		# apply processing to first 3 channels
		loss = self.loss_process(loss)

		# per sample
		loss = tf.math.reduce_sum(loss, reduction_axis)
		# per batch
		loss = tf.math.reduce_mean(loss)
		return loss

class ImageBCE(): # binary cross entropy
	def __init__(self, loss_process=lambda x:x):
		self.loss_process = loss_process

	def __call__(self, actu, pred, label_smooting_pad=1e-5):
		reduction_axis = range(1,len(actu.shape))

		# apply label smooting
		actu = actu*(1-2*label_smooting_pad)+label_smooting_pad
		pred = pred*(1-2*label_smooting_pad)+label_smooting_pad

		# per point
		loss = actu*(-tf.math.log(pred))+(1-actu)*(-tf.math.log(1-pred))

		# apply processing to first 3 channels
		loss = self.loss_process(loss)

		# per sample
		loss = tf.math.reduce_sum(loss, reduction_axis)
		# per batch
		loss = tf.math.reduce_mean(loss)
		return loss

	def numpy(self, actu, pred, label_smooting_pad=1e-5):
		reduction_axis = list(range(1,len(actu.shape)))

		# apply label smooting
		actu = actu*(1-2*label_smooting_pad)+label_smooting_pad
		pred = pred*(1-2*label_smooting_pad)+label_smooting_pad

		# per point
		loss = actu*(-np.log(pred))+(1-actu)*(-np.log(1-pred))

		# apply processing to first 3 channels
		loss = self.loss_process(loss)

		# per sample
		loss = np.sum(loss, axis=tuple(reduction_axis))
		# per batch
		loss = np.mean(loss)
		return loss
		
# regularization loss
def kld_loss_reduction(kld_loss):
	# per batch
	kld_loss = tf.math.reduce_mean(kld_loss)
	return kld_loss

def kld_loss_reduction_numpy(kld_loss):
	# per batch
	kld_loss = np.mean(kld_loss)
	return kld_loss