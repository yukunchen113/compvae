import tensorflow as tf
import numpy as np
class _Base(tf.keras.layers.Layer):
	"""
	input tensors a and b.
	will output in the shape of b.
	"""
	def __init__(self, input_shape, output_shape):
		super().__init__()
		assert np.equal(input_shape[:-1], output_shape[:-1]).all()
		self._shape_input = input_shape
		self._shape_output = output_shape
		if input_shape[-1] == output_shape[-1]:
			self._input_proj = None
		else:
			self._input_proj= tf.keras.Sequential([
					tf.keras.Input(self._shape_input),
					tf.keras.layers.Conv2D(
						self._shape_output[-1],1,1,
						padding="same",
						input_shape=self._shape_input)])

	def input_proj(self, inputs):
		if not self._input_proj is None:
			inputs = self._input_proj(inputs)
		return inputs
	


class LinearConcat(_Base):
	"""concatenates two arrays. Projects one to the other for matching 
	"""
	def __init__(self,*ar,**kw):
		super().__init__(*ar,**kw)
		out_proj_input_shape = list(self._shape_input[:-1])+[self._shape_input[-1]+self._shape_output[-1]]
		self.output_proj = tf.keras.Sequential([
			tf.keras.Input(out_proj_input_shape), # we need this so we don't need to use data to build the weights
			tf.keras.layers.Conv2D(
				self._shape_output[-1],1,1,
				padding="same",
				input_shape=out_proj_input_shape)])

	def call(self, inputs, output=None):
		inputs = self.input_proj(inputs)
		if not output is None:
			output = tf.concat([inputs,output],axis=-1)
			output = self.output_proj(output)
		else:
			output = inputs
		return output

class ResidualConcat(_Base):
	pass

class ResidualAdd(_Base):
	"""
	adds two arrays. Projects one to the other for matching 
	"""
	def call(self, inputs, output=None):
		inputs = self.input_proj(inputs)
		if not output is None:
			output = inputs+output
		else:
			output = inputs
		return output

