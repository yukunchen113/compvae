from utils.tf_custom.architectures import base
from utils.tf_custom.architectures.encoders import GaussianEncoder
from utils.tf_custom.architectures.decoders import Decoder
from utils.tf_custom.architectures.variational_autoencoder import BetaTCVAE, BetaVAE 
from utils.other_library_tools.disentanglementlib_tools import total_correlation 
from utils.tf_custom.loss import kl_divergence_with_normal, kl_divergence_between_gaussians
from utilities.standard import is_weighted_layer, get_weighted_layers, split_latent_into_layer
import utilities.vlae_method as vlm
import numpy as np
import tensorflow as tf
import os
import copy
from functools import reduce

class LatentSpace(tf.keras.layers.Layer):
	def __init__(self, layer_params, shape, num_latents, activation=tf.keras.activations.linear, name="LatentSpace"):
		super().__init__(name=name)
		self.num_latents = num_latents
		self.activation = activation

		# build model
		self.latent_layer = GaussianEncoder(layer_params, self.num_latents, shape, activations=None)

		# future set:
		self.decode_layer = None
		self.latent_space = None

	def set_decode(self, shape):
		fshape = reduce(lambda x,y: x*y, shape)
		input_layer = tf.keras.Input(self.num_latents)
		dense_layer = tf.keras.layers.Dense(fshape, activation=self.activation)
		reshape_layer = tf.keras.layers.Reshape(shape)
		self.decode_layer = tf.keras.Sequential([input_layer, 
				dense_layer, reshape_layer])

	def run_decode(self, samples):
		return self.decode_layer(samples)

	def call(self, inputs):
		# will create latent space block given inputs
		self.latent_space = self.latent_layer(inputs)
		return self.latent_space

class ProVLAEGaussianEncoder64(GaussianEncoder):
	def __init__(self, num_latents=10, activations=None, **kwargs):
		"""This is a gaussian encoder that takes in 64x64x3 images
		This is the architecture used in beta-VAE literature
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		self.shape_input = [64,64,3]
		self.layer_params = vlm.vlae_encoder_layer_params
		if "num_channels" in kwargs:
			self.shape_input[-1] = kwargs["num_channels"]
		super().__init__(self.layer_params, num_latents, self.shape_input, activations, **kwargs)

class ProVLAEDecoder64(Decoder):
	def __init__(self, num_latents=10, activations=None, **kwargs):
		"""Decoder network for 64x64x3 images
		
		Args:
		    activations (None, dict): This is a dictionary of specified actions
		"""
		self.shape_image = [64,64,3]
		self.layer_params = vlm.vlae_decoder_layer_params
		self.shape_before_flatten = vlm.vlae_shape_before_flatten
		super().__init__(self.layer_params, 
			num_latents=num_latents, 
			shape_image=self.shape_image, 
			activations=activations, 
			shape_before_flatten=self.shape_before_flatten, **kwargs)

	def build_sequential(self): # do not build sequential so we can inject inputs
		pass 

	def create_conv2d_layers(self, layer_p, layer_num, conv2d_obj=tf.keras.layers.Conv2D):
		layer_type = self.is_which_layer(layer_p, is_separate=False)
		layer_p = [i for i in layer_p if not type(i)==str]
		layer = super().create_conv2d_layers(layer_p=layer_p, layer_num=layer_num, conv2d_obj=conv2d_obj)
		# identify new type of layer
		if layer_type == 4:
			layer = base.ConvBlock(*layer_p,
								activation=self._apply_activation(layer_num),
								conv2d_obj=conv2d_obj
								)
			# can add layer processing here, such as batch norm. needs to be output as one layer, so use tf.keras.Sequential()
		return layer

	@staticmethod
	def is_which_layer(layer_param, is_separate=True):
		num = base.ConvNetBase.is_which_layer(layer_param, is_separate)
		if not num == 1:
			if is_separate:
				layer_param, _ = ProVLAEDecoder64.separate_upscale_or_pooling_parameter(layer_param)

			# overwrite resnet
			if "resnet" in layer_param and base.ResnetBlock.is_layers_valid([i for i in layer_param if not i=="resnet"]):
				num = 3

			# add conv block as default
			if base.ConvBlock.is_layers_valid(layer_param):
				num = 4

		return num




def set_shape(layer, shape):
	sequence = tf.keras.Sequential([
		tf.keras.Input(shape), layer])
	return sequence

class ProVLAEBase(BetaVAE):
	def create_default_encoder(self, **kwargs):
		# default encoder decoder pair:
		self.create_default_provlae_encoder(**kwargs)

	def create_default_provlae_encoder(self, **kwargs):
		self._encoder = ProVLAEGaussianEncoder64(**kwargs)
		self._decoder = ProVLAEDecoder64(**kwargs)
		self._setup()

class ProVLAE(ProVLAEBase):
	"""
	Creation:
		- create base VAE architecture,
		- create latent bridge between encoder and decoder given layer selection
			- create linear encoder latent space
			- double the num channels in corresponding decoder architecture to prep
				for concatenation
			- reshape encoder latent space 


	Requirements:
		- access any layer of architecture through layer num

	Assumptions
		- encoder and decoder are mirrored

	TODO:
		- specified layers only work with conv layers right now, not dense layers
	"""
	def __init__(self, beta, latent_connections=None, gamma=0.5, name="ProVLAE", **kwargs):
		self.latent_connections = latent_connections
		self.gamma = gamma
		self.ls_layer_params = vlm.vlae_latent_spaces if not "ls_layer_params" in kwargs else kwargs["ls_layer_params"]
		assert len(latent_connections) == len(self.ls_layer_params), "the latent connections should be matched to the number of \
			latent params len(latent_connections) = %d, len(self.ls_layer_params) = %d"%(len(latent_connections), len(self.ls_layer_params))
		super().__init__(name=name, beta=beta, **kwargs)

		self._alpha = None
		self.latent_space = None

	def _setup(self):
		# redefine encoder and decoder
		self._setup_encoder()
		self._setup_decoder()
		self._check_valid_architecture()

	def _check_valid_architecture(self):
		#for i,j in zip(self._encoder_layer_sizes, self._decoder_layer_sizes[::-1]):
		#	if not tuple(i)==tuple(j):
		#		string = "\n".join(["ENC:%s DEC:%s"%(i,j) for i,j in zip(self._encoder_layer_sizes,self._decoder_layer_sizes[::-1])])
		#		raise Exception("Invalid ProVLAE architecture, output of layers must be reflected\n%s"%string)
		pass

	def _setup_encoder(self):
		#print("ENCODER:")
		self._encoder_layer_sizes = [tuple(self._encoder.shape_input)]
		layer_objects = self._encoder.layer_objects
		model_layers = get_weighted_layers(layer_objects.layers)

		available_layers = len(model_layers)-2  # do not include last 2 layers, which is the last latent layer
		if not self.latent_connections is None:
			for i in self.latent_connections:
				assert i < available_layers, "invalid specified layers. Must be any of the following %s"%(list(range(available_layers)))

		# get latent connections:
		if self.latent_connections is None:
			lc = range(available_layers)
		else:
			lc = self.latent_connections
		
		# get latent connections (layer creation)
		self.latent_layers = []
		for i,layer in enumerate(model_layers):
			self._encoder_layer_sizes.append(layer.output_shape[1:])
			if i in lc:
				#print("Connecting layer:", layer.name)
				shape = layer.output_shape[1:]
				self.latent_layers.append(LatentSpace(
					layer_params=self.ls_layer_params[len(self.latent_layers)], 
					shape=shape, num_latents=self.num_latents, name="LatentSpace_%d"%(len(self.latent_layers))))
			else:
				self.latent_layers.append(None)


		# set call
		def call(inputs):
			latent_space = []
			enc_layers = self._encoder.layer_objects.layers
			valid_layer_num = 0
			layer_output = inputs
			for layer in enc_layers:
				layer_output = layer(layer_output)
				if is_weighted_layer(layer): # this if will not be triggered for last two layers (latent layer and layer leading up to it)
					ls = self.latent_layers[valid_layer_num]
					if not ls is None:
						latent_space.append(ls(self.alpha[len(latent_space)]*layer_output))
					valid_layer_num+=1
			latent_space.append(split_latent_into_layer(self.alpha[len(latent_space)]*layer_output, self.num_latents)) # highest level latent
			return latent_space

		self._encoder.call = call

	def _setup_decoder(self):
		#print("\nDECODER:")
		self._decoder_layer_sizes = [(None, self._decoder.shape_input)]
		layer_objects = self._decoder.layer_objects
		layers = layer_objects.layers
		modified_decoder_layers = []
		for model_layer, latent_layer in zip(layers, self.latent_layers[::-1]):
			# setup decoder latent space
			if not latent_layer is None:
				shape = self._decoder_layer_sizes[-1][1:]
				latent_layer.set_decode(shape) # this will set original due to shallow copy
		
			# setup decoder input channels
			#"""
			input_shape = list(self._decoder_layer_sizes[-1])[1:]
			layer_name = model_layer.name
			if not latent_layer is None:
				with tf.name_scope("modified_%s"%layer_name) as scope:
					input_shape[-1] *=2
					set_shape(model_layer, input_shape) # this successfully modifies the input even through the layer.input_shape would still be unchanged
			else:
				with tf.name_scope("%s"%layer_name) as scope:
					set_shape(model_layer, input_shape)
			#"""
			self._decoder_layer_sizes.append(model_layer.output_shape)

		def call(latent_space):
			dec_layers = self._decoder.layer_objects.layers
			layer_output = latent_space[-1] # get samples from last layer
			valid_layer_num = -1
			alpha_num = -1
			for layer in dec_layers:
				ls = self.latent_layers[valid_layer_num]
				if not ls is None:
					samples = latent_space[alpha_num-1]
					additional_recon = self.alpha[alpha_num]*ls.run_decode(samples)
					layer_output = tf.concat((
						additional_recon,
						layer_output),-1)
					alpha_num-=1
				valid_layer_num-=1
				layer_output = layer(layer_output)
			reconstruction = layer_output
			return reconstruction
		self._decoder.call = call

	def get_config(self):
		config_param = {
			**super().get_config(),
			"latent_connections":str(self.latent_connections),
			"gamma":str(self.gamma)}
		return config_param

	def get_latent_space(self):
		return self.latent_space

	@property
	def alpha(self):
		if self._alpha is None:
			self._alpha = [1]*len(self._encoder.layer_objects.layers) # latent_space size is unknown, so use all possible as worst case
		return self._alpha
	@alpha.setter
	def alpha(self, alpha=None):
		self._alpha = alpha


	def call(self, inputs, alpha=None, beta=None):
		# alpha is list of size num latent_space
		#called during each training step and inference
		#TBD: run latent space and next layer in parallel
		self.alpha = alpha
		if beta is None:
			beta = [self.beta]*len(self.alpha)


		self.latent_space = self.encoder(inputs)
		reconstruction = self.decoder([i[0] for i in self.latent_space])
		# get reconstruction and regularization loss
		for losses in self.provlae_regularizer(self.latent_space, self.alpha, beta):
			self.add_loss(losses)

		return reconstruction

	def provlae_regularizer(self, latent_space, alpha, beta):
		#regularizer creation for each specified layer.
		# use regularizations from parent vae
		losses = []
		for i,ls in enumerate(latent_space):
			if alpha[i]:
				losses.append(beta[i]*self.regularizer(*ls))
			else:
				losses.append(self.gamma*self.regularizer(*ls))
		return losses

class CondVAE(BetaTCVAE):
	def __init__(self, beta, name="CondVAE", **kwargs):
		super().__init__(name=name, beta=beta, **kwargs)

	def call(self, inputs, cond_logvar, cond_mean, gamma, latent_to_condition):
		self.gamma = np.clip(gamma, 0, 0.75).astype(np.float32)
		self.latent_to_condition = latent_to_condition

		self.latest_sample, self.latest_mean, self.latest_logvar = self.encoder(inputs)
		reconstruction = self.decoder(self.latest_sample)
		self.add_loss(self.regularizer(self.latest_sample, self.latest_mean, self.latest_logvar, cond_logvar, cond_mean))
		return reconstruction

	def regularizer(self, sample, mean, logvar, cond_logvar, cond_mean):
		# regular tcvae regularization loss
		model_kl_loss = kl_divergence_with_normal(mean, logvar)
		tc = (self.beta - 1) * total_correlation(sample, mean, logvar)

		# condition one of the mask latents onto the entire cond representation. Use mean to normalize 
		latent_focused_mean = tf.expand_dims(mean[:,self.latent_to_condition], -1)
		latent_focused_logvar = tf.expand_dims(logvar[:,self.latent_to_condition], -1)
		cond_kl_loss = kl_divergence_between_gaussians(latent_focused_mean, latent_focused_logvar, cond_logvar, cond_mean)
		cond_kl_loss = self.gamma*tf.math.reduce_mean(cond_kl_loss,1)+(1-self.gamma)*model_kl_loss[:,self.latent_to_condition]

		mask = tf.math.not_equal(tf.range(self.num_latents), self.latent_to_condition)

		non_conditioned_model_kl_loss = tf.boolean_mask(model_kl_loss, mask, axis=1)
		model_kl_loss = tf.math.reduce_sum(non_conditioned_model_kl_loss,1)+cond_kl_loss

		return tc + model_kl_loss

def test_condvae():
	inputs = np.random.normal(size=(32,64,64,3))
	cond_logvar = np.random.normal(size=(32,10)).astype(np.float32)
	cond_mean = np.random.normal(size=(32,10)).astype(np.float32)


	mask = CondVAE(10)
	#mask(inputs)
	mask(inputs, cond_sampled, cond_logvar, cond_mean)
	print(len(mask.losses))

def test_provlae():
	model = ProVLAE()


if __name__ == '__main__':
	test_provlae()