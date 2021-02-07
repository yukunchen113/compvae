from disentangle.architectures import base, block
from disentangle.architectures.encoder import GaussianEncoder
from disentangle.architectures.decoder import Decoder
from . import architecture_params as ap
import tensorflow as tf

class LadderGaussianEncoder(GaussianEncoder):
	@classmethod
	def get_available_layer_types(cls):
		pool = base.AveragePooling2D
		batch_norm = base.BatchNormalization
		flatten = base.Flatten
		conv2d_obj = base.Conv2D
		conv2d_obj.default_kw = dict(padding="same") # add default keyword arguments, (default_kw only works with BaseTFWrapper instances)
		conv2d_opt_obj = block.create_option_block(conv2d_obj, pool, batch_norm)		
		dense_obj = base.Dense
		dense_opt_obj = block.create_option_block(dense_obj, batch_norm, flatten, pool)
		
		# ProVLAE specific instructions, use a NetworkBlock as default and make resnet into an option
		# networkblock
		class networkblock_obj(block.NetworkBlock):
			@classmethod
			def get_available_layer_types(cls):
				return [conv2d_obj, conv2d_opt_obj, dense_obj, dense_opt_obj]		
		networkblock_opt_obj = block.create_option_block(networkblock_obj, pool, batch_norm)

		# resnet
		class resnet_obj(block.ResnetBlock):
			@classmethod
			def get_available_layer_types(cls):
				return [conv2d_obj, conv2d_opt_obj]
		resnet_obj = base.OptionWrapper(resnet_obj, identifier="resnet")# make resnet an option
		resnet_opt_obj = block.create_option_block(resnet_obj, pool, batch_norm)

		return [pool, batch_norm, conv2d_obj, dense_obj, resnet_obj, conv2d_opt_obj, dense_opt_obj, resnet_opt_obj, flatten, networkblock_obj, networkblock_opt_obj]

class LadderGaussianEncoder64(LadderGaussianEncoder):
	def __init__(self, num_latents, activation=None, layer_param=None, **kwargs):
		"""
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		shape_input = [64,64,3]
		super().__init__(
			layer_param=layer_param, 
			num_latents=num_latents, 
			shape_input=shape_input, 
			activation=activation, 
			**kwargs)
	
class LadderDecoder(Decoder):
	@classmethod
	def get_available_layer_types(cls):
		# same as network.DeconvolutionalNeuralNetwork get_available_layer_types here:
		upscale = base.UpSampling2D
		upscale.default_kw = dict(interpolation="nearest")
		batch_norm = base.BatchNormalization
		reshape = base.Reshape
		conv2dtranspose_obj = base.Conv2DTranspose
		conv2dtranspose_obj.default_kw = dict(padding="same") # add default keyword arguments, (default_kw only works with BaseTFWrapper instances)
		conv2dtranspose_opt_obj = block.create_option_block(conv2dtranspose_obj, upscale, batch_norm)
		dense_obj = base.Dense
		dense_opt_obj = block.create_option_block(dense_obj, batch_norm, reshape, upscale)
		
		# ProVLAE specific instructions, use a NetworkBlock as default and make resnet into an option
		# networkblock
		class networkblock_obj(block.NetworkBlock):
			@classmethod
			def get_available_layer_types(cls):
				return [conv2dtranspose_obj, conv2dtranspose_opt_obj, dense_obj, dense_opt_obj]		
		networkblock_opt_obj = block.create_option_block(networkblock_obj, upscale, batch_norm)

		# resnet
		class resnet_obj(block.ResnetBlock):
			@classmethod
			def get_available_layer_types(cls):
				return [conv2dtranspose_obj, conv2dtranspose_opt_obj]
		resnet_obj = base.OptionWrapper(resnet_obj, identifier="resnet")# make resnet an option
		resnet_opt_obj = block.create_option_block(resnet_obj, upscale, batch_norm)

		return [upscale, batch_norm, conv2dtranspose_obj, dense_obj, resnet_obj, networkblock_obj, networkblock_opt_obj, conv2dtranspose_opt_obj, dense_opt_obj, resnet_opt_obj, reshape]

class LadderDecoder64(LadderDecoder):
	def __init__(self, num_latents, activation=None, layer_param=None, **kwargs):
		"""Decoder network for 64x64x3 images,
		used for to setup provlae
		
		Args:
		    activation (None, dict): This is a dictionary of specified actions
		"""
		shape_image = [64,64,3]
		super().__init__(
			layer_param=layer_param, 
			num_latents=num_latents, 
			shape_image=shape_image, 
			activation=activation, 
			is_create_sequential=False, # don't create sequential as we shouldn't build just yet - we need to redefine the layers.
			**kwargs)	

##################################
# Prebuilt Encoders and Decoders #
##################################
class LadderGaussianEncoderLarge64(LadderGaussianEncoder64):
	def __init__(self, num_latents=7, activation=None, **kwargs):
		"""This is a gaussian encoder that takes in 64x64x3 images
		This is the architecture used in ProVLAE literature for CelebA
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		super().__init__(
			num_latents=num_latents, 
			activation=activation, 
			layer_param=ap.vlae_encoder_layer_param_large64,
			**kwargs)

class LadderDecoderLarge64(LadderDecoder64):
	def __init__(self, num_latents=7, activation=None, **kwargs):
		"""Decoder network for 64x64x3 images
		This is the architecture used in ProVLAE literature for CelebA
		
		Args:
		    activation (None, dict): This is a dictionary of specified actions
		"""
		super().__init__(
			num_latents=num_latents, 
			activation=activation, 
			layer_param=ap.vlae_decoder_layer_param_large64, 
			**kwargs)

class LadderGaussianEncoderSmall64(LadderGaussianEncoder64):
	def __init__(self, num_latents=4, activation=None, **kwargs):
		"""This is a gaussian encoder that takes in 64x64x3 images
		This is the architecture used in ProVLAE literature for Shapes3D and dsprites
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		if activation is None:
			activation = ap.vlae_activations_encoder_small64
		super().__init__(
			num_latents=num_latents, 
			activation=activation, 
			layer_param=ap.vlae_encoder_layer_param_small64,
			**kwargs)

class LadderDecoderSmall64(LadderDecoder64):
	def __init__(self, num_latents=4, activation=None, **kwargs):
		"""Decoder network for 64x64x3 images
		This is the architecture used in ProVLAE literature for Shapes3D
		
		Args:
		    activation (None, dict): This is a dictionary of specified actions
		"""
		if activation is None:
			activation = ap.vlae_activations_decoder_small64
		super().__init__(
			num_latents=num_latents, 
			activation=activation, 
			layer_param=ap.vlae_decoder_layer_param_small64, 
			**kwargs)

		