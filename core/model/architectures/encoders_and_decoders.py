from utils.tf_custom.architectures import base
from utils.tf_custom.architectures.encoders import GaussianEncoder
from utils.tf_custom.architectures.decoders import Decoder
import utilities.vlae_method as vlm
import tensorflow as tf
class _ProVLAEGaussianEncoder64(GaussianEncoder):
	def __init__(self, num_latents, layer_params, activations=None, **kwargs):
		"""This is a gaussian encoder that takes in 64x64x3 images
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		self.shape_input = [64,64,3]
		self.layer_params = layer_params
		if "num_channels" in kwargs:
			self.shape_input[-1] = kwargs["num_channels"]
		super().__init__(self.layer_params, num_latents, self.shape_input, activations, **kwargs)

class _ProVLAEDecoder64(Decoder):
	def __init__(self, num_latents, layer_params, shape_before_flatten, activations=None, **kwargs):
		"""Decoder network for 64x64x3 images,
		used for to setup provlae
		
		Args:
		    activations (None, dict): This is a dictionary of specified actions
		"""
		self.shape_image = [64,64,3]
		self.layer_params = layer_params
		self.shape_before_flatten = shape_before_flatten
		super().__init__(
			self.layer_params, 
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

	@classmethod
	def is_which_layer(cls, layer_param, is_separate=True):
		num = base.ConvNetBase.is_which_layer(layer_param, is_separate)
		if not num == 1:
			if is_separate:
				layer_param, _ = cls.separate_upscale_or_pooling_parameter(layer_param)

			# overwrite resnet
			if "resnet" in layer_param and base.ResnetBlock.is_layers_valid([i for i in layer_param if not i=="resnet"]):
				num = 3

			# add conv block as default
			if base.ConvBlock.is_layers_valid(layer_param):
				num = 4

		return num

class ProVLAEGaussianEncoderLarge64(_ProVLAEGaussianEncoder64):
	def __init__(self, num_latents=7, activations=None, **kwargs):
		"""This is a gaussian encoder that takes in 64x64x3 images
		This is the architecture used in ProVLAE literature for CelebA
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		super().__init__(
			num_latents=num_latents, 
			layer_params=vlm.vlae_encoder_layer_params_large64,
			activations=activations, 
			**kwargs)

class ProVLAEDecoderLarge64(_ProVLAEDecoder64):
	def __init__(self, num_latents=7, activations=None, **kwargs):
		"""Decoder network for 64x64x3 images
		This is the architecture used in ProVLAE literature for CelebA
		
		Args:
		    activations (None, dict): This is a dictionary of specified actions
		"""
		super().__init__(
			num_latents=num_latents, 
			layer_params=vlm.vlae_decoder_layer_params_large64, 
			shape_before_flatten=vlm.vlae_shape_before_flatten_large64, 
			activations=activations, 
			**kwargs)

class ProVLAEGaussianEncoderSmall64(_ProVLAEGaussianEncoder64):
	def __init__(self, num_latents=4, activations=None, **kwargs):
		"""This is a gaussian encoder that takes in 64x64x3 images
		This is the architecture used in ProVLAE literature for Shapes3D and dsprites
		
		Args:
			num_latents (int): the number of latent elements
			shape_input (list): the shape of the input image (not including batch size)
		"""
		super().__init__(
			num_latents=num_latents, 
			layer_params=vlm.vlae_encoder_layer_params_small64,
			activations=activations, 
			**kwargs)

class ProVLAEDecoderSmall64(_ProVLAEDecoder64):
	def __init__(self, num_latents=4, activations=None, **kwargs):
		"""Decoder network for 64x64x3 images
		
		Args:
		    activations (None, dict): This is a dictionary of specified actions
		"""
		super().__init__(
			num_latents=num_latents, 
			layer_params=vlm.vlae_decoder_layer_params_small64, 
			shape_before_flatten=vlm.vlae_shape_before_flatten_small64, 
			activations=activations, 
			**kwargs)

		