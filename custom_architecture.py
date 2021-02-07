"""
For BoxheadOneEye,

This architecture set will show transformations of the overall inscribed square but not the components.

This also depends on how correlated the factors are (inscribed square with cube color), if correlation is too strong, 

the architecture will blend in the colors into just cube color

"""
import hsr
import hsr.model as md
import tensorflow as tf
import numpy as np
from disentangle.loss import kl_divergence_with_normal, kl_divergence_between_gaussians
# architectures:
encoder_layers = [
	[[[32,4,2],["bn"]]],
	[[[64,4,2],["bn"]]],
	[[[128,2,2],["bn"]]],
	[[[256,2,2],["bn"]],[["flatten"],[512],["bn"]]],
	]
decoder_layers =[
	[[[512],["bn"]],[[4*4*256],["bn"],["reshape",[4,4,256]]],
		[[128,4,2],["bn"]]],
	[[[128,4,1],["bn"]],[[64,4,2],["bn"]]],
	[[[64,4,2],["bn"]],[[32,4,1],["bn"]]],
	[3,4,2],
	]
ladder_params = [
	[#latent layer 1
		[[[64,4,2],["bn"]],[[128,4,1],["bn"]],[["flatten"],[512],["bn"]]],
		lambda output_shape: [[[512],["bn"]], [[int(np.prod(output_shape))],["bn"],["reshape", list(output_shape)]]] # decoder
		],
	[#latent layer 2
		[[[64,2,2],["bn"]],[[128,2,1],["bn"]],[["flatten"],[512],["bn"]]],
		lambda output_shape: [[[512],["bn"]], [[int(np.prod(output_shape))],["bn"],["reshape", list(output_shape)]]] # decoder
		],
	]
# activations
encoder_activations = {"default":tf.nn.leaky_relu, -1:tf.keras.activations.linear}
decoder_activations = {"default":tf.nn.leaky_relu, -1:tf.math.sigmoid}
ladder_connections = [(0,-1),(1,-2)]# (encoder output, decoder input)

class LVAE(md.vae.LVAE):
	def create_default_vae(self, **kwargs):
		#self.create_large_ladder64(**kwargs)
		self.create_custom_vae(**kwargs)

	def create_custom_vae(self, **kwargs):
		# default encoder decoder pair:
		self._encoder = md.encoders_and_decoders.LadderGaussianEncoder64(
			activation = encoder_activations,
			layer_param = encoder_layers,
			**kwargs)
		self._decoder = md.encoders_and_decoders.LadderDecoder64(
			activation = decoder_activations,
			layer_param = decoder_layers, 
			**kwargs)
		self.latent_connections = ladder_connections
		self.ladder_params = ladder_params
		self._setup()

class VLAE(md.vae.VLAE):
	def create_default_vae(self, **kwargs):
		#self.create_large_ladder64(**kwargs)
		self.create_custom_vae(**kwargs)

	def create_custom_vae(self, **kwargs):
		# default encoder decoder pair:
		self._encoder = md.encoders_and_decoders.LadderGaussianEncoder64(
			activation = encoder_activations,
			layer_param = encoder_layers,
			**kwargs)
		self._decoder = md.encoders_and_decoders.LadderDecoder64(
			activation = decoder_activations,
			layer_param = decoder_layers, 
			**kwargs)
		self.latent_connections = ladder_connections
		self.ladder_params = ladder_params
		self._setup()

from disentangle.other_library_tools.disentanglementlib_tools import total_correlation 
import disentangle.architectures.vae as vae
class BetaVAE(md.vae.VLAE):
	def create_default_vae(self, **kwargs):
		#self.create_large_ladder64(**kwargs)
		self.create_custom_vae(**kwargs)

	def create_custom_vae(self, **kwargs):
		# default encoder decoder pair:
		self._encoder = md.encoders_and_decoders.LadderGaussianEncoder64(
			activation = encoder_activations,
			layer_param = encoder_layers,
			**kwargs)
		self._decoder = md.encoders_and_decoders.LadderDecoder64(
			activation = decoder_activations,
			layer_param = decoder_layers, 
			**kwargs)
		self.latent_connections = []
		self.ladder_params = []
		self._setup()

class BetaTCVAE(md.vae.VLAE):
	def create_default_vae(self, **kwargs):
		#self.create_large_ladder64(**kwargs)
		self.create_custom_vae(**kwargs)

	def create_custom_vae(self, **kwargs):
		# default encoder decoder pair:
		self._encoder = md.encoders_and_decoders.LadderGaussianEncoder64(
			activation = encoder_activations,
			layer_param = encoder_layers,
			**kwargs)
		self._decoder = md.encoders_and_decoders.LadderDecoder64(
			activation = decoder_activations,
			layer_param = decoder_layers, 
			**kwargs)
		self.latent_connections = []
		self.ladder_params = []
		self._setup()

	def layer_regularizer(self, sample, mean, logvar, cond_mean=None, cond_logvar=None, beta=None, *,return_kld=False,**kw):
		# base regularization method
		assert not (cond_mean is None != cond_logvar is None), "mean and logvar must both be sepecified if one is specified"
		if cond_mean is None:
			cond_mean = tf.zeros_like(mean)
			cond_logvar = tf.zeros_like(logvar)
		if beta is None:
			beta = self.beta
		kld = kl_divergence_between_gaussians(mean, logvar, cond_mean, cond_logvar)
		
		# tc loss
		kl_loss = tf.reduce_sum(kld,1)
		tc = (beta - 1) * total_correlation(sample, mean, logvar)
		kl_loss += tc

		if not return_kld:
			return kl_loss
		return kl_loss, kld

def main():
	dataset = hsr.dataset.Shapes3D()
	test_data = dataset.preprocess(dataset.test())
	lvae = LVAE(num_latents=4)
	out = lvae(test_data)
	# see layer outputs
	print("Encoder")
	for layer in lvae.encoder.layers.layers:
		print(layer.output_shape)
	print("Decoder")
	for layer in lvae.decoder.layers.layers:
		print(layer.input_shape)
	print(out.shape)

if __name__ == '__main__':
	main()