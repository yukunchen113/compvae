import hsr
import hsr.model as md
import tensorflow as tf
import numpy as np
from disentangle.loss import kl_divergence_with_normal, kl_divergence_between_gaussians
# architectures:
encoder_layers = [
	[64,1,2],["resnet",[[64,4,1],["bn"]],[[64,4,1],["bn"]]],
	[128,1,2],["resnet",[[128,4,1],["bn"]],[[128,4,1],["bn"]]],
	[256,1,2],["resnet",[[256,2,1],["bn"]],[[256,2,1],["bn"]]],
	[[[512,2,2],["bn"]],[["flatten"],[1024],["bn"]]],
	]
decoder_layers =[
	[[[1024],["bn"]],[[4*4*512],["bn"],["reshape",[4,4,512]]],[[256,4,2],["bn"]]],
	[256,1,2],["resnet",[[256,4,1],["bn"]],[[256,4,1],["bn"]]],
	[128,4,1],[128,1,1],["resnet",[[128,4,1],["bn"]],[[128,4,1],["bn"]]],
	[64,4,2],[64,1,1],["resnet",[[64,4,1],["bn"]],[[64,4,1],["bn"]]],
	[3,4,2],
	]


ladder_params = [
	[#latent layer 1
		[[[128,4,2],["bn"]],[[256,4,1],["bn"]],[["flatten"],[1024],["bn"]]],
		lambda output_shape: [[[1024],["bn"]], [[int(np.prod(output_shape))],["bn"],["reshape", list(output_shape)]]] # decoder
		],
	[#latent layer 2
		[[[128,2,2],["bn"]],[[256,2,1],["bn"]],[["flatten"],[1024],["bn"]]],
		lambda output_shape: [[[1024],["bn"]], [[int(np.prod(output_shape))],["bn"],["reshape", list(output_shape)]]] # decoder
		],
	]
# activations
encoder_activations = {"default":tf.nn.leaky_relu,
	0:tf.keras.activations.linear, # projection layer
	2:tf.keras.activations.linear, # projection layer
	4:tf.keras.activations.linear, # projection layer
	-1:tf.keras.activations.linear}
decoder_activations = {"default":tf.nn.leaky_relu, 
	1:tf.keras.activations.linear, # projection layer
	3:tf.keras.activations.linear, # projection layer
	5:tf.keras.activations.linear, # projection layer
	-1:tf.math.sigmoid}
ladder_connections = [(1,-3),(3,-6)]# (encoder output, decoder input)

#ladder_connections,ladder_params = [],[]

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
	dataset = hsr.dataset.HierShapesBoxheadSimple2(use_preloaded=True)
	test_data = dataset.preprocess(dataset.test())
	model = LVAE(num_latents=4)
	out = model(test_data)
	# see layer outputs
	print("Encoder")
	for i,layer in enumerate(model.encoder.layers.layers):
		print(i,layer.output_shape)
	print("Decoder - REMEMBER:latent depth is doubled for ladder layers")
	for i,layer in enumerate(model.decoder.layers.layers[:-1]):
		print(i-len(model.decoder.layers.layers),layer.input_shape, layer.output_shape)
	print(out.shape)

if __name__ == '__main__':
	main()