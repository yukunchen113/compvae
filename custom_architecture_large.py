import hsr
import hsr.model as md
import tensorflow as tf
import numpy as np
# architectures:
encoder_layers = [
	[[[64,4,2],["bn"]]],
	[[[128,4,2],["bn"]]],
	[[[256,2,2],["bn"]]],
	[[[512,2,2],["bn"]],[["flatten"],[1024],["bn"]]],
	]
decoder_layers =[
	[[[1024],["bn"]],[[4*4*512],["bn"],["reshape",[4,4,512]]],
		[[256,4,2],["bn"]]],
	[[[256,4,1],["bn"]],[[128,4,2],["bn"]]],
	[[[128,4,2],["bn"]],[[64,4,1],["bn"]]],
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

import disentangle.architectures.vae as vae
class BetaVAE(vae.BetaVAE):
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
		self._setup()

class BetaTCVAE(vae.BetaTCVAE):
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
		self._setup()

def main():
	dataset = hsr.dataset.Shapes3D()
	test_data = dataset.preprocess(dataset.test())
	lvae = LVAE(num_latents=4)
	out = lvae(test_data)
	# see layer outputs
	print("Encoder")
	for layer in lvae.encoder.layers.layers:
		print(layer.output_shape)
	print("Decoder - REMEMBER:latent depth is doubled for ladder layers")
	for layer in lvae.decoder.layers.layers:
		print(layer.input_shape)
	print(out.shape)

if __name__ == '__main__':
	main()