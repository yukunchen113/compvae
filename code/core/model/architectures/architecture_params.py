import numpy as np
import tensorflow as tf
####################################
# ProVLAE Large Network Parameters #
####################################
vlae_encoder_layer_param_large64 = [
	[[64,4,2],["bn"]],
	[[128,4,2],["bn"]],
	[[256,4,2],["bn"]],
	[[512,4,2],["bn"]],
	[[["flatten"],[1024],["bn"]],# this is grouped together so alpha could be applied on it.
		[[1024],["bn"]]],
	]
vlae_decoder_layer_param_large64 = [
	[[[1024],["bn"]],# this is grouped together so alpha could be applied on it.
		[[1024],["bn"]],
		[[4*4*512],["bn"],["reshape",[4,4,512]]]],
	[[[512,4,1],["bn"]], [[256,4,2],["bn"]]], # resnet will be: ["resnet", [512,4,1], [256,4,2]]
	[[[128,4,2],["bn"]], [[64,4,1],["bn"]]],
	[64,4,2],
	[3,4,2],
	]
vlae_latent_spaces_large64 = [
	[# latent layer 1
		[[[[64,4,2],["bn"]],[[64,4,1],["bn"]]], ["flatten"]], # encoder
		lambda output_shape: [[int(np.prod(output_shape))], ["reshape", list(output_shape)]] # decoder
		], 
	[# latent layer 2
		[[[[128,4,2],["bn"]],[[256,4,1],["bn"]]], ["flatten"]], # encoder
		lambda output_shape: [[[int(np.prod(output_shape))], ["bn"]], ["reshape", list(output_shape)]] # decoder
		],
	[# latent layer 3
		[[[[256,4,2],["bn"]],[[512,4,1],["bn"]]], ["flatten"]], # encoder
		lambda output_shape: [[[int(np.prod(output_shape))], ["bn"]], ["reshape", list(output_shape)]] # decoder
		],
	]
vlae_latent_connections_large64 = [(0,-2),(1,-3),(2,-4)]

####################################
# ProVLAE Small Network Parameters #
####################################
"""
vlae_encoder_layer_param_small64 = [
	[[32,4,2],["bn"]],
	[[32,4,2],["bn"]],
	[[32,4,2],["bn"]],
	[[[32,4,2],["bn"]],# this is grouped together so alpha could be applied on it.
		[["flatten"],[256],["bn"]]],
	]
vlae_decoder_layer_param_small64 =[
	[[[256],["bn"]], # this is grouped together so alpha could be applied on it.
		[[4*4*32],["bn"],["reshape",[4,4,32]]],
		[[32,4,2],["bn"]]],
	[[32,4,2],["bn"]],
	[[32,4,2],["bn"]],
	[3,4,2],
	]
vlae_latent_spaces_small64 = [
	[#latent layer 1
		[[[32,4,2],["bn"]],[["flatten"],[256],["bn"]]],
		lambda output_shape: [[[256],["bn"]], [[int(np.prod(output_shape))],["bn"],["reshape", list(output_shape)]]] # decoder
		],
	[#latent layer 2
		[[[32,4,2],["bn"]],[["flatten"],[256],["bn"]]],
		lambda output_shape: [[[256],["bn"]], [[int(np.prod(output_shape))],["bn"],["reshape", list(output_shape)]]] # decoder
		],
	]
#"""
#"""
# Standard Parameters From https://openreview.net/pdf?id=SygagpEKwB
vlae_encoder_layer_param_small64 = [
	[[32,4,2],["bn"]],
	[[32,4,2],["bn"]],
	[[64,2,2],["bn"]],
	[[[64,2,2],["bn"]],# this is grouped together so alpha could be applied on it.
		[["flatten"],[256],["bn"]]],
	]
vlae_decoder_layer_param_small64 =[
	[[[256],["bn"]], # this is grouped together so alpha could be applied on it.
		[[4*4*64],["bn"],["reshape",[4,4,64]]],
		[[64,4,2],["bn"]]],
	[[32,4,2],["bn"]],
	[[32,4,2],["bn"]],
	[3,4,2],
	]
vlae_latent_spaces_small64 = [
	[#latent layer 1
		[[[32,4,2],["bn"]],[["flatten"],[256],["bn"]]],
		lambda output_shape: [[[256],["bn"]], [[int(np.prod(output_shape))],["bn"],["reshape", list(output_shape)]]] # decoder
		],
	[#latent layer 2
		[[[64,2,2],["bn"]],[["flatten"],[256],["bn"]]],
		lambda output_shape: [[[256],["bn"]], [[int(np.prod(output_shape))],["bn"],["reshape", list(output_shape)]]] # decoder
		],
	]
#"""
vlae_activations_encoder_small64 = {"default":tf.nn.leaky_relu, -1:tf.keras.activations.linear}
vlae_activations_decoder_small64 = {"default":tf.nn.leaky_relu, -1:tf.math.sigmoid}
vlae_latent_connections_small64 = [(0,-2),(1,-3)]

