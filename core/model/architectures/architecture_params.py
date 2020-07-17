vlae_encoder_layer_params_large64 = [
	[64,4,2,"bn",None],
	[128,4,2,"bn",None],
	[256,4,2,"bn",None],
	[512,4,2,"bn",None],
	[1024,"bn"],
	[1024,"bn"],
	]
vlae_decoder_layer_params_large64 = [
	[1024,"bn"],
	[1024,"bn"],
	[[512,4,1,"bn"], [256,4,2,"bn"], None], # resnet will be: [[512,4,1], [256,4,2], "resnet", None], pooling/upscale must be last element
	[[128,4,2,"bn"], [64,4,1,"bn"], None],
	[64,4,2,"bn", None],
	[3,4,2, None],
	]
vlae_shape_before_flatten_large64 = [4,4,512]
vlae_latent_spaces_large64 = [
	[[64,4,2,None],[64,4,1,None]],
	[[128,4,2,None],[256,4,1,None]],
	[[256,4,2,None],[512,4,1,None]],
	]
vlae_latent_connections_small64 = [0,1]

vlae_encoder_layer_params_small64 = [
	[32,4,2,None],
	[32,4,2,None],
	[32,4,2,None],
	[32,4,2,None],
	[256],
	]
vlae_decoder_layer_params_small64 =[
	[256],
	[32,4,2,None],
	[32,4,2,None],
	[32,4,2,None],
	[3,4,2, None],
	]
vlae_shape_before_flatten_small64 = [4,4,32]
vlae_latent_spaces_small64 = [
	[[32,4,2,None],[256]],
	[[32,4,2,None],[256]],
	]
vlae_latent_connections_large64 = [0,1,2]

