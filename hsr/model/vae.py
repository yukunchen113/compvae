"""
TBD: latent resample is hard coded
TBD: latent shortcut doesn't account for multiple parents

"""
from disentangle.architectures.vae import BetaTCVAE, BetaVAE, VariationalAutoencoder
from disentangle.other_library_tools.disentanglementlib_tools import total_correlation 
from disentangle.loss import kl_divergence_with_normal, kl_divergence_between_gaussians
from . import architecture_params as ap
from . import encoders_and_decoders as ead
from . import bidirectional_junction as bij
import numpy as np
import tensorflow as tf
import os
import copy
from functools import reduce, wraps

def split_into_latent_layer(inputs, num_latents):
	mean = inputs[:,:num_latents]
	logvar = inputs[:,num_latents:]
	sample = tf.exp(0.5*logvar)*tf.random.normal(
		tf.shape(logvar))+mean
	return sample, mean, logvar

def set_shape(layer, shape):
	sequence = tf.keras.Sequential([
		tf.keras.Input(shape), layer])
	return sequence

def is_weighted_layer(layer):
	return bool(layer.weights)

def check_weighted_layers(layers):
	for l in layers:
		assert is_weighted_layer(l), "each layer must contain weights. For example, tf.keras.layers.Dense. layer name: %s. %s"%(
			l.name)

class LatentSpace(VariationalAutoencoder):
	def __init__(self, name="LatentSpace"):
		super().__init__(name=name)

	def create_default_vae(self, **kwargs):
		# no default vae.
		self._encoder = None
		self._decoder = None

	def set_encoder(self, *ar, **kw):
		# build model
		assert self._encoder is None, "Encoder is already defined for %s"%(self.name)
		self._encoder = ead.LadderGaussianEncoder(*ar, **kw)

	def set_decoder(self, *ar, **kw):
		assert self._decoder is None, "Decoder is already defined for %s"%(self.name)
		self._decoder = ead.LadderDecoder(*ar, **kw)
	
	def call(self, *ar, **kw):
		assert not (self.decoder is None or self.encoder is None), "Decoder or encoder is not defined"
		return super().call(*ar, **kw)

class BidirectionalLatentSpace(LatentSpace):
	def __init__(self, *ar,**kw):
		self._bidirectional_proj = None
		self._prior_encoder = None
		super().__init__(*ar,**kw)

	def set_bidirectional(self):
		if not self._decoder is None and not self._encoder is None:
			self._bidirectional_proj = bij.LinearConcat(
				self._decoder.layers.layers[-1].output_shape[1:], 
				self._encoder.layers.layers[0].input_shape[1:])
	@property
	def prior_encoder(self):
		return self._prior_encoder
	
	def set_encoder(self, *ar, **kw):
		prior_kw = {**kw}
		prior_kw["layer_param"] = [["flatten"]] # just flatten the array, prior space will be constructed from a linear function of this.
		
		super().set_encoder(*ar,**kw)
		self._prior_encoder = ead.LadderGaussianEncoder(*ar, **prior_kw)
		self.set_bidirectional() #try setting bidirectional

	def set_decoder(self, *ar, **kw):
		super().set_decoder(*ar,**kw)
		self.set_bidirectional() #try setting bidirectional

		# hp handling
		decoder_call = self._decoder.call
		def call(*ar,alpha=1,beta=1,**kw):
			return decoder_call(*ar,**kw)*alpha
		self._decoder.call=call

	def bidirectional_encode(self,inputs,outputs=None,alpha=1,beta=1):
		# inputs should be from decoder, outputs should be from encoder, if encoder is None, then will just project decoder to shape.
		assert not self._bidirectional_proj is None, "encoder or decoder is not activated"
		projected_outputs = self._bidirectional_proj(inputs, outputs)*alpha
		if outputs is None:
			projected_outputs = self.prior_encoder(projected_outputs)
		else:
			projected_outputs = self.encoder(projected_outputs)
		return projected_outputs

class LadderBase(BetaVAE):
	# accepts the additional and wraps the specified functions
	def __init__(self, beta=1, name="LadderBase", **kwargs):
		# beta will be used as the top layer beta value
		super().__init__(name=name, beta=beta, **kwargs)
		# custom ladders
		self.latent_connections = self.latent_connections if (not "latent_connections" in kwargs) or (kwargs["latent_connections"] is None) else kwargs["latent_connections"]
		self.ladder_params = self.ladder_params if (not "ladder_params" in kwargs) or (kwargs["ladder_params"] is None) else kwargs["ladder_params"]
		#dynamic
		self.latent_space = None
		self._past_kld = None

	# default architectures
	def create_default_vae(self, **kwargs):
		# default encoder decoder pair:
		self.create_small_ladder64(**kwargs)

	def create_small_ladder64(self, **kwargs):
		self._encoder = ead.LadderGaussianEncoderSmall64(**kwargs)
		self._decoder = ead.LadderDecoderSmall64(**kwargs)
		self.latent_connections = ap.vlae_latent_connections_small64
		self.ladder_params = ap.vlae_latent_spaces_small64
		self._setup()

	def create_large_ladder64(self, **kwargs):
		self._encoder = ead.LadderGaussianEncoderLarge64(**kwargs)
		self._decoder = ead.LadderDecoderLarge64(**kwargs)
		self.latent_connections = ap.vlae_latent_connections_large64
		self.ladder_params = ap.vlae_latent_spaces_large64
		self._setup()

	@property
	def ladder_params(self):
		assert len(self.latent_connections) == len(self._ladder_params), ("the latent connections should be matched to the number of"+
			"latent params len(latent_connections) = %d, len(self.ladder_params) = %d"%(len(self.latent_connections), len(self._ladder_params)))
		return self._ladder_params
	@ladder_params.setter
	def ladder_params(self, params):
		assert len(self.latent_connections) == len(params), ("the latent connections should be matched to the number of"+
			"latent params len(latent_connections) = %d, len(self.ladder_params) = %d"%(len(self.latent_connections), len(params)))
		self._ladder_params = params

	def _setup(self):
		# redefine encoder and decoder
		self._setup_encoder()
		self._setup_decoder()

	def _setup_encoder(self):
		raise Exception("Not Implemented error.")

	def _setup_decoder(self):
		raise Exception("Not Implemented error.")

	def get_config(self):
		config_param = {
			**super().get_config(),
			"ladder_param":str(self.ladder_params),
			"latent_connections":str(self.latent_connections),}
		return config_param

	def get_latent_space(self):
		return self.latent_space

	@property
	def past_kld(self):
		return self._past_kld

	def call(self, inputs, **kw):
		raise Exception("Not Implemented error.")

	def layer_regularizer(self, sample, mean, logvar, cond_mean=None, cond_logvar=None, beta=None, *,return_kld=False,**kw):
		# base regularization method
		assert not (cond_mean is None != cond_logvar is None), "mean and logvar must both be sepecified if one is specified"
		if cond_mean is None:
			cond_mean = tf.zeros_like(mean)
			cond_logvar = tf.zeros_like(logvar)
		kld = kl_divergence_between_gaussians(mean, logvar, cond_mean, cond_logvar)
		if beta is None:
			beta = self.beta
		kl_loss = beta*kld
		kl_loss = tf.reduce_sum(kl_loss,1)
		if not return_kld:
			return kl_loss
		return kl_loss, kld

class VLAE(LadderBase):
	# simple VLAE setup, no bidirection.
	def __init__(self,*ar, gamma=0.5,name="VLAE",**kw):
		self.gamma=gamma
		super().__init__(*ar,name=name,**kw)
		self._alpha = None
	# hyper parameters
	@property
	def alpha(self):
		if self._alpha is None:
			self._alpha = [1]*(len(self.latent_connections)+1) # latent_space size is unknown, so use all possible as worst case
		return self._alpha
	@alpha.setter
	def alpha(self, alpha=None):
		assert alpha is None or len(alpha) == (len(self.latent_connections)+1)
		self._alpha = alpha
	def _setup(self):
		self._setup_encoder()
		self._setup_decoder()
		self._setup_encoder_call()
		self._setup_decoder_call()
	# architecture connections
	def _setup_encoder(self):
		"""
		- Creates the latent space objects
			- basic check on latent_connections
		- Replaces encoder call to include 
			- all latent space outputs as a list
			- fade in wrt alpha
		"""
		# perform some checks
		layers = self.encoder.layers.layers # the first layers is the sequential layers.
		check_weighted_layers(layers)
		assert not self.latent_connections is None, "must specify latent layers"
		for lc_enc, _ in self.latent_connections:
			assert lc_enc in range(-len(layers), len(layers)), "encoder latent connection specifications is invalid. latent connections %s and num layers is %d"%(lc, len(layers)) 

		# create the latent layers
		self.ladders = [] 
		latent_connections = {lc[0]:i for i,lc in enumerate(self.latent_connections)}
		for i,layer in enumerate(layers):
			if (i in latent_connections) or (i-len(layers) in latent_connections):
				if i-len(layers) in latent_connections: i = i-len(layers)
				# create new latent space
				ladder = LatentSpace(name="LatentSpace_%d"%latent_connections[i])
				ladder.set_encoder(
					layer_param=self.ladder_params[latent_connections[i]][0], 
					num_latents=self.num_latents, 
					shape_input=layer.output_shape[1:], 
					activation=None
					)
				self.ladders.append(ladder)
	def _setup_decoder(self):
		"""
		- Sets up latent layer decoder.
			- allow function layer_param (passes shape as arg) or list layer_param
		- Sets up each decoder layer inputs
		- sets up call by
			- applying fade in and concatenates
			- returns final reconstruction
		"""
		# perform some checks
		layers = self.decoder.layers # decoder did not convert sequential, so we use layers instead of layers.layers
		assert not self.latent_connections is None, "must specify latent layers"
		for _, lc_dec in self.latent_connections:
			assert lc_dec in range(-len(layers), len(layers)), "decoder latent connection specifications is invalid. latent connections %s and num layers is %d"%(lc, len(layers)) 

		latent_connections = {lc[1]:i for i,lc in enumerate(self.latent_connections)}
		shape_input = list(self.decoder.shape_input) 
		for i,layer in enumerate(layers): # this is for the intermediate ladders			
			#print("TEST:","decoder input",shape_input)
			if (i in latent_connections) or (i-len(layers) in latent_connections):
				if i-len(layers) in latent_connections: i = i-len(layers)
				# set ladder decoder 
				ladder_param = self.ladder_params[latent_connections[i]][1]
				if callable(ladder_param):
					ladder_param = ladder_param(shape_input)
				self.ladders[latent_connections[i]].set_decoder(
					layer_param=ladder_param, 
					num_latents=self.num_latents, 
					shape_image=shape_input, 
					activation=None
					)

				# increase decoder input channels
				with tf.name_scope("modified_%s"%layer.name) as scope:
					modified_decoder_shape = [*shape_input]
					modified_decoder_shape[-1] *=2
					set_shape(layer, modified_decoder_shape)

			else:
				with tf.name_scope("%s"%layer.name) as scope:
					set_shape(layer, shape_input)

			shape_input = list(layer.output_shape[1:])
	def _setup_encoder_call(self):
		layers = self.encoder.layers.layers
		latent_connections_map = {lc[0]:i for i,lc in enumerate(self.latent_connections)}
		
		def call(inputs,**kw):
			latent_space = []
			pred = inputs
			# run the intermediary ladders 
			for i,layer in enumerate(layers[:-1]): # run the last layer, which is the latent layer, outside of the for loop 
				pred = layer(pred)
				if (i in latent_connections_map) or (i-len(layers) in latent_connections_map): # intermediate latent layers 
					if i-len(layers) in latent_connections_map: i = i-len(layers) 
					ladder_num = latent_connections_map[i]
					latent_space.append(list(self.ladders[ladder_num].encoder(pred*self.alpha[ladder_num])))

			# run the last latent layer TBD: change encoder decoder architecture to include final layer as a part of latent space
			latent_space.append(list(self.encoder.convert_layer_to_latent_space(layers[-1](pred*self.alpha[-1]))))
			return latent_space

		self._encoder.call = call
	def _setup_decoder_call(self):
		layers = self.decoder.layers
		latent_connections = {lc[1]:i for i,lc in enumerate(self.latent_connections)}
		def call(latent_space_samples=None, latent_space=None,**kw):
			if latent_space_samples is None: 
				latent_space_samples = [i[0] for i in latent_space]
			pred = latent_space_samples[-1] # get samples from last layer
			for i,layer in enumerate(layers):
				# concatenate latent space
				if (i in latent_connections) or (i-len(layers) in latent_connections):
					if i-len(layers) in latent_connections: idx = i-len(layers)
					ladder_idx = latent_connections[idx]
					ladder_recon = self.alpha[ladder_idx]*self.ladders[ladder_idx].decoder(latent_space_samples[ladder_idx])
					pred = tf.concat((pred, ladder_recon),-1)
				pred = layer(pred)
				if not i:
					pred = self.alpha[-1]*pred
			return pred
		self._decoder.call = call
	
	# configuration
	def get_config(self):
		config_param = {
			**super().get_config(),
			"gamma":str(self.gamma)}
		return config_param

	# run model
	def call(self, inputs, alpha=None, beta=None, **kw):
		# alpha is list of size num latent_space
		#called during each training step and inference
		#TBD: run latent space and next layer in parallel
		self.alpha = alpha
		if beta is None:
			beta = [self.beta]*len(self.alpha)
		self.latent_space = self.encoder(inputs,**kw)
		
		# get reconstruction and regularization loss, also applies shortcut routing to latent_space
		losses = self.regularizer(self.latent_space, alpha=self.alpha, beta=beta, **kw)
		for loss in losses:
			self.add_loss(loss)
		
		reconstruction = self.decoder(None, latent_space=self.latent_space,**kw)
		return reconstruction
	def regularizer(self, latent_space, alpha, beta, **kw):
		"""Regularizer for latent space
		
		Args:
			latent_space (list): latent space output from encoder [num layers, (samples,mean,logvar), batch size, num latents]
			alpha (list): list of ints or lists, if is list, must be length of num latents
			beta (list): list of ints or lists, if is list, must be length of num latents
		
		Returns:
			tf tensor: regularization loss
		"""
		#regularizer creation for each specified layer.
		# use regularizations from parent vae
		losses = []
		self._past_kld = []
		for i,ls in enumerate(latent_space):
			# select the prior to be a specific distribution
			if alpha[i]:
				reg_param = beta[i]
			else:
				reg_param = self.gamma
			reg_loss,kld=self.layer_regularizer(*ls, beta=reg_param, return_kld=True, layer_num=i, **kw)
			losses.append(reg_loss)
			self._past_kld.append(kld)
		return losses

class LVAE(VLAE):
	"""
	lvae+ from BIVA paper
		- forms bidirectional encoder
		- decoder forms lower layer priors
		- changes regularization from normal prior to the decoder bidirectional prior
	"""
	def __init__(self,*ar,name="LVAE",**kw):
		super().__init__(*ar,name=name,**kw)

	def get_config(self):
		config = {k:v for k,v in super().get_config().items() if not k == "gamma"}
		return config
	
	def _setup_encoder(self):
		"""
		- Creates the latent space objects
			- basic check on latent_connections
		- Replaces encoder call to include 
			- all latent space outputs as a list
		"""
		# perform some checks
		layers = self.encoder.layers.layers # the first layers is the sequential layers.
		check_weighted_layers(layers)
		assert not self.latent_connections is None, "must specify latent layers"
		for lc_enc, _ in self.latent_connections:
			assert lc_enc in range(-len(layers), len(layers)), "encoder latent connection specifications is invalid. latent connections %s and num layers is %d"%(lc, len(layers)) 

		# create the latent layers
		self.ladders = [] 
		latent_connections = {lc[0]:i for i,lc in enumerate(self.latent_connections)}
		for i,layer in enumerate(layers):
			if (i in latent_connections) or (i-len(layers) in latent_connections):
				if i-len(layers) in latent_connections: i = i-len(layers)
				# create new latent space
				ladder = BidirectionalLatentSpace(name="LatentSpace_%d"%latent_connections[i])
				ladder.set_encoder(
					layer_param=self.ladder_params[latent_connections[i]][0], 
					num_latents=self.num_latents, 
					shape_input=layer.output_shape[1:], 
					activation=None
					)
				self.ladders.append(ladder)
			#print("TEST:","encoder output",layer.output_shape[1:])

	def _setup_encoder_call(self):
		enc_layers = self.encoder.layers.layers
		dec_layers = self.decoder.layers
		dec_latent_connections = {lc[1]:i for i,lc in enumerate(self.latent_connections)}
		enc_latent_connections_map = {lc[0]:i for i,lc in enumerate(self.latent_connections)}
		
		def call(inputs,use_mean=False,is_return_encoder_outputs=False,**hparams):
			encoder_ls_inputs = []
			pred = inputs
			latent_space=[]
			# Bottom up pass 
			for i,layer in enumerate(enc_layers[:-1]):
				pred = layer(pred)
				if (i in enc_latent_connections_map) or (i-len(enc_layers) in enc_latent_connections_map): # intermediate latent layers 
					if i-len(enc_layers) in enc_latent_connections_map: i = i-len(enc_layers) 
					ladder_num = enc_latent_connections_map[i]
					encoder_ls_inputs.append(pred)
			top_layer = list(self.encoder.convert_layer_to_latent_space(enc_layers[-1](pred)))
			latent_space.append(top_layer)
			
			# apply top down pass
			pred = top_layer[0] #top layer sample
			for i,layer in enumerate(dec_layers):
				# concatenate latent space
				if (i in dec_latent_connections) or (i-len(dec_layers) in dec_latent_connections):
					if i-len(dec_layers) in dec_latent_connections: idx = i-len(dec_layers)
					ladder_idx = dec_latent_connections[idx]
					
					# feed into ls
					ladder_recon, ls = self.top_down_step(pred, self.ladders[ladder_idx], encoder_ls_inputs[ladder_idx], use_mean=use_mean,
							**{k:v[ladder_idx] for k,v in hparams.items()})
					latent_space.append(ls)

					# ls back to decoder
					pred = tf.concat((pred, ladder_recon),-1)
				pred = layer(pred)
			if is_return_encoder_outputs:
				return latent_space[::-1], encoder_ls_inputs#inputs into ls from encoder
			return latent_space[::-1]

		self._encoder.call = call

	def top_down_step(self, inputs, latent_space_obj, encoder_input=None, latent_samples=None, use_mean=False,**hparams):
		"""Applies top down pass for one latent layer
		
		Args:
		    inputs (nparray): input to latent space from decoder
		    latent_space_obj (LatentSpace object): latent space object, will be called on input.
		    encoder_input (numpy array): output from encoder layer into latent layer.
		    latent_samples (list, optional): (sample, mean, logvar) or None, if specified, just reconstruct.
		
		Returns:
		    tuple: returns latent space (sample, mean, logvar) and latent space reconstruction
		"""
		latent_out = None
		if latent_samples is None:
			# project input to latent space input shape 
			# apply latent space on input to get latents
			latent_out = list(latent_space_obj.bidirectional_encode(inputs, encoder_input,**hparams))
			latent_samples = latent_out[0] if not use_mean else latent_out[1]

		# reconstruct
		recon = latent_space_obj.decoder(latent_samples,**hparams)
		return recon, latent_out # return latent space result

	def _setup_decoder_call(self):
		dec_layers = self.decoder.layers
		dec_latent_connections = {lc[1]:i for i,lc in enumerate(self.latent_connections)}
		def call(latent_space_samples=None, latent_space=None, return_ls=False,use_mean=False,encoder_outputs=None,**hparams):
			""" decode.
			 
			latent_space_samples example: [None, None, (samples, mean ,logvar), (samples, mean ,logvar)]

			Args:
			    latent_space_samples (list, None, optional): latent space samples, list of numpy arrays
			    latent_space (list, None, optional): latent space, will retrieve samples from here if not specified
			    **hparams: Description
			
			Returns:
			    TYPE: Description
			"""
			if latent_space_samples is None: 
				latent_space_samples = [None if i is None else i[0+use_mean] for i in latent_space]
			prior_space = [None for _ in latent_space_samples] # will be specified where latent_space_samples is not specified
			# apply top down pass
			pred = latent_space_samples[-1] # get samples from last layer
			for i,layer in enumerate(dec_layers):
				# concatenate latent space
				if (i in dec_latent_connections) or (i-len(dec_layers) in dec_latent_connections):
					if i-len(dec_layers) in dec_latent_connections: idx = i-len(dec_layers)
					ladder_idx = dec_latent_connections[idx]
					
					# feed into ls
					encoder_input = None if encoder_outputs is None else encoder_outputs[ladder_idx]
					ladder_recon, ls = self.top_down_step(
						pred, self.ladders[ladder_idx], latent_samples=latent_space_samples[ladder_idx],use_mean=use_mean,
							encoder_input=encoder_input, **{k:v[ladder_idx] for k,v in hparams.items()})
					if not ls is None: prior_space[ladder_idx] = ls
					# ls back to decoder
					#ladder_recon = self.alpha[ladder_idx]*ladder_recon
					pred = tf.concat((pred, ladder_recon),-1)
				pred = layer(pred)
				#if not i:
				#	pred = self.alpha[-1]*pred
			if return_ls: return pred, prior_space
			return pred

		self._decoder.call = call

	def top_layer_latent_space(self,shape):
		mean = np.zeros(shape,dtype=np.float32)
		logvar = np.zeros(shape,dtype=np.float32)
		samples = np.exp(0.5*logvar)*np.random.normal(size=logvar.shape)+mean
		return samples, mean, logvar
	
	def call(self, inputs, **kw):
		"""runs inference
		"""
		# run encoder
		self.latent_space = self.encoder(inputs,**kw)
		# run decoder (TD with encoder)
		reconstruction = self.decoder(None, latent_space=self.latent_space,**kw)

		prior_space=[None for _ in self.latent_space]
		prior_space[-1] = list(self.top_layer_latent_space(self.latent_space[-1][0].shape))
		
		# run decoder prior (just make all except last latents space samples None)
		prior_space[:-1] = self.decoder(None, latent_space=prior_space, return_ls=True,**kw)[1][:-1]
		assert len(prior_space)== len(self.latent_space)
		# run regularization
		losses = self.regularizer(self.latent_space, prior_space, **kw)
		for loss in losses:
			self.add_loss(loss)

		return reconstruction
	
	def regularizer(self, latent_space, prior_space, **hparams):
		"""
		LVAE+ regularization.
		"""
		losses = []
		self._past_kld = []
		for i,ls,ps in zip(range(len(latent_space)), latent_space, prior_space):
			# select the prior to be a specific distribution
			if not self.gamma is None and hparams["alpha"][i]<1e-7:
				hparams["beta"][i] = self.gamma
			reg_loss,kld=self.layer_regularizer(*ls,*ps[1:], cond_sample=ps[0], return_kld=True, layer_num=i, 
				**{k:v[i] for k,v in hparams.items()})
			losses.append(reg_loss)
			self._past_kld.append(kld)
		return losses

####################
# LVAE Additionals #
####################
class StandardNormalBetaPriorLVAE:
	def __call__(obj, model):
		def layer_regularizer_wrapper(layer_regularizer_method): # this can be pickled
			@wraps(layer_regularizer_method)
			def layer_regularizer(self, sample, mean, logvar, cond_mean=None, cond_logvar=None, beta=None, *, cond_sample=None,
						return_kld=False, layer_num=None,**kw):
				assert not cond_sample is None
				assert not layer_num is None
				if not layer_num >= len(self.latent_connections): 
					kl_loss_standard, kld_standard = layer_regularizer_method(self, sample=cond_sample, mean=cond_mean, logvar=cond_logvar, 
								cond_mean=None, cond_logvar=None, beta=beta, return_kld=True,**kw)
					beta = 1
				kl_loss, kld = layer_regularizer_method(self, sample=sample, mean=mean, logvar=logvar, 
							cond_mean=cond_mean, cond_logvar=cond_logvar, beta=beta, return_kld=True,**kw)
				if not layer_num >= len(self.latent_connections): 
					kl_loss = kl_loss+kl_loss_standard
					kld = kld+kld_standard
				if not return_kld:
					return kl_loss
				return kl_loss, kld
			return layer_regularizer
		model.layer_regularizer = layer_regularizer_wrapper(model.layer_regularizer)
		return model	

class StandardNormalBetaLVAE:
	def __call__(obj, model):
		def layer_regularizer_wrapper(layer_regularizer_method): # this can be pickled
			@wraps(layer_regularizer_method)
			def layer_regularizer(self, sample, mean, logvar, cond_mean=None, cond_logvar=None, beta=None, *, return_kld=False, layer_num=None,**kw):
				assert not layer_num is None
				if not layer_num >= len(self.latent_connections): 
					kl_loss_standard, kld_standard = layer_regularizer_method(self, sample=sample, mean=mean, logvar=logvar, 
								cond_mean=None, cond_logvar=None, beta=beta, return_kld=True,**kw)
					beta = 1
				kl_loss, kld = layer_regularizer_method(self, sample=sample, mean=mean, logvar=logvar, 
							cond_mean=cond_mean, cond_logvar=cond_logvar, beta=beta, return_kld=True,**kw)
				if not layer_num >= len(self.latent_connections): 
					kl_loss = kl_loss+kl_loss_standard
					kld = kld+kld_standard
				if not return_kld:
					return kl_loss
				return kl_loss, kld
			return layer_regularizer
		model.layer_regularizer = layer_regularizer_wrapper(model.layer_regularizer)
		return model	

class LatentMaskLVAE:
	def __init__(self, num_latents,is_mask_sample=True):
		self.num_latents = num_latents
		self.is_mask_sample = is_mask_sample

	def create_mask(self, shape, dtype, num_latents):
		assert num_latents<=shape[-1]
		mask = np.zeros(shape)
		mask[...,:num_latents] = 1 
		mask = tf.cast(tf.convert_to_tensor(mask), dtype)
		return mask

	def apply_encoder_mask(self,encoder,num_latents):
		convert_layer_to_latent_space_old = encoder.convert_layer_to_latent_space
		def convert_layer_to_latent_space(out_layer):
			sample, mean, logvar = convert_layer_to_latent_space_old(out_layer)
			# mask
			mask = self.create_mask(mean.shape, mean.dtype, num_latents)
			mean,logvar = mean*mask, logvar*mask
			if self.is_mask_sample:
				sample = sample*mask
			else: # resample
				sample = tf.exp(0.5*logvar)*tf.random.normal(
					tf.shape(logvar))+mean
			return sample, mean, logvar
		encoder.convert_layer_to_latent_space = convert_layer_to_latent_space

	def setup_latent_layers(self,model):
		assert len(self.num_latents) == len(model.ladders)+1
		for i,ladder in enumerate(model.ladders):
			self.apply_encoder_mask(ladder._encoder,num_latents=self.num_latents[i])
			self.apply_encoder_mask(ladder._prior_encoder,num_latents=self.num_latents[i])
		self.apply_encoder_mask(model._encoder,num_latents=self.num_latents[-1])

	def __call__(obj, model):
		def setup_wrapper(setup_method): # this can be pickled
			@wraps(setup_method)
			def _setup(self,*ar,**kw):
				setup_method(self,*ar,**kw)
				obj.setup_latent_layers(self)
			return _setup
		model._setup = setup_wrapper(model._setup)
		return model

class LatentSubspaceLVAE:
	def __init__(self, subspace_constraints):
		self.subspace_constraints = subspace_constraints

	def apply_encoder_mask(self,encoder,subspace_constraints):
		convert_layer_to_latent_space_old = encoder.convert_layer_to_latent_space
		def convert_layer_to_latent_space(out_layer):
			sample, mean, logvar = convert_layer_to_latent_space_old(out_layer)
			
			elements = np.arange(mean.shape[-1])
			assert len(elements)>=len(subspace_constraints)
			elements[:len(subspace_constraints)] = subspace_constraints
			# elements
			mean,logvar = tf.gather(mean,elements,axis=-1), tf.gather(logvar,elements,axis=-1)
			# resample
			sample = tf.exp(0.5*logvar)*tf.random.normal(tf.shape(logvar))+mean
			return sample, mean, logvar
		encoder.convert_layer_to_latent_space = convert_layer_to_latent_space

	def setup_latent_layers(self,model):
		assert len(self.subspace_constraints) == len(model.ladders)
		for i,ladder in enumerate(model.ladders):
			self.apply_encoder_mask(ladder._prior_encoder,subspace_constraints=self.subspace_constraints[i])

	def __call__(obj, model):
		def setup_wrapper(setup_method): # this can be pickled
			@wraps(setup_method)
			def _setup(self,*ar,**kw):
				setup_method(self,*ar,**kw)
				obj.setup_latent_layers(self)
			return _setup
		model._setup = setup_wrapper(model._setup)
		return model


################
# VLAE Methods #
################

def Sequential(base, additionals):
	for a in additionals:
		a.setup_encoder(base._encoder)
		a.setup_decoder(base._decoder)
		a.setup_regularizer(base)
		a.setup_layer_regularizer(base)
	return base

class AdditionalBase:
	# base model to be inherited. provides additional specifications to the VLAE class.
	# these are just to provide wrappers.
	def __init__(self):
		# parameter setup for the specifications
		pass
	def setup_encoder(self, encoder):
		pass
	def setup_regularizer(self, baseobj):
		pass
	def setup_layer_regularizer(self, baseobj):
		pass
	def setup_decoder(self, decoder):
		pass

class _RoutingBase(AdditionalBase):
	"""
	parent class for routing type additionals
	"""
	def create_regularized_routing(self,latent_space,routing):
		"""Creates a routing for a given layer.
		
		Args:
			latent_space (list): latent space output from encoder [num layers, (samples,mean,logvar), batch size, num latents]
		    routing (dict): {(prior layer num->pln, prior element num->pen):[layer num, [conditioned elements->cond_elements]]}

		Returns:
			list of list of tuple [[(prior mean, prior logvar),...,beta_divisors]], first list is number of latent layers, 2nd dim is for num of regularizations
		"""
		ls_shape = latent_space[0][0].shape
		network_reg = [None for i in latent_space]
		for i,ls in zip(list(range(len(latent_space)))[::-1], latent_space[::-1]):
			layer_reg = []
			beta_divisors=np.zeros(ls_shape[-1])
			for k,v in routing.items():
				if not i == v[0]: continue # only use the routing that is relevent to this layer
				cond_elements=v[1] #conditioned elements
				pln,pen=k#prior latent layer number, and prior element number
				_,m,lv=latent_space[pln]#get the mean and logvar for this latent space
				beta_divisors[cond_elements]+=1
				prior_mean = m[:,pen] if len(v)<=2 else m[:,pen]*v[2] # m*modval +(1-modval)*0
				prior_logvar = lv[:,pen] if len(v)<=2 else lv[:,pen]*v[2] # lv*modval +(1-modval)*0
				mask=np.isin(np.arange(ls_shape[-1]),cond_elements).astype(bool).reshape(1,-1)
				mean=tf.where(
					mask, # masking for conditioned elements
					tf.broadcast_to(tf.expand_dims(prior_mean,axis=-1),ls_shape), # latent prior
					tf.zeros(ls_shape)) # normal gaussian prior
				logvar=tf.where(
					mask, # masking for conditioned elements
					tf.broadcast_to(tf.expand_dims(prior_logvar,axis=-1),ls_shape), # latent prior
					tf.zeros(ls_shape)) # normal gaussian prior
				layer_reg.append((mean,logvar))

			if layer_reg == []:#if layer was not specified in routing
				layer_reg.append((tf.zeros(ls_shape),tf.zeros(ls_shape)))

			beta_divisors=np.where(beta_divisors==0,1,beta_divisors)#0 represents nonaltered gaussian, beta divisor should be one for this
			layer_reg.append(beta_divisors)
			network_reg[i] = layer_reg
		return network_reg

class LatentPriorRouting(_RoutingBase):
	"""
	regular routing capabilities, which switches prior
	can affect: 
		- regularization
			- switches the prior (or provide a combination of original prior and this new one.)
	"""
	def setup_regularizer(self, baseobj):
		breg = baseobj.regularizer
		def regularizer(latent_space, routing={},network_reg_params=None,**kw):
			if network_reg_params is None:
				network_reg_params = self.create_regularized_routing(latent_space, routing)
			return breg(latent_space, network_reg_params=network_reg_params,routing=routing,**kw)
		baseobj.regularizer = regularizer

	def setup_layer_regularizer(self, baseobj):
		blreg = baseobj.layer_regularizer
		def layer_regularizer(*ar,layer_num=None,beta=None,network_reg_params=None, return_kld=False,**kw):
			assert not beta is None
			assert not network_reg_params is None
			assert not layer_num is None
			assert not "cond_mean" in kw, "can't have another additional specify the cond_mean"
			assert not "cond_logvar" in kw, "can't have another additional specify the cond_logvar"
			layer_reg_params=network_reg_params[layer_num]
			#print(layer_num, layer_reg_params)
			reg_loss, kld = 0,0
			beta=np.divide(beta,layer_reg_params[-1]).reshape(1,-1)# reshape to account for batch size
			for m,lv in layer_reg_params[:-1]:
				rloss,kl=blreg(*ar, cond_mean=m, cond_logvar=lv, beta=beta, return_kld=True,**kw)
				reg_loss+=rloss
				kld+=kl
			if return_kld:
				return reg_loss,kld
			return reg_loss
		baseobj.layer_regularizer = layer_regularizer

class ResidualLatentRouting(_RoutingBase):
	"""
	routing, but treats encoder as providing deltas for the latents
	when there is a routing, parent will be added to the child.
	can affect: 
		- regularization
			- applies shortcut for latent for regularization.
		- decoder (optional)
			- applies shortcut for latent for decoding 
	"""
	def __init__(self, apply_on_decoder=False):
		self.dec_apply = apply_on_decoder

	def get_residual_latent_space(self, latent_space, network_reg_params):
		latent_space=[list(i) if type(i) == tuple else i for i in latent_space]
		for i,layer_params in enumerate(network_reg_params):
			# residual 
			for m,lv in layer_params[:-1]:
				latent_space[i][1]+=m
				latent_space[i][2]+=lv
			# resample
			latent_space[i][0] = tf.exp(0.5*latent_space[i][2])*tf.random.normal(tf.shape(latent_space[i][2]))+latent_space[i][1]
		return latent_space

	def setup_regularizer(self, baseobj):
		breg = baseobj.regularizer
		def regularizer(latent_space, routing={},network_reg_params=None,**kw):
			if network_reg_params is None:
				network_reg_params = self.create_regularized_routing(latent_space, routing)
			latent_space = self.get_residual_latent_space(latent_space, network_reg_params)
			return breg(latent_space, network_reg_params=network_reg_params,routing=routing,**kw)
		baseobj.regularizer = regularizer

	def setup_decoder(self, decoder):
		if self.dec_apply:
			dcall = decoder.call
			def call(latent_space, routing={},network_reg_params=None,**kw):
				if network_reg_params is None:
					network_reg_params = self.create_regularized_routing(latent_space, routing)
				latent_space = self.get_residual_latent_space(latent_space, network_reg_params)
				return dcall(None, latent_space=latent_space, network_reg_params=network_reg_params,routing=routing,**kw)
			decoder.call = call

class LatentMask(AdditionalBase):
	"""
	TBD: add inference capabilities

	latent masking, 
	can affect: 
		- decoder
			- applies masking on latent variables for decoding
		- regularization (optional)
			- applies masking on latent variables for regularization
			- this is equivalent to stoping the gradients.
	"""
	def __init__(self, apply_on_regularization=False):
		self.reg_apply = apply_on_regularization

	def setup_regularizer(self, baseobj):
		if self.reg_apply:
			breg = baseobj.regularizer 
			def regularizer(latent_space,latent_mask=None,**kw):
				if not latent_mask is None:
					latent_space = self.apply_latent_mask(latent_space, latent_mask)
				return breg(latent_space,**kw)
			baseobj.regularizer = regularizer

	def setup_decoder(self, decoder):
		dcall = decoder.call
		def call(latent_space, latent_mask=None,**kw):
			if not latent_mask is None:
				latent_space = self.apply_latent_mask(latent_space, latent_mask)
			return dcall(None,latent_space=latent_space,**kw)
		decoder.call = call
	
	def setup_mask(self, mask, mean_shape):
		mask = np.logical_not(mask)
		if len(mask.shape)<2:
			mask = mask.reshape(1,-1)
		mask = tf.broadcast_to(mask.astype(np.float32), mean_shape)
		return mask
	
	def apply_latent_mask(self, latent_space, mask, return_samples_only=False):
		new_ls=[]
		for ls, ma in zip(latent_space, mask):
			s,m,lv = ls
			# where mask is true, make mean and logvar = 0 and then resample for s.
			ma = self.setup_mask(ma, m.shape)
			m=m*ma
			lv=lv*ma
			# TBD: latent resample is hard coded
			s=tf.where(tf.cast(ma,tf.bool),s,tf.exp(0.5*lv)*tf.random.normal(tf.shape(lv))+m) # resample from where was masked with 0
			if return_samples_only:
				new_ls.append(s)
			else:
				new_ls.append((s,m,lv))
		return new_ls

if __name__ == '__main__':
	test_provlae()
