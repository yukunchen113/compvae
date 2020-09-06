from disentangle.architectures.vae import BetaTCVAE, BetaVAE, VariationalAutoencoder
from disentangle.other_library_tools.disentanglementlib_tools import total_correlation 
from disentangle.loss import kl_divergence_with_normal, kl_divergence_between_gaussians
from utilities.standard import split_latent_into_layer, set_shape
from . import architecture_params as ap
from . import encoders_and_decoders as ead
import numpy as np
import tensorflow as tf
import os
import copy
from functools import reduce

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
		self._encoder = ead.ProVLAEGaussianEncoder(*ar, **kw)

	def set_decoder(self, *ar, **kw):
		assert self._decoder is None, "Decoder is already defined for %s"%(self.name)
		self._decoder = ead.ProVLAEDecoder(*ar, **kw)
	
	def call(self, *ar, **kw):
		assert not (self.decoder is None or self.encoder is None), "Decoder or encoder is not defined"
		return super().call(*ar, **kw)



class ProVLAEBase(BetaVAE):
	def create_default_vae(self, **kwargs):
		# default encoder decoder pair:
		self.create_small_provlae64(**kwargs)

	def create_small_provlae64(self, **kwargs):
		self._encoder = ead.ProVLAEGaussianEncoderSmall64(**kwargs)
		self._decoder = ead.ProVLAEDecoderSmall64(**kwargs)
		self.latent_connections = ap.vlae_latent_connections_small64
		self.ladder_params = ap.vlae_latent_spaces_small64
		self._setup()

	def create_large_provlae64(self, **kwargs):
		self._encoder = ead.ProVLAEGaussianEncoderLarge64(**kwargs)
		self._decoder = ead.ProVLAEDecoderLarge64(**kwargs)
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

class ProVLAE(ProVLAEBase):
	"""
	This is used to configure the architecture based on the connections parameters (parameters that are not base encoder decoder specific)
	
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
		self.gamma = gamma
		super().__init__(name=name, beta=beta, **kwargs)
		self.latent_connections = self.latent_connections if latent_connections is None else latent_connections
		self.ladder_params = self.ladder_params if (not "ladder_params" in kwargs) or (kwargs["ladder_params"] is None) else kwargs["ladder_params"]
		
		self._alpha = None
		self.latent_space = None
		self._past_kld = None
	def _setup(self):
		# redefine encoder and decoder
		self._setup_encoder()
		self._setup_decoder()

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

		# set call
		def call(inputs):
			latent_space = []
			pred = inputs
			# run the intermediary ladders 
			for i,layer in enumerate(layers[:-1]): # run the last layer, which is the latent layer, outside of the for loop 
				pred = layer(pred)
				if (i in latent_connections) or (i-len(layers) in latent_connections): # intermediate latent layers 
					if i-len(layers) in latent_connections: i = i-len(layers) 
					ladder_num = latent_connections[i]
					latent_space.append(self.ladders[ladder_num].encoder(pred*self.alpha[ladder_num]))

			# run the last latent layer TODO: change encoder decoder architecture to include final layer as a part of latent space
			latent_space.append(self.encoder.convert_layer_to_latent_space(layers[-1](pred*self.alpha[-1])))
			return latent_space

		self._encoder.call = call

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

		def call(latent_space_samples):
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

	def get_config(self):
		config_param = {
			**super().get_config(),
			"ladder_param":str(self.ladder_params),
			"latent_connections":str(self.latent_connections),
			"gamma":str(self.gamma)}
		return config_param

	def get_latent_space(self):
		return self.latent_space

	@property
	def alpha(self):
		if self._alpha is None:
			self._alpha = [1]*(len(self.latent_connections)+1) # latent_space size is unknown, so use all possible as worst case
		return self._alpha
	@alpha.setter
	def alpha(self, alpha=None):
		assert len(alpha) == (len(self.latent_connections)+1)
		self._alpha = alpha

	@property
	def past_kld(self):
		return self._past_kld

	def call(self, inputs, alpha=None, beta=None, **kw):
		# alpha is list of size num latent_space
		#called during each training step and inference
		#TBD: run latent space and next layer in parallel
		self.alpha = alpha
		if beta is None:
			beta = [self.beta]*len(self.alpha)

		self.latent_space = self.encoder(inputs)
		reconstruction = self.decoder([i[0] for i in self.latent_space])
		# get reconstruction and regularization loss
		for losses in self.provlae_regularizer(self.latent_space, self.alpha, beta, **kw):
			self.add_loss(losses)

		return reconstruction

	def provlae_regularizer(self, latent_space, alpha, beta, routing={}, **kw):
		"""Regularizer for latent space
		
		Args:
			latent_space (list): latent space output from encoder [num layers, (samples,mean,logvar), batch size, num latents]
			alpha (list): list of ints or lists, if is list, must be length of num latents
			beta (list): list of ints or lists, if is list, must be length of num latents
			routing (dict): {(prior layer num, prior element num):[layer num, [conditioned elements]]}
		
		Returns:
			tf tensor: regularization loss
		"""
		#regularizer creation for each specified layer.
		# use regularizations from parent vae
		losses = []
		self._past_kld = []
		network_reg_params=self.create_regularized_routing(latent_space, routing)
		for i,ls in enumerate(latent_space):
			
			# select the prior to be a specific distribution

			if alpha[i]:
				reg_param = beta[i]
			else:
				reg_param = self.gamma
			
			layer_reg_params=network_reg_params[i]
			reg_loss, kld = None,None
			reg_param=np.divide(reg_param,layer_reg_params[-1]).reshape(1,-1)# reshape to account for batch size
			for params in layer_reg_params[:-1]:
				m,lv=params
				rloss,kl=self.regularizer(*ls, m, lv, beta=reg_param, return_past_kld=True)
				if reg_loss is None:
					reg_loss,kld=rloss,kl
				else:
					reg_loss+=rloss
					kld+=kl

			losses.append(reg_loss)
			self._past_kld.append(kld)
		return losses

	def create_regularized_routing(self,latent_space,routing):
		"""Creates a routing for a given layer.
		
		Args:
			latent_space (list): latent space output from encoder [num layers, (samples,mean,logvar), batch size, num latents]
		    routing (dict): {(prior layer num->pln, prior element num->pen):[layer num, [conditioned elements->cond_elements]]}

		Returns:
			list of list of tuple [[(prior mean, prior logvar),...,beta_divisors]], first list is number of latent layers, 2nd dim is for num of regularizations
		"""
		ls_shape = latent_space[0][0].shape
		network_reg = []
		for i,ls in enumerate(latent_space):
			layer_reg = []
			beta_divisors=np.zeros(ls_shape[-1])
			for k,v in routing.items():
				if not i == v[0]: continue # only use the routing that is relevent to this layer
				cond_elements=v[1] #conditioned elements
				pln,pen=k#prior latent layer number, and prior element number
				_,m,lv=latent_space[pln]#get the mean and logvar for this latent space
				beta_divisors[cond_elements]+=1
				prior_mean = m[:,pen] if len(v)<=2 else m[:,pen]*v[2] # m*gamma +(1-gamma)*0
				prior_logvar = lv[:,pen] if len(v)<=2 else lv[:,pen]*v[2] # lv*gamma +(1-gamma)*0
				mean=tf.where(
					np.isin(np.arange(ls_shape[-1]),cond_elements).astype(bool).reshape(1,-1), # masking for conditioned elements
					tf.broadcast_to(tf.expand_dims(prior_mean,axis=-1),ls_shape), # latent prior
					tf.zeros(ls_shape)) # normal gaussian prior
				logvar=tf.where(
					np.isin(np.arange(ls_shape[-1]),cond_elements).astype(bool).reshape(1,-1), # masking for conditioned elements
					tf.broadcast_to(tf.expand_dims(prior_logvar,axis=-1),ls_shape), # latent prior
					tf.zeros(ls_shape)) # normal gaussian prior
				layer_reg.append((mean,logvar))

			if layer_reg == []:#if layer was not specified in routing
				layer_reg.append((tf.zeros(ls_shape),tf.zeros(ls_shape)))

			beta_divisors=np.where(beta_divisors==0,1,beta_divisors)#0 represents nonaltered gaussian, beta divisor should be one for this
			layer_reg.append(beta_divisors)
			network_reg.append(layer_reg)
		return network_reg


	def regularizer(self, sample, mean, logvar, cond_mean=None, cond_logvar=None, beta=None, *,return_past_kld=False):
		# regularization uses disentanglementlib method
		assert not (cond_mean is None != cond_logvar is None), "mean and logvar must both be sepecified if one is specified"
		if cond_mean is None:
			cond_mean = tf.zeros_like(mean)
			cond_logvar = tf.zeros_like(logvar)
		if beta is None:
			beta = self.beta
		kld = kl_divergence_between_gaussians(mean, logvar, cond_mean, cond_logvar)
		kl_loss = beta*kld
		kl_loss = tf.reduce_sum(kl_loss,1)
		if not return_past_kld:
			return kl_loss
		return kl_loss, kld

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