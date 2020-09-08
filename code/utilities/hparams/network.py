import numpy as np 
import copy
from . import layers
class SingleLayer:
	"""Manages and combines many layerHPs into one NetworkHP
	one layer only
	"""
	def __init__(self, num_layers, layerhps=None, **kw):
		super().__init__(**kw)

		# indicate none layers
		if not layerhps is None:
			self.none_layerhps=[i is None for i in layerhps]
		else:
			self.none_layerhps=[1 for i in num_layers]
			self.none_layerhps[-1]=0

		# set default layerhps
		if layerhps is None:
			layerhps = [None for i in range(num_layers)]
		assert len(layerhps) == num_layers, "num_layers and layerhps don't match"
		for i in range(num_layers):
			if layerhps[i] is None:
				layerhps[i] = layers.RecentBetaHold()
			layerhps[i].layer_num=i
		self.layerhps=layerhps
		self.num_layers=num_layers

		# additional parameters
		self.past_hp={}

	def __call__(self, step, model, **kw):
		# add layer specific parameters
		hp = self.get_network_params(step,model,**kw)
		self.past_hp.update(hp) #update current network parameters.
		# add network specific parameters
		hp.update(self.get_layer_params(step,model,**kw))
		self.past_hp=hp
		return hp

	def get_network_params(self,step,model,**kw):
		hp = {}
		alpha = self.get_alpha(step,model,**kw)
		alpha = [0 if n else i for i,n in zip(alpha,self.none_layerhps)] # don't activate the unspecified layers
		hp["alpha"]=alpha
		return hp

	def get_alpha(self, step, model,**kw):
		alpha=[0 for _ in range(self.num_layers)]
		alpha[-1]=1
		return alpha

	@property
	def activated_layers(self):
		return self.get_activated_layers()

	def get_activated_layers(self,hp=None):
		if hp is None:
			hp = self.past_hp
		return np.abs(np.asarray(hp["alpha"])-1)<1e-8
	
	def get_layer_params(self, step, model,**kw):
		hp={}
		for is_activated, layerhp in zip(self.activated_layers, self.layerhps):
			if not is_activated:
				layerhp.offset_step = step+1
			lhp=layerhp(step,model,**kw)
			if hp == {}:
				for k in lhp.keys():
					hp[k]=[]
			for k,v in lhp.items():
				hp[k].append(v)
		return hp

class StepTrigger(SingleLayer):
	# multilayer trigger based on step.
	def __init__(self, *ar, alpha_kw={"duration":5000, "start_val":0, "final_val":1, "start_step":5000}, **kw):
		self.alpha_kw=alpha_kw
		super().__init__(*ar, **kw)
		self.alpha_hp={}

	def get_alpha(self,step,model,**kw):
		if not self.past_hp: #initial step
			alpha=[0 for _ in range(self.num_layers)]
			alpha[-1]=self.alpha_kw["final_val"]
		else:
			alpha=self.past_hp["alpha"]
			layer_num = np.sum(np.logical_not(
				np.abs(np.asarray(self.past_hp["alpha"])-self.alpha_kw["final_val"])<1e-8))-1
			layer_num = np.clip(layer_num,0,len(alpha))
			if not layer_num in self.alpha_hp:
				self.alpha_hp[layer_num]=layers.LinearChange(**self.alpha_kw)
				self.alpha_hp[layer_num].start_offset=step-1
			for k,v in self.alpha_hp.items():
				alpha[k]=v(step)
		return alpha

	def get_activated_layers(self,hp=None):
		if hp is None:
			hp = self.past_hp
		return np.abs(np.asarray(hp["alpha"])-self.alpha_kw["final_val"])<1e-8

class StepTriggerReverse(StepTrigger):
	# use lowest expressive layer first.
	def __init__(self, num_layers, layerhps=None, **kw):
		if not layerhps is None:
			layerhps=layerhps[::-1]
		super().__init__(num_layers=num_layers, layerhps=layerhps,**kw)
		if layerhps is None:
			self.layerhps=self.layerhps[::-1]

	def __call__(self, step, model, **kw):
		hp=copy.deepcopy(super().__call__(step,model,**kw))
		hp = {k:v[::-1] for k,v in hp.items()}
		return hp

class KLDTrigger(SingleLayer):
	def get_alpha(self,step,model,**kw):
		# assumes this will only be called once per step.
		if not self.past_hp: #initial step
			alpha=[0 for _ in range(self.num_layers)]
			alpha[-1]=1
		else:
			alpha=self.past_hp["alpha"]
			layer_num = np.sum(np.logical_not(alpha))
			if self.layerhps[layer_num].changed_latent_detection(model.past_kld):
				alpha[layer_num-1]=1
		return alpha

class CondKLDTrigger(KLDTrigger):
	# applies conditioning between layers
	# hierarchy is a tree - children are not shared
	def __init__(self, *ar,num_child=2,**kw):
		self.num_child=num_child
		super().__init__(*ar,**kw)

	def get_network_params(self,step,model,**kw):
		hp=super().get_network_params(step,model,**kw) # this hp contains kl activated alphas
		hp["routing"]=self.get_routing(step,model,**kw)

		#TESTING:
		# if hp["routing"] == {} and np.sum(hp["alpha"])>1:
		# 	print(step, model.past_kld, self.none_layerhps, 
		# 		self.layerhps[-1].changed_latent_detection(model.past_kld),
		# 		self.layerhps[-1].get_kld_breakout(model.past_kld),
		# 		self.none_layerhps,
		# 		)
		# 	exit()

		return hp

	def get_routing(self, step, model, **kw):
		routing = {}
		if self.past_hp:
			routing = self.past_hp["routing"]
			for i,layerhp in enumerate(self.layerhps):
				if layerhp.changed_latent_detection(model.past_kld) and i and not self.none_layerhps[i-1]:
					#routing: {(prior layer num, prior element num):[layer num, [conditioned elements]]}
					prior_element_nums=layerhp.get_kld_breakout(model.past_kld)
					prior_element_nums=np.arange(len(prior_element_nums))[prior_element_nums]
					routing_priors=[(i,pen) for pen in prior_element_nums]
					
					# check existing priors
					existing_priors=list(routing.keys())
					latents_in_use = []
					for k in existing_priors:
						if k[0] == i:
							if not k in routing_priors: # Fix this part
								del routing[k]
							else:
								latents_in_use += routing[k][1]

					# add new routing which are not in already in past routing, using new latent only.
					available_latents = np.arange(len(self.layerhps[i-1].get_kld_breakout(model.past_kld)))
					available_latents = [i for i in available_latents if not i in latents_in_use]
					for k in routing_priors:
						if not k in routing:
							if len(available_latents) < self.num_child:
								print("WARNING: num children not enough for conditioning. Skipping routing for latent", k)
								continue 
							routing[k]=[i-1,available_latents[:self.num_child]]
							available_latents = available_latents[self.num_child:]
		return routing

class CondKLDTriggerLayerMask(CondKLDTrigger):
	def get_network_params(self, step, model, **kw):
		hp=super().get_network_params(step, model, **kw)
		hp["latent_mask"]=self.get_latent_mask(hp,**kw)

	def get_latent_mask(self,hp,**kw):
		# make across each latents, dependent on routing.
		# goes through routing and masks the specified latents
		#routing: {(prior layer num, prior element num):[layer num, [conditioned elements]]}
		for k,v in hp["routing"].items():
			pass

class GradualCondKLDTrigger(CondKLDTrigger):
	def __init__(self, *ar, gp_kw={"duration":5000,"start_val":0,"final_val":0.75,"start_step":2000}, **kw):
		self.gp_kw=gp_kw
		super().__init__(*ar,**kw)
		self.routing_hp={}

	def get_routing(self, step, model, **kw):
		routing = super().get_routing(step, model, **kw)
		new_routing = {}
		for k, v in routing.items():
			if len(v)<3:
				new_routing[k]=v+[0] #0 is placeholder
				self.routing_hp[k] = layers.LinearChange(**self.gp_kw)
				self.routing_hp[k].start_offset = step
			else:
				new_routing[k]=v
			new_routing[k][2] = self.routing_hp[k](step)
		return new_routing
