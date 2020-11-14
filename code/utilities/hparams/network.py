import numpy as np 
import copy
from . import layers
from utilities.causal_tools import ExactTransformationRouting 
from utilities.standard import cprint
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

	def off_step(self,**kw):
		return self.past_kld

	def __call__(self, step=None, model=None, **kw):
		if step is None:
			return self.off_step(**kw)
		# add layer specific parameters
		self.past_hp.update(self.get_network_params(step,model,**kw))#update current network parameters.
		# add network specific parameters
		self.past_hp.update(self.get_layer_params(step,model,**kw))
		return {k:v for k,v in self.past_hp.items() if not k.startswith("_")}

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
	def __init__(self, *ar, alpha_kw={"duration":1, "start_val":0, "final_val":1, "start_step":0}, **kw):
		self.alpha_kw=alpha_kw
		super().__init__(*ar, **kw)
		self.alpha_hp={}

	def get_alpha(self,step,model,**kw):
		if not self.past_hp: #initial step
			alpha=[0 for _ in range(self.num_layers)]
			alpha[-1]=self.alpha_kw["final_val"]
		else:
			alpha=self.past_hp["alpha"]
			# trigger: start running next layer when layer above has reached a final value
			layer_num = np.sum(np.logical_not(
				np.abs(np.asarray(self.past_hp["alpha"])-self.alpha_kw["final_val"])<1e-8))-1
			layer_num = np.clip(layer_num,0,len(alpha))

			if self.alpha_trigger(step,model,layer_num,**kw):
				# add new alpha control if new layer is triggered
				if not layer_num in self.alpha_hp:
					self.alpha_hp[layer_num]=layers.LinearChange(**self.alpha_kw)
					self.alpha_hp[layer_num].start_offset=step-1
			for k,v in self.alpha_hp.items():
				alpha[k]=v(step)
		return alpha

	def alpha_trigger(self,step,model,layer_num,**kw):
		return True

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

class KLDTrigger(StepTrigger):
	def alpha_trigger(self,step,model,layer_num,**kw):
		layer_num=layer_num+1 # detect parent layer
		return self.layerhps[layer_num].changed_latent_detection(model.past_kld)

class CondKLDTrigger(KLDTrigger):
	# applies conditioning between layers
	# hierarchy is a tree - children are not shared
	def __init__(self, *ar,num_child=2,routing_start_step=0,**kw):
		self.num_child=num_child
		super().__init__(*ar,**kw)
		self.routing_start_dict = {}
		self.routing_start_step = routing_start_step

	def routing_start(self, routing, step):
		for k,v in routing.items():
			if not (k in self.routing_start_dict and v == self.routing_start_dict[k][1]):
				self.routing_start_dict[k] = [step,v]
		self.routing_start_dict = {k:v for k,v in self.routing_start_dict.items() if k in routing} 

	def start_routing(self, routing,step=None,routing_start_step=None):
		self.routing_start(routing=routing,step=step)
		if routing_start_step is None:
			routing_start_step = self.routing_start_step
		return copy.deepcopy({k:v[1] for k,v in self.routing_start_dict.items() if step is None or v[0] is None or (v[0]+routing_start_step<=step)})

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
			routing = self.past_hp["_routing"]
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
							#if k in routing_priors: 
							latents_in_use += routing[k][1]
							#else:# Fix this part
							#	del routing[k]

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
		self.past_hp["_routing"] = routing
		return self.start_routing(copy.deepcopy(routing), step=step)

class CondKLDTriggerLayerMask(CondKLDTrigger):
	def __init__(self,*ar,mask_start_step=None,**kw):
		self.mask_transform = ExactTransformationRouting()
		super().__init__(*ar,**kw)
		if mask_start_step is None:
			mask_start_step = self.routing_start_step
		self.mask_start_step = mask_start_step
	def get_layer_params(self, step, model, **kw):
		hp=super().get_layer_params(step, model, **kw)
		hp["latent_mask"]=self.get_latent_mask(step=step,**kw)
		return hp

	def start_mask_routing(self,step=None,mask_start_step=None):
		if mask_start_step is None:
			mask_start_step = self.mask_start_step
		return copy.deepcopy({k:v[1] for k,v in self.routing_start_dict.items() if step is None or v[0] is None or (v[0]+mask_start_step<=step)})

	def get_latent_mask(self,system_num=None, elements=[],step=None,**kw):
		# make across each latents, dependent on routing.
		# goes through routing and masks the specified latents
		# routing: {(prior layer num, prior element num):[layer num, [conditioned elements]]}
		
		# will apply mask where the mask is True.
		mask_start_step = None if not "mask_start_step" in kw else kw["mask_start_step"]
		self.mask_transform.set_routing(self.start_mask_routing(step=step, mask_start_step=mask_start_step))
		possible_systems = sorted(self.mask_transform(elements))
		#cprint.blue(sorted(possible_systems), elements)
		if not possible_systems == []:
			if system_num is None: system_num = np.random.randint(len(possible_systems))
			selected_system = list(possible_systems[system_num])
		else:
			selected_system = []
		
		# default masking is True for lower layers, and unmasked for upper layer.
		mask = [np.ones(layer.num_latents).astype(bool) for layer in self.layerhps]
		mask[-1]*=False
		# mask all the elements in mask transform
		mask_nodes = self.mask_transform.all_nodes
		for l,e in mask_nodes:
			mask[l][e] = True

		# turn on the elements in the selected latents 
		for l,e in selected_system: #latent layer, latent element
			mask[l][e]=False
		return mask
	def off_step(self,latent_element=None,**kw):
		assert not latent_element is None, "latent_element must be specified for CondKLDTriggerLayerMask"
		hp = copy.deepcopy(self.past_hp)
		if latent_element in self.mask_transform.all_nodes:
			elements = [latent_element]
		else:
			elements = []
		hp["latent_mask"] = self.get_latent_mask(-1,elements,step=None,mask_start_step=0)
		return hp
	def get_parent(self,latent_element=None,**kw):
		assert not latent_element is None, "latent_element must be specified for CondKLDTriggerLayerMask"
		parents = self.mask_transform.reversed_routing
		if not latent_element in parents: 
			return None
		else:
			return copy.deepcopy(parents[latent_element])

class GradualCondKLDTrigger(CondKLDTrigger):
	def __init__(self, *ar, gp_kw={"duration":5000,"start_val":0,"final_val":0.75,"start_step":2000}, **kw):
		self.gp_kw=gp_kw
		super().__init__(*ar,**kw)
		self.routing_hp={}

	def get_routing(self, step, model, **kw):
		routing = super().get_routing(step, model, **kw)
		new_routing = {}
		self.routing_hp = {k:v for k,v in self.routing_hp.items() if k in routing}
		for k, v in routing.items():
			if not k in self.routing_hp:
				self.routing_hp[k] = layers.LinearChange(**self.gp_kw)
				self.routing_hp[k].start_offset = step
			if len(v)<3:
				new_routing[k]=v+[0]
			else:
				new_routing[k]=v
			new_routing[k][2] = self.routing_hp[k](step)
		return new_routing
