import numpy as np 

class BaseHP():

	"""
	Base hyperparameter class, containing various useful methods for but are not specific to NetworkHP and LayerHP
	"""
	def __init__(self,kld_detection_threshold=1,**kw):
		self.kld_detection_threshold = kld_detection_threshold
		self.kld_breakout = None
	def changed_latent_detection(self, kld, layer_num):
		if kld is None: return None
		kld=list(kld)[layer_num]
		kld = np.mean(np.abs(kld),axis=0)
		kld_breakout = kld>self.kld_detection_threshold
		if self.kld_breakout is None:
			self.kld_breakout=np.zeros_like(kld_breakout).astype(bool)
		if not np.all(kld_breakout == self.kld_breakout):
			return True
		return False

class NetworkHP(BaseHP):
	"""Manages and combines many layerHPs into one NetworkHP
	one layer only
	"""
	def __init__(self, num_layers, layerhps=None, **kw):
		super().__init__(**kw)
		# handle layerhps
		if layerhps is None:
			layerhps = [None for i in range(num_layers)]
		assert len(layerhps) == num_layers, "num_layers and layerhps don't match"
		for i in range(num_layers):
			if layerhps[i] is None:
				layerhps[i] = LayerHP()
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
		alpha=[0 for _ in range(self.num_layers)]
		alpha[-1]=1
		hp["alpha"]=alpha
		return hp
	@property
	def activated_layers(self):
		return np.asarray(self.past_hp["alpha"]).astype(bool)
	
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

class NetworkHP2(NetworkHP):
	# multiple layer, alpha switches on.
	def __init__(self, num_layers, layerhps=None, **kw):
		if not layerhps is None:
			self.none_layerhps=[i is None for i in layerhps]
		else:
			self.none_layerhps=[0 for i in num_layers]
			self.none_layerhps[-1]=1
		super().__init__(num_layers=num_layers, layerhps=layerhps, **kw)
	
	def get_network_params(self,step,model,**kw):
		hp = {}
		alpha = self.get_alpha(step,model,**kw)
		alpha = [0 if n else i for i,n in zip(alpha,self.none_layerhps)]
		hp["alpha"]=alpha
		return hp

	def get_alpha(self,step,model,**kw):
		# assumes this will only be called once per step.
		if not self.past_hp: #initial step
			alpha=[0 for _ in range(self.num_layers)]
			alpha[-1]=1
		else:
			alpha=self.past_hp["alpha"]
			layer_num = np.sum(np.logical_not(alpha))
			if self.changed_latent_detection(model.past_kld, layer_num):
				alpha[layer_num-1]=1
		return alpha

class LayerHP(BaseHP):
	"""Manages state of each layer (hold, run)
	"""
	@property
	def layer_num(self):
		if self._layer_num is None:
			return 0
		else:
			return self._layer_num
	@layer_num.setter
	def layer_num(self, layer_num):
		assert self._layer_num == layer_num or self._layer_num is None, "layer num already specified and does not match newly specified layer_num"
		self._layer_num = layer_num

	def __init__(self, beta_anneal_duration=5000, start_beta=80, final_beta=8, 
		wait_steps=10000, start_step=0,**kw):
		super().__init__(**kw)
		self.beta_anneal_duration=beta_anneal_duration
		self.start_beta=start_beta
		self.final_beta=final_beta
		self._start_step=start_step #initially set, and internaly controlled
		self.wait_steps=wait_steps

		self.hold_step=start_step
		self.offset_step=0#externally controlled variable
		self._layer_num=None
		self.state=0
		self.kld_breakout=None
		self.past_hp=None

	@property
	def start_step(self):
		return self._start_step+self.offset_step
	

	def __call__(self, step, model, **kw):
		#finite state machine
		self.check_state(step,model)
		if self.past_hp is None or self.state==0:#transition
			hp = self.state0(step)
			self.past_hp = hp
		elif self.state==1: #hold last hparam
			hp = self.state1(step)
			self.past_hp = hp
		return hp

	def state0(self,step):
		beta = (self.start_beta-self.final_beta)*(
				1-np.clip((step-self.start_step)/self.beta_anneal_duration, 0, 1))+self.final_beta
		hp = dict(beta=beta)
		return hp

	def state1(self,step):
		return self.past_hp

	def check_state(self,step,model):
		if self.changed_latent_detection(model.past_kld, self.layer_num):
			if step>=self.start_step:
				self._start_step=self._start_step+self.wait_steps-np.maximum(self.hold_step-step,0) #this last part will account for if hold step is not reached.
			self.hold_step=step+self.wait_steps
		self.hold_step=max(self.hold_step,self.start_step)
		if step<self.hold_step:
			self.state=1
		else:
			self.state=0

class LayerHP2(LayerHP):
	#holds at start beta
	def state1(self,step):
		hp=self.past_hp
		beta=self.start_beta
		hp["beta"]=beta
		return hp

class LayerHP3(LayerHP):
	#holds at specified beta
	def __init__(self, *ar, converge_beta=None,**kw):
		assert not converge_beta is None
		self.converge_beta=converge_beta
		super().__init__(*ar,**kw)
	def state1(self,step):
		hp=self.past_hp
		beta=self.converge_beta
		hp["beta"]=beta
		return hp