import numpy as np

class LinearChange:
	# note, this is not a layer!
	def __init__(self, duration, start_val, final_val, start_step=0):
		self.duration=duration
		self.start_val=start_val
		self.final_val=final_val
		self._start_step=start_step

		self.start_offset=0
	@property
	def start_step(self):
		return self._start_step+self.start_offset
	
	def __call__(self, step, **kw):
		return (self.start_val-self.final_val)*(
			1-np.clip((step-self.start_step)/self.duration, 0, 1))+self.final_val

class NoOp:
	def __init__(self,**kw):
		self.kw = kw
	def __call__(self,*ar,**kw):
		return self.kw

class LinearBeta(LinearChange):
	def __call__(self, step, model=None,**kw):
		beta = super().__call__(step=step)
		return {"beta":beta}

class BaseBetaKLDDetection:
	def __init__(self,kld_detection_threshold=1,**kw):
		self.kld_detection_threshold = kld_detection_threshold
		self.kld_breakout = None
		self._layer_num=None
		self.kld=None
	@property	
	def num_latents(self):
		if self.kld_breakout is None:
			return None
		return len(self.kld_breakout)

	def get_kld_breakout(self, kld):
		if kld is None: return None
		kld=list(kld)[self.layer_num]
		self.kld = np.mean(np.abs(kld),axis=0)
		kld_breakout = self.kld>self.kld_detection_threshold
		return kld_breakout

	def changed_latent_detection(self, kld, is_set=False):
		kld_breakout=self.get_kld_breakout(kld=kld)
		if kld_breakout is None: return None
		if self.kld_breakout is None:
			self.kld_breakout=np.zeros_like(kld_breakout).astype(bool)
		if not np.all(kld_breakout == self.kld_breakout):
			if is_set:
				self.kld_breakout=kld_breakout.astype(bool)
			return True
		return False

	@property
	def layer_num(self):
		assert not self._layer_num is None
		return self._layer_num
			
	@layer_num.setter
	def layer_num(self, layer_num):
		assert self._layer_num == layer_num or self._layer_num is None, "layer num already specified and does not match newly specified layer_num"
		self._layer_num = layer_num

class RecentBetaHold(BaseBetaKLDDetection):
	"""Manages state of each layer (hold, run)
	"""
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
		self.state=0
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
		if self.changed_latent_detection(model.past_kld,is_set=True):
			if step>=self.start_step:
				self._start_step=self._start_step+self.wait_steps-np.maximum(self.hold_step-step,0) #this last part will account for if hold step is not reached.
			self.hold_step=step+self.wait_steps
		self.hold_step=max(self.hold_step,self.start_step)
		if step<self.hold_step:
			self.state=1
		else:
			self.state=0

class StartBetaHold(RecentBetaHold):
	#holds at start beta
	def state1(self,step):
		hp=self.past_hp
		beta=self.start_beta
		hp["beta"]=beta
		return hp

class SpecifiedBetaHold(RecentBetaHold):
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

		