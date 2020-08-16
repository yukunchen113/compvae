import numpy as np

class HPSchedule:

	"""
	Holds at most recent beta

	Example:
		>>> hps=HPSchedule(3, 
		>>> 		beta_anneal_duration=30000, start_beta=175, final_beta=8, wait_steps=10000)
		>>> class TestModel:
		>>> 	def __init__(self,val=False):
		>>> 		if not val:
		>>> 			self.past_kld=np.ones((3,100,8))/2
		>>> 		else:
		>>> 			self.past_kld=np.ones((3,100,8))*2
		>>> print(hps(0,TestModel()))
		>>> print(hps(100,TestModel()))
		>>> print(hps.past_hp)
		>>> print(hps(100,TestModel(val=True)))

	"""

	def __init__(self, num_latent_connections, beta_anneal_duration, start_beta, final_beta, 
		wait_steps=10000, start_step=0, kld_detection_threshold=1, layer_num=-1):
		self.beta_anneal_duration=beta_anneal_duration
		self.start_beta=start_beta
		self.final_beta=final_beta
		self.start_step=start_step
		self.num_latent_connections=num_latent_connections
		self.wait_steps=wait_steps
		self.layer_num=layer_num

		self.past_hp=None
		self.state=0
		self.kld_breakout=None
		self.kld_detection_threshold=kld_detection_threshold
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
		alpha = [0]*self.num_latent_connections
		alpha[self.layer_num] = 1
		beta = [0]*self.num_latent_connections
		beta[self.layer_num] = (self.start_beta-self.final_beta)*(
				1-np.clip((step-self.start_step)/self.beta_anneal_duration, 0, 1))+self.final_beta
		hp = dict(alpha=alpha,beta=beta)
		return hp

	def state1(self,step):
		return self.past_hp

	def check_state(self,step,model):
		kld=model.past_kld
		if kld is None: return
		kld = np.mean(np.abs(list(kld)[self.layer_num]),axis=0)
		kld_breakout = kld>self.kld_detection_threshold
		if self.kld_breakout is None:
			self.kld_breakout=np.zeros_like(kld_breakout).astype(bool)
		if not np.all(kld_breakout == self.kld_breakout):
			self.start_step=step+self.wait_steps
			self.kld_breakout = kld_breakout
		if step<self.start_step:
			self.state=1
		else:
			self.state=0

class HPSchedule2(HPSchedule):
	#holds at start beta
	def state1(self,step):
		hp=self.past_hp
		beta = [0]*self.num_latent_connections
		beta[-1]=self.start_beta
		hp["beta"]=beta
		return hp

class HPSchedule3(HPSchedule):
	#holds at specified beta
	def __init__(self, *ar, converge_beta=None,**kw):
		assert not converge_beta is None
		self.converge_beta=converge_beta
		super().__init__(*ar,**kw)
	def state1(self,step):
		hp=self.past_hp
		beta = [0]*self.num_latent_connections
		beta[-1]=self.converge_beta
		hp["beta"]=beta
		return hp


def test_HPSchedule(HPSchedule=HPSchedule):
	hps=HPSchedule(3, beta_anneal_duration=30000, start_beta=175, final_beta=8, wait_steps=10000)
	class TestModel:
		def __init__(self,val=False):
			if not val:
				self.past_kld=np.ones((3,100,8))/2
			else:
				self.past_kld=np.ones((3,100,8))/2
				self.past_kld[:,:,1]=10
	print(hps(0,TestModel()))
	print(hps(100,TestModel()))
	print(hps.past_hp)
	print(hps(500,TestModel(val=True)))
	print(hps(1000,TestModel()))
	print(hps(15000,TestModel()))

def main():
	test_HPSchedule(HPSchedule)

if __name__ == '__main__':
	main()



'''
def hparam_schedule_alpha3(step, num_latent_connections, start_step=15000, alpha_duration=10000, beta_duration=10000, start_beta = 10, final_beta = 1):
	# only one layer

	# sets alpha
	alpha = [0]*num_latent_connections
	alpha[-1] = 1

	return dict(alpha=alpha)


def hparam_schedule_alpha_beta(step, num_latent_connections, start_step=15000, alpha_duration=10000, beta_duration=10000, start_beta = 10, final_beta = 1):
	# start_step is where the a new latent space starts getting integrated
	# alpha_duration is how long a new latent space takes for architecture to get integrated
	# beta_duration is how long it takes for a beta value to drop to a certain value

	# changes alpha
	alpha = [0]*num_latent_connections
	alpha[-1] = 1
	for i in range(1,len(alpha)):
		alpha[len(alpha)-i-1] = np.clip((step-start_step*i)/alpha_duration, 0, 1) # after the first alpha_duration steps, evolve alpha for a steps
	
	# changes beta
	beta = [0]*num_latent_connections
	for i in range(len(beta)):
		beta[len(beta)-i-1] = (start_beta-final_beta)*(1-np.clip((step-start_step*i)/beta_duration, 0, 1))+final_beta

	return dict(alpha=alpha, beta=beta)

def hparam_schedule_alpha_beta2(step, num_latent_connections, start_step=15000, alpha_duration=10000, beta_duration=10000, start_beta = 10, final_beta = 1):
	# same beta until model integration is complete
	# start_step is where the a new latent space starts getting integrated
	# alpha_duration is how long a new latent space takes for architecture to get integrated
	# beta_duration is how long it takes for a beta value to drop to a certain value

	# changes alpha
	alpha = [0]*num_latent_connections
	alpha[-1] = 1
	for i in range(1,len(alpha)):
		alpha[len(alpha)-i-1] = np.clip((step-start_step*i)/alpha_duration, 0, 1) # after the first alpha_duration steps, evolve alpha for a steps
	
	# changes beta
	beta = [0]*num_latent_connections
	for i in range(len(beta)):
		beta[len(beta)-i-1] = (start_beta-final_beta)*(1-np.clip((step-(start_step*(len(alpha)-1)+alpha_duration))/beta_duration, 0, 1))+final_beta

	return dict(alpha=alpha, beta=beta)

def hparam_schedule_alpha_beta3(step, num_latent_connections, start_step=0, beta_duration=10000, start_beta = 10, final_beta = 1):
	# same beta until model integration is complete
	# start_step is where the a new latent space starts getting integrated
	# alpha_duration is how long a new latent space takes for architecture to get integrated
	# beta_duration is how long it takes for a beta value to drop to a certain value

	# changes alpha
	alpha = [0]*num_latent_connections
	alpha[-1] = 1
	#for i in range(1,len(alpha)):
	#	alpha[len(alpha)-i-1] = np.clip((step-start_step*i)/alpha_duration, 0, 1) # after the first alpha_duration steps, evolve alpha for a steps
	
	# changes beta
	beta = [0]*num_latent_connections
	beta[-1] = (start_beta-final_beta)*(1-np.clip((step-start_step)/beta_duration, 0, 1))+final_beta
	#for i in range(len(beta)):
	#	beta[len(beta)-i-1] = (start_beta-final_beta)*(1-np.clip((step-start_step*(i+1))/beta_duration, 0, 1))+final_beta

	return dict(alpha=alpha, beta=beta)


def hparam_schedule_alpha_beta3_quantized(step, num_latent_connections, start_step=0, beta_duration=10000, start_beta = 10, final_beta = 1, n_multiplier=1000):
	# same beta until model integration is complete
	# start_step is where the a new latent space starts getting integrated
	# alpha_duration is how long a new latent space takes for architecture to get integrated
	# beta_duration is how long it takes for a beta value to drop to a certain value

	# changes alpha
	alpha = [0]*num_latent_connections
	alpha[-1] = 1
	#for i in range(1,len(alpha)):
	#	alpha[len(alpha)-i-1] = np.clip((step-start_step*i)/alpha_duration, 0, 1) # after the first alpha_duration steps, evolve alpha for a steps
	
	# changes beta
	beta = [0]*num_latent_connections
	beta[-1] = int(((start_beta-final_beta)*(1-np.clip((step-start_step)/beta_duration, 0, 1)))*n_multiplier)/n_multiplier +final_beta
	#for i in range(len(beta)):
	#	beta[len(beta)-i-1] = (start_beta-final_beta)*(1-np.clip((step-start_step*(i+1))/beta_duration, 0, 1))+final_beta

	return dict(alpha=alpha, beta=beta)

'''