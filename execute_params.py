import numpy as np

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