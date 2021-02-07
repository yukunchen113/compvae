import numpy as np# implements metrics

class InterventionalRobustnessScore:
	def __init__(self, diff_percentile=0.99):
		#based on https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/evaluation/metrics/irs.py
		self.diff_percentile = diff_percentile # used to account for outliers than if were to just use max
	
	def distance_func(self, latents, latent_means):
		return np.abs(latents-latent_means)# disentanglement lib uses L1

	def mpida(self, latents):
		# set latents that have the same I factors
		# assumes that g_j is already the mean (which is ok if getting VAE mean)

		# distance(mean latents, latents)
		pida = self.distance_func(latents,np.mean(latents,axis=0,keepdims=True))
		#mpida = np.amax(pida,axis=0) 
		mpida = np.percentile(pida,q=self.diff_percentile*100,axis=0)
		return mpida

	def empida(self, latent_sets):
		# latent_sets are a list of numpy array, each array row contains same I factors with different J factors, [batch size, num desired latents]
		# I,J sets are consistent acorss latent sets (eg. I is always the 1st,2nd and 4th latents across all sets)
		empida = []
		for arr in latent_sets:
			empida.append(self.mpida(arr))
		empida = np.mean(empida, axis=0)
		return empida

	def __call__(self, latent_sets, max_dev):
		# latent_sets are a list of numpy array, each array row contains same I factors with different J factors, [batch size, num desired latents]
		# I,J sets are consistent acorss latent sets (eg. I is always the 1st,2nd and 4th latents across all sets)
		empida = self.empida(latent_sets)
		irs = 1-empida/max_dev
		return irs



