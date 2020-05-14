from utils.tf_custom.architectures.variational_autoencoder import BetaTCVAE
from utils.other_library_tools.disentanglementlib_tools import total_correlation 
from utils.tf_custom.loss import kl_divergence_with_normal, kl_divergence_between_gaussians
import numpy as np
import tensorflow as tf

class CondVAE(BetaTCVAE):
	def __init__(self, beta, name="CondVAE", **kwargs):
		super().__init__(name=name, beta=beta, **kwargs)

	def call(self, inputs, cond_logvar, cond_mean, gamma, latent_to_condition):
		self.gamma = np.clip(gamma, 0, 1).astype(np.float32)
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



def main():
	inputs = np.random.normal(size=(32,64,64,3))
	cond_logvar = np.random.normal(size=(32,10)).astype(np.float32)
	cond_mean = np.random.normal(size=(32,10)).astype(np.float32)


	mask = CondVAE(10)
	#mask(inputs)
	mask(inputs, cond_sampled, cond_logvar, cond_mean)
	print(len(mask.losses))

if __name__ == '__main__':
	main()