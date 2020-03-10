from utils.tf_custom.architectures.variational_autoencoder import VariationalAutoencoder
from utils.other_library_tools.disentanglementlib_tools import gaussian_log_density, total_correlation 
from utils.tf_custom.loss import kl_divergence_with_normal

class CompVAE(VariationalAutoencoder):
	def __init__(self, beta, name="BetaTCVAE", **kwargs):
		super().__init__(name=name, **kwargs)
		self.create_encoder_decoder_512() # use the larger model
		self.beta = beta

	def call(self, inputs, m_sampled, m_logvar, m_mean):
		sample, mean, logvar = self.encoder(inputs)
		reconstruction = self.decoder(sample)
		self.add_loss(self.regularizer(sample, mean, logvar, 
				m_sampled, m_logvar, m_mean))
		return reconstruction

	def regularizer(self, sample, mean, logvar, m_sampled, m_logvar, m_mean):
		# regularization uses disentanglementlib method
		kl_loss = kl_divergence_with_normal(mean, logvar)
		tc = (self.beta - 1) * total_correlation(sample, mean, logvar)
		return tc + kl_loss