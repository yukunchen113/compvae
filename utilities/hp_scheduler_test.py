import hp_scheduler as hs
import matplotlib.pyplot as plt
import numpy as np
hps=hs.NetworkHP2(
			num_layers=3, layerhps = [
			None,
			hs.LayerHP3(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=5000, start_step=5000, converge_beta=300),
			hs.LayerHP3(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=5000, start_step=5000, converge_beta=300),
			])

class TestModel:
	def __init__(self,layers=[True,True,True]):
		self.past_kld=np.ones((len(layers),100,8))*0.01
		self.past_kld[layers]=2



change_steps = {6000:[False,False,True], 12000:[False,True,True], 18000:[True,True,True]}

beta = []
for step in range(30000):
	print(str(step)+"\r", end="")
	val=[False,False,False]
	if step in change_steps:# since it is a quick switch the final addition hold would be change_steps+1 
		val=change_steps[step]
	model=TestModel(val)
	beta.append(hps(step,model)["beta"])
beta=np.asarray(beta).transpose()
for i,b in enumerate(beta):
	plt.plot(b, label="layer "+str(i))
plt.legend()
plt.show()