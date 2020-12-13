from . import network 
from . import layers
import matplotlib.pyplot as plt
import numpy as np
from utilities.standard import cprint, Compare
class TestModel:
	def __init__(self,num_layers=3):
		self.num_layers=8
		self.past_kld=np.ones((num_layers,100,self.num_layers))*0.01

	def set_layer(self,layer_num,elements,value=2):
		self.past_kld[layer_num,:,elements]=value

def check_layer(check_dict, layer, num_steps=300, model=None):
	if model is None:
		model = TestModel()
	for step in range(num_steps):
		if step in check_dict:
			mp, ex = check_dict[step]
			if not mp is None:
				model.set_layer(**mp)
			out = layer(step,model)
			for key,v in out.items():
				assert abs(v - ex[key])<1e-8, step
		else:
			out = layer(step,model)

def test_layers_BaseBetaKLDDetection():
	model = TestModel()
	b = layers.BaseBetaKLDDetection()
	b.layer_num=1
	assert not b.changed_latent_detection(model.past_kld,is_set=True)
	assert not b.changed_latent_detection(model.past_kld,is_set=True)
	assert np.all(b.kld_breakout == False)
	model.set_layer(1,1)
	assert b.changed_latent_detection(model.past_kld,is_set=False)
	assert b.changed_latent_detection(model.past_kld,is_set=True)
	assert not np.all(b.kld_breakout == False)
	assert not b.changed_latent_detection(model.past_kld,is_set=True)

def test_layers_RecentBetaHold():
	b = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	b.layer_num=2
	# interference during start step
	# {step:[model set, expected anser]}
	check_dict = {
		0:[None,{"beta":100}],
		5:[None, {"beta":100}],
		10:[None,{"beta":100}],
		20:[None,{"beta":90}],
		29:[None,{"beta":81}],
		30:[dict(layer_num=2,elements=2),{"beta":81}],
		35:[None,{"beta":81}],
		40:[None,{"beta":80}],
		60:[None,{"beta":60}],
		69:[None,{"beta":51}],
		70:[dict(layer_num=2,elements=3),{"beta":51}],
		75:[None,{"beta":51}],
		80:[None,{"beta":50}],
		100:[None,{"beta":30}],
		130:[None,{"beta":0}],
		200:[None,{"beta":0}],
		}
	check_layer(check_dict, layer=b)

def test_layers_StartBetaHold():
	b = layers.StartBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	b.layer_num=2
	# interference during start step
	# {step:[model set, expected anser]}
	check_dict = {
		0:[None,{"beta":100}],
		5:[None, {"beta":100}],
		10:[None,{"beta":100}],
		20:[None,{"beta":90}],
		29:[None,{"beta":81}],
		30:[dict(layer_num=2,elements=2),{"beta":100}],
		35:[None,{"beta":100}],
		40:[None,{"beta":80}],
		60:[None,{"beta":60}],
		69:[None,{"beta":51}],
		70:[dict(layer_num=2,elements=3),{"beta":100}],
		75:[None,{"beta":100}],
		80:[None,{"beta":50}],
		100:[None,{"beta":30}],
		130:[None,{"beta":0}],
		200:[None,{"beta":0}],
		}
	check_layer(check_dict, layer=b)

def test_layers_SpecifiedBetaHold():
	b = layers.SpecifiedBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10, converge_beta=200)
	b.layer_num=2
	# interference during start step
	# {step:[model set, expected anser]}
	check_dict = {
		0:[None,{"beta":100}],
		1:[None,{"beta":200}],
		5:[None, {"beta":200}],
		10:[None,{"beta":100}],
		20:[None,{"beta":90}],
		29:[None,{"beta":81}],
		30:[dict(layer_num=2,elements=2),{"beta":200}],
		35:[None,{"beta":200}],
		40:[None,{"beta":80}],
		60:[None,{"beta":60}],
		69:[None,{"beta":51}],
		70:[dict(layer_num=2,elements=3),{"beta":200}],
		75:[None,{"beta":200}],
		80:[None,{"beta":50}],
		100:[None,{"beta":30}],
		130:[None,{"beta":0}],
		200:[None,{"beta":0}],
		}
	check_layer(check_dict, layer=b)

def check_network(check_dict, network, num_steps=300, model=None, *, is_assert=True):
	if model is None:
		model = TestModel()
	failed = False
	for step in range(num_steps):
		if step in check_dict:
			mp, ex = check_dict[step]
			if not mp is None:
				model.set_layer(**mp)
			out = network(step,model)

			if not Compare.dictionary(out,ex):
				print(str(step)+"|\n"+str(out)+"|\n"+str(ex))
				failed = True
		else:
			out=network(step,model)
	if is_assert:
		assert not failed

def test_network_SingleLayer():
	b = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	a = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	c = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	b = network.SingleLayer(3, layerhps=[c,a,b])
	b.layer_num=2
	# interference during start step
	# {step:[model set, expected anser]}
	check_dict = {
		0:[None,{"alpha":[0,0,1],"beta":[100,100,100]}],
		5:[None, {"alpha":[0,0,1],"beta":[100,100,100]}],
		10:[None,{"alpha":[0,0,1],"beta":[100,100,100]}],
		20:[None,{"alpha":[0,0,1],"beta":[100,100,90]}],
		29:[None,{"alpha":[0,0,1],"beta":[100,100,81]}],
		30:[dict(layer_num=2,elements=2),{"alpha":[0,0,1],"beta":[100,100,81]}],
		35:[None,{"alpha":[0,0,1],"beta":[100,100,81]}],
		40:[None,{"alpha":[0,0,1],"beta":[100,100,80]}],
		60:[None,{"alpha":[0,0,1],"beta":[100,100,60]}],
		69:[None,{"alpha":[0,0,1],"beta":[100,100,51]}],
		70:[dict(layer_num=2,elements=3),{"alpha":[0,0,1],"beta":[100,100,51]}],
		75:[None,{"alpha":[0,0,1],"beta":[100,100,51]}],
		80:[None,{"alpha":[0,0,1],"beta":[100,100,50]}],
		100:[None,{"alpha":[0,0,1],"beta":[100,100,30]}],
		130:[None,{"alpha":[0,0,1],"beta":[100,100,0]}],
		200:[None,{"alpha":[0,0,1],"beta":[100,100,0]}],
		}
	check_network(check_dict, network=b)

def test_network_StepTrigger():
	b = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	a = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	c = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	b = network.StepTrigger(3, layerhps=[c,a,b], alpha_kw={
			"duration":50,"start_val":0,"final_val":5,"start_step":10})
	b.layer_num=2
	# interference during start step
	# {step:[model set, expected anser]}
	check_dict = {
		0:[None,{"alpha":[0,0,5],"beta":[100,100,100]}],
		5:[None, {"alpha":[0,0,5],"beta":[100,100,100]}],
		10:[None,{"alpha":[0,0,5],"beta":[100,100,100]}],
		20:[None,{"alpha":[0,1,5],"beta":[100,100,90]}],
		29:[None,{"alpha":[0,1.9,5],"beta":[100,100,81]}],
		30:[dict(layer_num=2,elements=2),{"alpha":[0,2,5],"beta":[100,100,81]}],
		35:[None,{"alpha":[0,2.5,5],"beta":[100,100,81]}],
		40:[None,{"alpha":[0,3,5],"beta":[100,100,80]}],
		60:[None,{"alpha":[0,5,5],"beta":[100,100,60]}],
		69:[None,{"alpha":[0,5,5],"beta":[100,100,51]}],
		70:[dict(layer_num=[1,2],elements=3),{"alpha":[0,5,5],"beta":[100,100,51]}],
		75:[None,{"alpha":[0.5,5,5],"beta":[100,100,51]}],
		80:[None,{"alpha":[1,5,5],"beta":[100,100,50]}],
		100:[None,{"alpha":[3,5,5],"beta":[100,80,30]}],
		130:[None,{"alpha":[5,5,5],"beta":[100,50,0]}],
		200:[None,{"alpha":[5,5,5],"beta":[30,0,0]}],
		}
	check_network(check_dict, network=b)

def test_network_StepTriggerReverse():
	b = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	a = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	c = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	b = network.StepTriggerReverse(3, layerhps=[c,a,None], alpha_kw={
			"duration":50,"start_val":0,"final_val":5,"start_step":10})
	b.layer_num=2
	# interference during start step
	# {step:[model set, expected anser]}
	check_dict = {
		0:[None,{"alpha":[0,0,5][::-1],"beta":[100,100,80]}],
		5:[None, {"alpha":[0,0,5][::-1],"beta":[100,100,80]}],
		10:[None,{"alpha":[0,0,5][::-1],"beta":[100,100,80]}],
		20:[None,{"alpha":[0,1,5][::-1],"beta":[90,100,80]}],
		29:[None,{"alpha":[0,1.9,5][::-1],"beta":[81,100,80]}],
		30:[dict(layer_num=2,elements=2),{"alpha":[0,2,5][::-1],"beta":[81,100,80]}],
		35:[None,{"alpha":[0,2.5,5][::-1],"beta":[81,100,80]}],
		40:[None,{"alpha":[0,3,5][::-1],"beta":[80,100,80]}],
		60:[None,{"alpha":[0,5,5][::-1],"beta":[60,100,80]}],
		69:[None,{"alpha":[0,5,5][::-1],"beta":[51,100,80]}],
		70:[dict(layer_num=[1,2],elements=3),{"alpha":[0,5,5][::-1],"beta":[51,100,80]}],
		75:[None,{"alpha":[0,5,5][::-1],"beta":[51,100,80]}],
		80:[None,{"alpha":[0,5,5][::-1],"beta":[50,100,80]}],
		100:[None,{"alpha":[0,5,5][::-1],"beta":[30,80,80]}],
		130:[None,{"alpha":[0,5,5][::-1],"beta":[0,50,80]}],
		200:[None,{"alpha":[0,5,5][::-1],"beta":[0,0,80]}],
		}
	check_network(check_dict, network=b)

def test_network_KLDTrigger():
	b = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	a = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	c = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	b = network.KLDTrigger(3,alpha_kw={"duration":1, "start_val":0, "final_val":1, "start_step":0}, layerhps=[c,a,b])
	b.layer_num=2
	# interference during start step
	# {step:[model set, expected anser]}
	check_dict = {
		0:[None,{"alpha":[0,0,1],"beta":[100,100,100]}],
		5:[None, {"alpha":[0,0,1],"beta":[100,100,100]}],
		10:[None,{"alpha":[0,0,1],"beta":[100,100,100]}],
		20:[None,{"alpha":[0,0,1],"beta":[100,100,90]}],
		29:[None,{"alpha":[0,0,1],"beta":[100,100,81]}],
		30:[dict(layer_num=2,elements=2),{"alpha":[0,1,1],"beta":[100,100,81]}],
		35:[None,{"alpha":[0,1,1],"beta":[100,100,81]}],
		40:[None,{"alpha":[0,1,1],"beta":[100,100,80]}],
		60:[None,{"alpha":[0,1,1],"beta":[100,80,60]}],
		69:[None,{"alpha":[0,1,1],"beta":[100,71,51]}],
		70:[dict(layer_num=[1,2],elements=3),{"alpha":[1,1,1],"beta":[100,71,51]}],
		75:[None,{"alpha":[1,1,1],"beta":[100,71,51]}],
		80:[None,{"alpha":[1,1,1],"beta":[100,70,50]}],
		100:[None,{"alpha":[1,1,1],"beta":[80,50,30]}],
		130:[None,{"alpha":[1,1,1],"beta":[50,20,0]}],
		200:[None,{"alpha":[1,1,1],"beta":[0,0,0]}],
		}
	check_network(check_dict, network=b)

def test_network_GradualCondKLDTrigger():
	b = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	a = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	c = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	n = network.GradualCondKLDTrigger(3, layerhps=[c,a,b], num_child=2,
		routing_start_step=0, 
		alpha_kw={"duration":1, "start_val":0, "final_val":1, "start_step":0}, 
		gp_kw={"duration":50,"start_val":0,"final_val":5,"start_step":10})

	# interference during start step
	# {step:[model set, expected anser]}
	check_dict = {
		0:[None,{"alpha":[0,0,1],"beta":[100,100,100],"routing":{}}],
		5:[None, {"alpha":[0,0,1],"beta":[100,100,100],"routing":{}}],
		10:[None,{"alpha":[0,0,1],"beta":[100,100,100],"routing":{}}],
		20:[None,{"alpha":[0,0,1],"beta":[100,100,90],"routing":{}}],
		29:[None,{"alpha":[0,0,1],"beta":[100,100,81],"routing":{}}],
		30:[dict(layer_num=2,elements=2),{"alpha":[0,1,1],"beta":[100,100,81],"routing":{(2,2):[1,[0,1],0]}}],
		32:[dict(layer_num=2,elements=2,value=0),{"alpha":[0,1,1],"beta":[100,100,81],"routing":{}}],
		34:[dict(layer_num=2,elements=2),{"alpha":[0,1,1],"beta":[100,100,81],"routing":{(2,2):[1,[0,1],0]}}],
		35:[None,{"alpha":[0,1,1],"beta":[100,100,81],"routing":{(2,2):[1,[0,1],0]}}],
		40:[None,{"alpha":[0,1,1],"beta":[100,100,81],"routing":{(2,2):[1,[0,1],0]}}],
		60:[None,{"alpha":[0,1,1],"beta":[100,80,64],"routing":{(2,2):[1,[0,1],1.6]}}],
		69:[None,{"alpha":[0,1,1],"beta":[100,71,55],"routing":{(2,2):[1,[0,1],2.5]}}],
		70:[dict(layer_num=[1,2],elements=3),{"alpha":[1,1,1],"beta":[100,71,55],"routing":{(2,2):[1,[0,1],2.6],(2,3):[1,[2,3],0],(1,3):[0,[0,1],0]}}],
		75:[None,{"alpha":[1,1,1],"beta":[100,71,55],"routing":{(2,2):[1,[0,1],3.1],(2,3):[1,[2,3],0],(1,3):[0,[0,1],0]}}],
		80:[None,{"alpha":[1,1,1],"beta":[100,70,54],"routing":{(2,2):[1,[0,1],3.6],(2,3):[1,[2,3],0],(1,3):[0,[0,1],0]}}],
		100:[None,{"alpha":[1,1,1],"beta":[80,50,34],"routing":{(2,2):[1,[0,1],5],(2,3):[1,[2,3],2],(1,3):[0,[0,1],2]}}],
		130:[None,{"alpha":[1,1,1],"beta":[50,20,4],"routing":{(2,2):[1,[0,1],5],(2,3):[1,[2,3],5],(1,3):[0,[0,1],5]}}],
		200:[None,{"alpha":[1,1,1],"beta":[0,0,0],"routing":{(2,2):[1,[0,1],5],(2,3):[1,[2,3],5],(1,3):[0,[0,1],5]}}],
		}
	check_network(check_dict, network=n)

def test_network_CondKLDTrigger():
	b = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	a = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	c = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	n = network.CondKLDTrigger(3, layerhps=[c,a,b],
			routing_start_step=0,
		alpha_kw={"duration":1, "start_val":0, "final_val":1, "start_step":0},
		num_child=2)

	# interference during start step
	# {step:[model set, expected anser]}
	check_dict = {
		0:[None,{"alpha":[0,0,1],"beta":[100,100,100],"routing":{}}],
		5:[None, {"alpha":[0,0,1],"beta":[100,100,100],"routing":{}}],
		10:[None,{"alpha":[0,0,1],"beta":[100,100,100],"routing":{}}],
		20:[None,{"alpha":[0,0,1],"beta":[100,100,90],"routing":{}}],
		29:[None,{"alpha":[0,0,1],"beta":[100,100,81],"routing":{}}],
		30:[dict(layer_num=2,elements=2),{"alpha":[0,1,1],"beta":[100,100,81],"routing":{(2,2):[1,[0,1]]}}],
		35:[None,{"alpha":[0,1,1],"beta":[100,100,81],"routing":{(2,2):[1,[0,1]]}}],
		40:[None,{"alpha":[0,1,1],"beta":[100,100,80],"routing":{(2,2):[1,[0,1]]}}],
		60:[None,{"alpha":[0,1,1],"beta":[100,80,60],"routing":{(2,2):[1,[0,1]]}}],
		69:[None,{"alpha":[0,1,1],"beta":[100,71,51],"routing":{(2,2):[1,[0,1]]}}],
		70:[dict(layer_num=[1,2],elements=3),{"alpha":[1,1,1],"beta":[100,71,51],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		75:[None,{"alpha":[1,1,1],"beta":[100,71,51],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		80:[None,{"alpha":[1,1,1],"beta":[100,70,50],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		100:[None,{"alpha":[1,1,1],"beta":[80,50,30],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		130:[None,{"alpha":[1,1,1],"beta":[50,20,0],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		200:[None,{"alpha":[1,1,1],"beta":[0,0,0],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		}
	check_network(check_dict, network=n)
	assert False

"""
def test_network_CondKLDTriggerLayerMask():
	b = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	a = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	c = layers.RecentBetaHold(beta_anneal_duration=100, start_beta=100, final_beta=0, 
		wait_steps=10, start_step=10)
	n = network.CondKLDTriggerLayerMask(3, layerhps=[c,a,b], num_child=2)

	check_dict = {
		0:[None,{"alpha":[0,0,1],"beta":[100,100,100],"routing":{}}],
		5:[None, {"alpha":[0,0,1],"beta":[100,100,100],"routing":{}}],
		10:[None,{"alpha":[0,0,1],"beta":[100,100,100],"routing":{}}],
		20:[None,{"alpha":[0,0,1],"beta":[100,100,90],"routing":{}}],
		29:[None,{"alpha":[0,0,1],"beta":[100,100,81],"routing":{}}],
		30:[dict(layer_num=2,elements=2),{"alpha":[0,1,1],"beta":[100,100,81],"routing":{(2,2):[1,[0,1]]}}],
		32:[dict(layer_num=2,elements=2,value=0),{"alpha":[0,1,1],"beta":[100,100,81],"routing":{}}],
		34:[dict(layer_num=2,elements=2),{"alpha":[0,1,1],"beta":[100,100,81],"routing":{(2,2):[1,[0,1]]}}],
		35:[None,{"alpha":[0,1,1],"beta":[100,100,81],"routing":{(2,2):[1,[0,1]]}}],
		40:[None,{"alpha":[0,1,1],"beta":[100,100,81],"routing":{(2,2):[1,[0,1]]}}],
		60:[None,{"alpha":[0,1,1],"beta":[100,80,64],"routing":{(2,2):[1,[0,1]]}}],
		69:[None,{"alpha":[0,1,1],"beta":[100,71,55],"routing":{(2,2):[1,[0,1]]}}],
		70:[dict(layer_num=[1,2],elements=3),{"alpha":[1,1,1],"beta":[100,71,55],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		75:[None,{"alpha":[1,1,1],"beta":[100,71,55],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		80:[None,{"alpha":[1,1,1],"beta":[100,70,54],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		100:[None,{"alpha":[1,1,1],"beta":[80,50,34],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		130:[None,{"alpha":[1,1,1],"beta":[50,20,4],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		200:[None,{"alpha":[1,1,1],"beta":[0,0,0],"routing":{(2,2):[1,[0,1]],(2,3):[1,[2,3]],(1,3):[0,[0,1]]}}],
		}
	check_network(check_dict, network=n, is_assert=False)
	latent_element = (2,2)	
	cprint.warning(latent_element, n(latent_element=latent_element)["latent_mask"])
	latent_element = (2,0)	
	cprint.warning(latent_element, n(latent_element=latent_element)["latent_mask"])
	latent_element = (2,3)	
	cprint.warning(latent_element, n(latent_element=latent_element)["latent_mask"])
	latent_element = (1,0)	
	cprint.warning(latent_element, n(latent_element=latent_element)["latent_mask"])
	latent_element = (1,3)
	cprint.warning(latent_element, n(latent_element=latent_element)["latent_mask"])

#"""

def run_plots():
	class CustomNetwork(network.SingleLayer):
		# use lowest expressive layer first.
		def get_network_params(self,step,model,**kw):
			hp = {}
			alpha=[0 for _ in range(self.num_layers)]
			alpha[0]=1
			hp["alpha"]=alpha
			return hp	
			
	hps=CustomNetwork(
				num_layers=3, layerhps = [
				layers.SpecifiedBetaHold(beta_anneal_duration=30000, start_beta=80, final_beta=8,wait_steps=5000, start_step=5000, converge_beta=300),
				None,
				None,
				])

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

if __name__ == '__main__':
	run_plots()