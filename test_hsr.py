import hsr.dataset as ds
def test_dataset():
	dataset = ds.CelebA()
	print(dataset.preprocess(dataset.test()).shape)
	print(dataset.preprocess(dataset.train(32)).shape)

# test model save and load
# test saver
import os
from hsr.model.vae import LVAE,VLAE
from hsr.save import InitSaver, ModelWeightSaver, ModelSaver
from hsr.utils.regular import cprint
def test_model_save():
	savedir = "dev/model_save_load/"
	if not os.path.exists(savedir): os.makedirs(savedir)

	modelsaver = ModelSaver(savedir)

	dataset = ds.CelebA()
	data = dataset.preprocess(dataset.test()[:32])

	Model = modelsaver(LVAE) # wrap LVAE so it saves initialization parameters
	model = Model()
	out = model(data)
	print(out.shape)
	modelsaver.save(model)

	model = modelsaver.load()
	out = model(data)
	print(out.shape)



# test dataset 


if __name__ == '__main__':
	test_model_save()
