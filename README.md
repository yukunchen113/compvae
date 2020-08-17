## Setup:
Requires my library repo, please run
```bash
pip install disentangle
```
## Pipeline:

Currently using a ProVLAE architecture

__Currently, docker does not work__

to execute run:
```python
python code/execute.py
```



## Code/:

### run_traversals.py:
- used to visualize models and run inference

### execute.py:
- this is the file that runs experiments.

### figures
- this folder holds the images used for this readme

### core
This folder holds the tools necessary to run experiments.

#### core/config
core/config/config.py:
- this is one of the files. 
- contains configurations for models to use.
- these are used as default then overwritten by setting attributes.

core/config/addition.py:
- additional functions that are applied to config instances for relevant additional options
- this includes converting a config to be compatible with mask network and a config to be compatible with a comp network 

#### core/model
core/model/model.py:
- this is one of the _main_ files. 
- contains handler objects. These objects load configs into keras models and allow easy training and saving.

core/model/achitectures.py:
- this contains relevant ML architectures such as a modified VAE.

#### core/train
core/train/manager.py:
- Contains training manager objects. These objects train a keras model. This will run the training loop.

core/train/optimizer.py:
- contains the optimzer manager objects, which are used to optimize specific models during training
- These optimizers create the gradient tape, and optimizes. (gradient tape should be custom if model requires custom arguments during inference)

#### utilities
utilities/standard.py:
- commonly used functions and objects across this project.

utilities/mask.py
- commonly used mask functions and objects across this project.
