## Setup:
Requires my libraty repo, please add that to your PYTHONPATH, such that the utils file can be used.

## Research Journal
Here are some research notes.

### Experiment 1: Train BetaTCVAE, Create Mask by Randomizing Latents (Current)
This experiment will try to create a mask by randomizing nuisance latent variables. These should randomize latents according to correlation between the latent variable of interest.
There should be a degree of correlation between all of these variables since we are focusing on data that contain one object.

We can make the mask by subtracting with original generated image, an applying a proportional gaussian filter onto the image, creating a mask


#### Experiment 1.1: Mask type - small change in selected element
Purpose - create mask for current image

##### Experiment 1.1.1: Initial Mask Creation

We can see in the images below that the effects on the mask The mask was made using a threshold of 0.005, to determine what to mask. These are the deltas of image i and image i+1. 

![intw1](experiments/interweaved_mask1.png)
![m1](experiments/mask_between_latents1.png)

##### Experiment 1.1.2: Varying differences in delta
TBD

#### Experiment 1.2: Mask type - small change in selected element while randomizing the image to other elements
Purpose - randomization causes the conditioned latent dimension to ignore the other latents which are correlated.

TBD

### Experiment 2: Connect Masks with Original Images
#### Experiment 2.1: Data creation
1. create masking object and apply it to the original image, analysis of effects
2. resize masking image and apply it to a larger image, analysis of effects
3. apply to images and create dataset


#### Experiment 2.2: Model Training
1. run model
2. run regular model with same size images for baseline
3. analysis of results

### Experiment 3: Retrain original model
#### Experiment 3.1: Conditioning Original Model on Other Trained Model
