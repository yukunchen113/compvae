## Setup:
Requires my libraty repo, please add that to your PYTHONPATH, such that the utils file can be used.

### Research Journal
Here are some research notes.

#### Experiment 1: Train BetaTCVAE, Create Mask by Randomizing Latents (Current)
This experiment will try to create a mask by randomizing nuisance latent variables. These should randomize latents according to correlation between the latent variable of interest.
There should be a degree of correlation between all of these variables since we are focusing on data that contain one object.

We can make the mask by subtracting with original generated image, an applying a proportional gaussian filter onto the image, creating a mask
