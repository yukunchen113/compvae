# Hierarchical Dataset
- use simple dataset: pyramid
- try dataset on LVAE, ProVLAE, HSR
- perhaps the same as a causal DAG/Tree
	- include non-iid aspects - distribution shift, interventions etc.


# Literature Review
## Paper: Learning Unsupervised Hierarchical Part Decomposition of 3D Objects from a Single RGB Image
Link: https://openaccess.thecvf.com/content_CVPR_2020/papers/Paschalidou_Learning_Unsupervised_Hierarchical_Part_Decomposition_of_3D_Objects_From_a_CVPR_2020_paper.pdf
### Contribution:
- decompose object into rough geometries that match predetermined primatives

### Datasets
- shapenet, D-FAUST 
- 3D objects

### Hierarchy Details:
What kind of hierarchy
- levels of abstraction are defined in terms of complexity
	- level of composition
- geometric hierarchy 
	- composition of basic shapes into complex ones
- hierarchial decomposition of object into primative given in a gradual manner
- used binary tree like decomposition
	- each level gets more primatives (base object)
	- final representation is a combination of the levels of details
		- some aspects could be represented with more or less levels of detail 

Why is it important (motivation)
- find the structural relationship (between parts where parts are compositions of primatives)

How they used hierarchy (metrics, method)
- method:
	- create a binary tree of primatives by spliting the parent
	- maximize actual shape points contained in representation shape
	- minimize outside points contained in representation
	- parent child held together inherently
- metrics:
	- mean Volumetric Intersection over Union
	- Chamfer-L1

### Compared to Us:
Differences:
- they use preset primative
- they only look at positional/rotational relationships between objects, where we look at information
Their Strengths:
- they can create a representation with one datapoint
- easier to make as there are set primatives
- easy to define and children are easily and clearly constrained to the parents.
Our Strengths:
- our parts are meaningful (contain variance)
- we find the primatives (primatives are unknown)
- we learn factors of variation
	- By learning the FOV we can transfer to new domains (eg. audio) and also we might be able to align spaces of these other domains with ours.
	- their FOV are only inhereited through their primatives
		- eg. they only have position, shape, size, rotation 


## Paper: Learning Semantic Representations Of Objects And Their Parts
Link:http://www.iro.umontreal.ca/~lisa/pointeurs/2013_semantic_image_mlj.pdf

### Contribution:
- label objects and their parts without exhaustively labeled data
- leverages information in both labels/language and images to find parts of an object (relations between objects and parts)
	- uses language as proxy supervison to labels parts of images

### Dataset:
- combines Imagenet and Wordnet
	- Imagenet: images for object identificaiton
	- Wordnet: semantic meaning between labels

### Hierarchy Details:
What kind of hierarchy:
- parts: Factors of variation eg. car->wheel
Why is it important (motivation)
- translate between images and labels with relational information
How they used hierarchy (metrics, method)
- method
	- hierarchy is predefined through using the "part of" relation in wordnet
- metric
	- hierarchy is generated through supervised methods so no hierarchy metric
	- precision of image to label is used as metric

### Compared to Us:
Differences:
- their embedding space is word space 
Their Strengths:
- using weakly supervised data, making it easier and more concrete
Our Strengths:
- we use unsupervised to make it generalizable
- they force contextual information by checking labels against other labels
	- seems that it would be hard to generalize to ood objects

## Paper: Unsupervised Discovery of Parts, Structure, and Dynamics
- Contains useful usecases in introduction about psych we can use
Link:https://arxiv.org/pdf/1903.05136.pdf
- Interestingly, I think that since VAEs require smooth traversals to create a continuous space, if we have these motion type videos, we might be able to train a VAE for a more complex scene with multiple objects

### Contribution:
- segment object and parts, build hierarchical structure, model motion distribution during testing 

### Dataset:
- unlabeled video of paired frames during training
- one image during testing

### Hierarchy Details:
What kind of hierarchy
- based on motion a part will contribute motion, but is anchored to the main object
	- \*Note, this is similar to ours in which a FoV is used to represent the main body motion

Why is it important (motivation)
- model objects, make counterfactual inference

How they used hierarchy (metrics, method)
- method
	- VAE conditioned on motion (multiple frames of video)
- metric
	- Intersection over Union with objects
	- hierarchy is not evaluated as part motions are grouped together explicity by the structural descriptor into global motion
		- structural descriptor asserts children must contain parent motion

### Compared to Us:
Differences
- usage of motion as the identifier of an object. We use factors of variation as per https://arxiv.org/abs/1812.02230
Their Strengths:
- easy to evaluate and define hierarchy
- they are able to use multiple objects with varying positions, we can't have a very complicated scene.
Our Strengths:
- as we use factors of variation, we should be more generalizable than just motion and can move to different domains
- we leverage information of the main object to get the parts,
they use the parts to get the main object

## Paper: Learning Physical Graph Representations from Visual Scenes
Link:https://papers.nips.cc/paper/2020/file/4324e8d0d37b110ee1a4f1633ac52df5-Paper.pdf

### Contribution:
- PSGs to represent scenes as hierarcical graphs
	- self supervised
	- combines low and high level information
	- converts spatially uniform feature maps into object-centric graph structures
		- object-centric contains information about the object
	- encourages meaningful scene elemnent identification

### Dataset:
- TDW-Primatives, TDW-Playroom, and Gibson test sets
- complex real-world video
- done with scene

### Hierarchy Details:
What kind of hierarchy
- pools areas of pixels together to form objects. 
- hierarchy seems to be done through identification of objects in a scene
- same as MONet: https://arxiv.org/abs/1901.11390 and Slot Attention: https://arxiv.org/pdf/2006.15055.pdf
	- As a metric these compare object masks with actual object segementation using Adjusted Rand Index
Why is it important (motivation)
- CNNs do not explicitly encode hierarchy and physical properties
- these physical properties (object centric) support high level planning eg. object permanence
How they used hierarchy (metrics, method)
- method:
	- predicts PSGs, which contain
		- parent node to child node relationships
		- attributes of a given node (physical properties)
		- spatialtemporal registrations: mapping of a node to related set of pixels.
			- this matches in order of hierarchy
	- there is a graph pooling operation that determines grouped nodes of a given layer (physically related) given the parent layer's attribues.
		- provides the probability that two nodes are connected
- metrics
	- Recall 
	- mIoU
	- generalization (when transfered to other datasets)

### Compared to Us:
Differences
- they find objects in a scene and use hierarchies to identify objects
- different from us. We want the hierarchies to represent FoV


# Macro-Micro

## Paper
### What is the Hierarchy
### Significance of Hierarchy
### How do they achieve the hierarchy




https://arxiv.org/pdf/1707.00819.pdf

## Paper: Multi-Level Cause-Effect Systems
http://proceedings.mlr.press/v51/chalupka16.pdf

### What is the Hierarchy
- microlevel that increasingly detailed information but are too complex to be able to infer information on a macro scale
- this allows us to make decisions in an abstract space

### Significance of Hierarchy
- microlevel hierarchy can be used to infer information on the macrolevel

### How do they achieve the hierarchy
...

## Paper: Quantifying causal emergence shows that macro can beat micro
https://www.pnas.org/content/110/49/19790
### What is the Hierarchy
"It is widely assumed that, once a micro level is fixed, macro levels are fixed too, a relation called supervenience. It is also assumed that, although macro descriptions may be convenient, only the micro level is causally complete, because it includes every detail"

- supervenience == dependence (X supervenes on Y means that X will only change if Y changes)

### Significance of Hierarchy
- though a micro model is more complete, with sufficient effective information, macro models can perform just as well or better than micro models due to being "more deterministic and/or less degenerative"
	- effective information a mesure of how much the mechanisms in the system can constrain the system's past and future states

### How do they achieve the hierarchy
...



https://www.sciencedirect.com/science/article/abs/pii/0004370294900140
https://arxiv.org/pdf/1412.2309.pdf
https://www.jstor.org/stable/1909285?seq=1
https://arxiv.org/pdf/2010.03635.pdf


# Other Notes:
https://www.sciencedirect.com/science/article/pii/S0042698915000814
	- I don't think that human object detection is limited to motion eg. human brain can adapt to tactical, auditory.
	- human brain should be looking at the factors of variation.
- also different from categorical decision tree hierarchys which decide on an object at each leaf, we decide FoVs all for on object at the leaf.      