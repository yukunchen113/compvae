import disentangle as dt
import tensorflow as tf 
import numpy as np 
import importlib.util
import copy
##############
# Structures #
##############
class ExactTransformationRouting:
	def __init__(self):
		self.routing = None
		self.exact_transform_systems = None

	def set_routing(self, routing):
		#routing: {(prior layer num, prior element num):[layer num, [conditioned elements]]}
		self.routing = routing
		self.check_valid_routing(routing)
		self.exact_transform_systems, self.all_nodes = self.get_all_paths(routing)

	def check_valid_routing(self, routing):
		children_nodes = []
		for k,v in routing.items():
			# children must be located on lower layers.
			assert k[0]<v[0], "children must be located on lower layers."
			children_nodes+=[(v[0],i) for i in v[1]]
		
		root = False
		for k in routing.keys():
			if k[0]:
				assert k in children_nodes, "nodes from layer 1 onwards must have a parent"
			else:
				root = True
		assert root, "roots must start at layer 0"

	def bfs_get_children(self, layer, routing, rmap=None):
		reversed_routing = {}
		leaf_nodes = []
		edges = []
		# custom bfs traversal of tree
		while True:
			lsize = len(layer)
			if not lsize:
				break
			for _ in range(lsize):
				node = layer.pop(0)
				if node in routing:
					for child in routing[node]:
						if not rmap is None: rmap[child]=list(set(rmap[child]+rmap[node]+[node]))
						layer.append(child)
						edges.append((node, child))

						# add to revered routing
						if not child in reversed_routing:
							reversed_routing[child]=[node]
						else:
							reversed_routing[child]+=[node]
				else:
					leaf_nodes.append(node)
		return rmap, list(set(leaf_nodes)), reversed_routing, edges

	def get_all_paths(self, routing):
		routing = {k:[(v[0],i) for i in v[1]] for k,v in routing.items()}
		rmap = {i:[] for v in routing.values() for i in v}
		rmap.update({k:[] for k in routing.keys()})

		# get valid routing (no parents or children)
		prmap, leaf_nodes, reversed_routing, *_ = self.bfs_get_children( # top down to get parents
			[k for k in rmap if not k[0]], routing, copy.deepcopy(rmap))
		crmap,*_=self.bfs_get_children(copy.deepcopy(leaf_nodes), reversed_routing, rmap) # bottom up to get children
		rmap = {k:crmap[k]+prmap[k] for k in prmap.keys()}
		rmap = {k:sorted([i for i in rmap.keys() if not (i in v or i == k)]) for k,v in rmap.items()} #inverse rmap
		
		# check to see complete information
		# check to see this seeing if all paths to the leaf nodes are specified
		# we have already checked that each lower layer/leaf has a parent, and that the hierarchy leads up to the root nodes
		# which contain complete information of the system.
		leaf_edges = [(parent, leaf) for leaf in leaf_nodes for parent in reversed_routing[leaf]]
		edges_traversed = {k:self.bfs_get_children([k], routing)[3] for k in rmap.keys()}

		def complete_info(arr):
			cur_info = []
			for el in arr:
				cur_info+=edges_traversed[el]
				if el in leaf_nodes:
					cur_info+=[(parent, el) for parent in reversed_routing[el]]
			return np.all([el in cur_info for el in leaf_edges])

		def get_complements(arr):
			com = None
			for k in arr:
				if com is None:
					com = rmap[k]
				else:
					com = [n for n in com if n in rmap[k]]
			return com


		complete_paths = set()
		self.__memo_complete_check = {}
		for k,v in rmap.items():
			a = self.check_complete_info([k],complete_info,get_complements)
			complete_paths.update(a)
		self.__memo_complete_check = {} # clear
		complete_paths = list(complete_paths)
		return complete_paths, list(rmap.keys())

	def check_complete_info(self, carr, complete_info, get_complements):
		# will check if a combination of the all items in carr and some in aarr, mapped with vmap
		# will be able to completely describe complete arr.
		# if is complete, will return the arrays that are complete.

		carr = tuple(sorted(carr))
		if carr in self.__memo_complete_check:
			ans = self.__memo_complete_check[carr]
			return ans
		
		if complete_info(carr):
			ans = [carr]
			self.__memo_complete_check[carr]=ans
			return ans
		comp=get_complements(carr)
		if comp == []:
			self.__memo_complete_check[carr]=None
			return None
		ans = []
		for c in comp:
			a=self.check_complete_info([*carr,c], complete_info, get_complements)
			if not a is None:
				ans+=a
		self.__memo_complete_check[carr]=ans
		return ans

	def __call__(self, nodes=[]):
		"""Returns possible paths which allow for exact transformation of routing hierarchy
		
		input nodes will be evaluated to be independent (no parents or children) and will return a list of valid nodes
		to add to graph.

		Args:
			nodes (List(Tuples)): a list of nodes, if is empty will return all nodes
		
		Returns:
			list of possible paths given nodes or None if invalid
		"""
		assert not self.routing is None, "routing not defined"
		assert type(nodes) == list, "nodes parameter must be list but is %s for %s"%(str(type(nodes)),str(nodes))
		if nodes == []:
			return copy.deepcopy(self.exact_transform_systems)
		for i in nodes:
			assert type(i) == tuple, "elements in node must be tuple but is %s for %s"%(str(type(i)),str(i))
			assert i in self.all_nodes, f"node {i} not recognized in {self.all_nodes}"

		paths = []
		for systems in self.exact_transform_systems:
			valid_sys = np.all([n in systems for n in nodes])
			if valid_sys:
				paths.append(systems)
		if paths == []:
			return None
		return paths


#################
# Visualization #
#################
class VLAETraversal(dt.visualize.Traversal): #create wrapper for model encoder and decoder

	"""
	TODO:
		- add capabilities for image vertical stack for each hierarchical layer.
	"""
	
	def __init__(self,*ar,**kw):
		super().__init__(*ar, **kw)
		self.latent_hierarchy = None

	def encode(self, inputs):
		latent_space = self.model.encoder(inputs)
		# latent space is of shape [num latent space, 3-[sanmes, mean, logvar], batchsize, num latents]
		self.latent_hierarchy = [i[0].shape[-1] for i in latent_space]
		latent_space = tf.concat(latent_space, -1)
		return latent_space

	def decode(self, samples):
		samples = np.split(samples, np.cumsum(self.latent_hierarchy)[:-1], axis=-1) # we don't include last as that is the end
		ret = self.model.decoder(samples)
		return ret
	@property
	def num_latents(self):
		num_latents = sum([i.num_latents for i in self.model.ladders if not i is None])+self.model.num_latents
		return num_latents

	@property
	def samples_list(self):
		s = self.samples.shape
		new_shape = (s[0],-1,np.sum(self.latent_hierarchy),*s[2:])
		samples = self.samples.reshape(new_shape)
		samples = np.split(samples, np.cumsum(self.latent_hierarchy)[:-1], axis=2)
		self.inputs = np.broadcast_to(np.expand_dims(self.orig_inputs,1), samples[0].shape)
		self.inputs = self.inputs.reshape(self.inputs.shape[0],-1, *self.inputs.shape[-3:])
		samples = [i.reshape(i.shape[0],-1, *i.shape[-3:]) for i in samples]
		samples = [self.inputs]+samples
		return samples

def vlae_traversal(model, inputs, min_value=-3, max_value=3, num_steps=30, is_visualizable=True, latent_of_focus=None, Traversal=VLAETraversal, return_traversal_object=False):
	"""Standard raversal of the latent space
	
	Args:
		model (Tensorflow Keras Model): Tensorflow VAE from utils.tf_custom
		inputs (numpy arr): Input images in NHWC
		min_value (int): min value for traversal
		max_value (int): max value for traversal
		num_steps (int): The number of steps between min and max value
		is_visualizable (bool, optional): If false, will return a traversal tensor of shape [traversal_steps, num_images, W, H, C]
		Traversal (Traversal object, optional): This is the traversal object to use
		return_traversal_object (bool, optional): Whether to return the traversal or not
	
	Returns:
		Numpy arr: image
	"""
	t = dt.general.tools.Timer()
	traverse = Traversal(model=model, inputs=inputs)
	#t("Timer Creation")
	if latent_of_focus is None:
		traverse.traverse_complete_latent_space(min_value=min_value, max_value=max_value, num_steps=num_steps)
	else:
		traverse.traverse_latent_space(latent_of_focus=latent_of_focus, min_value=min_value, max_value=max_value, num_steps=num_steps)

	#t("Timer Traversed")
	traverse.create_samples()
	#t("Timer Create Samples")
	if return_traversal_object:
		return traverse
	if not is_visualizable:
		return traverse.samples
	image = traverse.construct_single_image()
	return image 
