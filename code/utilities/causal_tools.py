import numpy as np 
import copy
##############
# Structures #
##############
class ExactTransformationRouting:
	def __init__(self):
		self.routing = {}
		self.exact_transform_systems = []
		self.all_nodes = []
		self.reversed_routing = {}

	def set_routing(self, routing):
		#routing: {(prior layer num, prior element num):[layer num, [conditioned elements]]}
		if routing == self.routing: return
		self.routing = routing
		if routing == {}:
			self.exact_transform_systems = []
			self.all_nodes = []
		else:
			self.check_valid_routing(routing)
			self.exact_transform_systems, self.all_nodes = self.get_all_paths(routing)

	def check_valid_routing(self, routing):
		children_nodes = []
		self.root_layer = -1
		for k,v in routing.items():
			# children must be located on lower layers.
			assert k[0]>v[0], "children must be located on lower layers."
			children_nodes+=[(v[0],i) for i in v[1]]
			self.root_layer = max(self.root_layer,k[0])
		root = False
		for k in routing.keys():
			if not k[0]==self.root_layer:
				assert k in children_nodes, "nodes from layer 1 onwards must have a parent"
			else:
				root = True
		assert root, f"roots must start at layer {self.root_layer}"

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
		prmap, leaf_nodes, self.reversed_routing, *_ = self.bfs_get_children( # top down to get parents
			[k for k in rmap if k[0]==self.root_layer], routing, copy.deepcopy(rmap))
		crmap,*_=self.bfs_get_children(copy.deepcopy(leaf_nodes), self.reversed_routing, rmap) # bottom up to get children
		rmap = {k:crmap[k]+prmap[k] for k in prmap.keys()}
		rmap = {k:sorted([i for i in rmap.keys() if not (i in v or i == k)]) for k,v in rmap.items()} #inverse rmap
		# check to see complete information
		# check to see this seeing if all paths to the leaf nodes are specified
		# we have already checked that each lower layer/leaf has a parent, and that the hierarchy leads up to the root nodes
		# which contain complete information of the system.
		leaf_edges = [(parent, leaf) for leaf in leaf_nodes for parent in self.reversed_routing[leaf]]
		edges_traversed = {k:self.bfs_get_children([k], routing)[3] for k in rmap.keys()}

		def complete_info(arr):
			cur_info = []
			for el in arr:
				cur_info+=edges_traversed[el]
				if el in leaf_nodes:
					cur_info+=[(parent, el) for parent in self.reversed_routing[el]]
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
