class BaseLog:
	# provides base funcitonality
	def __init__(self, basepath):
		self.basepath = basepath
	def __call__(self):
		pass	

class TextLog:
	def __init__(self,basepath):
		super().__init__(basepath)
		self._logger = None
	@property
	def logger(self):
		if self._logger is None: self._logger = open(self.basepath,"a")
		return self._logger
	def __call__(self, string, end = "\n"):
		self.logger.write(string+end)
	def __del__(self):
		self.logger.close()
	def __exit__(self, *ar):
		self.logger.close()