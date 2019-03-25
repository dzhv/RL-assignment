class Logger(object):
	def __init__(self, output_file):
		self.output_file = output_file

	def log(self, message):
		with open(self.output_file, 'a') as f:
			f.write(message + "\n")
