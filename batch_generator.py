import numpy as np

class BatchGenerator:

	def __init__(self, seq, length):
		self.seq = np.array(seq)
		self.total = len(seq)
		self.length = length
		np.random.shuffle(self.seq)
		self.curNo = 0
		
	def hasNextBatch(self):
		return self.curNo < self.total
		
	def nextBatch(self):
		if self.curNo + self.length <= self.total:
			rtn = self.seq[self.curNo : self.curNo + self.length]
			self.curNo += self.length
		else:
			rest = self.length - (self.total - self.curNo)
			rtn = self.seq[self.curNo:]
			randRow = np.arange(0, self.curNo)
			np.random.shuffle(randRow)
			rtn = np.concatenate((rtn, self.seq[randRow[0 : rest]]))
			self.curNo = self.total
		return rtn
		
		
	def rewind(self):
		np.random.shuffle(self.seq)
		self.curNo = 0