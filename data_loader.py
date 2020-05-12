class DataLoader:
	
	def __init__(self, target):
		self.data = {}
		self.target = target
		self.num = [0, 0, 0, 0, 0, 0, 0]

	def load(self):
		if self.target == 'train':
			f = open('data/train.csv')
		elif self.target == 'test':
			f = open('data/test.csv')
		
		print('===== Loading %s.csv =====' % self.target)
		
		lines = f.readlines()
		cnt = 0
		for line in lines:
			if line.startswith('id'):
				continue
			event = line.split(',')
			atm = event[0]
			day = float(event[1]) / 86400
			type = int(event[2])
			if atm not in self.data.keys():
				self.data[atm] = []
			self.data[atm].append([day, type])
			self.num[type] += 1
			cnt += 1
		
		print('===== %d events loaded from %s.csv =====' % (cnt, self.target))
		
	def genSeq(self, length):
		rtn = []
		
		for atm in self.data.keys():
			events = self.data[atm]
			cnt = len(events)
			for i in range(0, cnt - length + 1):
				add = []
				firstTime = events[i][0]
				for j in range(0, length):
					add.append([events[i + j][0] - firstTime, events[i + j][1]])
				rtn.append(add)
				
		print('===== %d sequences generated =====' % len(rtn))
		return rtn
	
	def getNum(self):
		return self.num
		