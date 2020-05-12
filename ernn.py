import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ERNN(nn.Module):
	def __init__(self, param):
		super(ERNN, self).__init__()
		self.param = param
		
		self.embedding = nn.Embedding(self.param['numClasses'], self.param['embedDim'])
		self.dropout = nn.Dropout(self.param['dropRate'])
		self.lstm = nn.LSTM(self.param['embedDim'] + 1, self.param['lstmDim'], batch_first = True)
		self.mapping = nn.Linear(self.param['lstmDim'], self.param['mapDim'])
		self.timeLayer = nn.Linear(self.param['mapDim'], 1)
		self.eventLayer = nn.Linear(self.param['mapDim'], self.param['numClasses'])

	def forward(self, input):
	
		timeSeq = torch.tensor(input[:, :, 0:1], dtype = torch.float, device = self.param['device'])
		eventSeq = torch.tensor(input[:, :, 1], dtype = torch.long, device = self.param['device'])
		
		eventEmb = self.embedding(eventSeq)
		eventEmb = self.dropout(eventEmb)

		timeEvent = torch.cat((timeSeq, eventEmb), dim = 2)
		lstmOut, _ = self.lstm(timeEvent)
		lstmOut = lstmOut[:, -1, :]

		mapOut = self.mapping(lstmOut)
		mapOut = torch.sigmoid(mapOut)

		eventOut = self.eventLayer(mapOut)
		eventOut = F.log_softmax(eventOut, dim = 1)
		
		timeOut = self.timeLayer(mapOut)
		lastTime = timeSeq[:, -1]
		timeOut += lastTime
		
		return timeOut, eventOut
		
	def loss(self, timeOut, eventOut, target, weights):
		timeTarget = torch.tensor(target[:, 0], dtype = torch.float, device = self.param['device'])
		eventTarget = torch.tensor(target[:, 1], dtype = torch.long, device = self.param['device'])
		
		timeLoss = F.mse_loss(timeOut, timeTarget)
		eventLoss = F.cross_entropy(eventOut, eventTarget, weights)
		loss = self.param['alpha'] * timeLoss + eventLoss
		return timeLoss, eventLoss, loss
		
		
		
		
		
		
		
		
		