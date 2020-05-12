import time, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loader import *
from batch_generator import *
from rmtpp import *
from plot import *
from scipy.integrate import quad

param = {
	'numClasses': 7,
	'embedDim': 16,
	'dropRate': 0.1,
	'lstmDim': 32,
	'numRNNLayers': 1,
	'mapDim': 16,
	
	'seqLength': 10,
	'batchSize': 256,
	'testBatchSize': 8192,
	'alpha': 0.3,	# timeLoss * alpha + eventLoss
	'numEpochs': 50,
	'lr': 0.001,
	
	'device': torch.device('cpu'),
	}
if torch.cuda.is_available():
	param['device'] = torch.device('cuda:0')

# load train.csv
dl = DataLoader('train')
dl.load()
trainSeq = dl.genSeq(param['seqLength'] + 1)
numPerClass = np.array(dl.getNum())
totalNum = numPerClass.sum()
eventLossWeight = torch.from_numpy((totalNum / numPerClass).astype(np.float32))
bg = BatchGenerator(trainSeq, param['batchSize'])

# load test.csv
dltest = DataLoader('test')
dltest.load()
testSeq = dltest.genSeq(param['seqLength'] + 1)
bgtest = BatchGenerator(testSeq, param['testBatchSize'])
numPerClassTest = np.array(dltest.getNum())

model = RMTPP(param).to(param['device'])
optimizer = torch.optim.RMSprop(model.parameters(), lr = param['lr'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

classes = np.array(['PRT', 'CNG', 'IDC', 'COMM', 'LMTP', 'MISC', 'TIKT'])

print('===== Training Started =====')
os.system('rm -f rmtpp.log')
os.system('rm -f rmtpp_plots/*')
ff = open('rmtpp.log', 'a+')
stTime = int(time.time())

for epoch in range(param['numEpochs']):
	model.train()
	scheduler.step()
	batchNo = 0
	timeLossList = []
	eventLossList = []
	lossList = []
	bg.rewind()
	while bg.hasNextBatch():
		model.zero_grad()
		batch = bg.nextBatch()
		input = batch[:, 0:-1, :]

		target = batch[:, -1, :]
		optimizer.zero_grad()
		timeOut, eventOut, lastTime = model.forward(input)
		
		eventLossWeight = torch.as_tensor(eventLossWeight, dtype = torch.float, device = param['device'])
		timeOut = timeOut.reshape(param['batchSize'])
		timeLoss, eventLoss, loss = model.loss(timeOut, eventOut, target, eventLossWeight, lastTime)
		
		timeLossList.append(timeLoss.item())
		eventLossList.append(eventLoss.item())
		lossList.append(loss.item())
		
		loss.backward()
		optimizer.step()
		if batchNo % 200 == 0:
			print('Epoch: %d\tBatch: %d\tLoss: %f\tTimeLoss: %f\tEventLoss: %f' % (epoch, batchNo, np.array(lossList).mean(), np.array(timeLossList).mean(), np.array(eventLossList).mean()))
		batchNo += 1
	
	
	if epoch < param['numEpochs'] - 1:
		continue
	
	# evaluation on test dataset
	model.eval()
	with torch.no_grad():
		result = np.zeros((param['numClasses'], param['numClasses'])).astype(int)
		deltaTime = 0.0
		totalCnt = 0
		realCount = np.array([0, 0, 0, 0, 0, 0, 0])
		plotpred = []
		plottrue = []
		batchNo = 0
		print('===== Evaluating =====')
		while bgtest.hasNextBatch():
			print('batch No.', batchNo)
			batchNo += 1
			batch = bgtest.nextBatch()
			input = batch[:, 0 : -1, :]
			target = batch[:, -1, :]
			timeOut, eventOut, lastTime = model.forward(input)
			eventOut = eventOut.cpu().numpy()
			eventOut = np.argmax(eventOut, axis = 1).astype(int)
			eventTarget = target[:, 1].astype(int)
			num = eventOut.shape[0]
			for i in range(num):
				result[eventTarget[i]][eventOut[i]] += 1
				realCount[eventTarget[i]] += 1
				plotpred.append(eventOut[i])
				plottrue.append(eventTarget[i])
			
			lastTime = lastTime.squeeze().cpu().numpy()
			timeOut = timeOut.reshape(param['testBatchSize']).cpu().numpy()
			timeTarget = target[:, 0] - lastTime
			w = model.w.detach().cpu().item()
			b = model.b.detach().cpu().item()
			
			timeOutNew = torch.tensor([lt + quad(lambda t: t * np.exp(pastInflu + w * t + b + (np.exp(pastInflu + b) - np.exp(pastInflu + w * t + b)) / w), a = 0.0, b = 10.0)[0] for pastInflu, lt in zip(timeOut, lastTime)])
			
			timeOutNew = timeOutNew.cpu().numpy()
			deltaTime += np.abs(timeOutNew - timeTarget).sum()
			totalCnt += param['testBatchSize']
				
		bgtest.rewind()
		print('===== Confusion Matrix =====')
		print(result / realCount.reshape(param['numClasses'], -1))
		print('===== Statistics =====')
		mae = deltaTime / totalCnt
		pre = (result.diagonal() / result.sum(axis = 0)).mean()
		rec = (result.diagonal() / result.sum(axis = 1)).mean()
		f1 = 2 * pre * rec / (pre + rec)
		print('MAE: ', mae)
		print('Precision: ', pre)
		print('Recall: ', rec)
		print('F1-score: ', f1)
		
		ff.write('[Epoch ' + str(epoch) + '] MAE: ' + str(mae) + ', Precision: ' + str(pre) + ', Recall: ' + str(rec) + ', F1-score: ' + str(f1) + '\n')
		
		plot_confusion_matrix('./rmtpp_plots/' + str(epoch), plottrue, plotpred, classes, normalize=True)
		print('===== Confusion Matrix figure saved =====')
		
edTime = int(time.time())
ff.close()
print('===== Training Finished in %.2f second =====' % ((edTime - stTime) / 1000.0))	
		
		
		
		