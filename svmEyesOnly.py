import numpy as np
import pdb
import glob
from pathlib import Path
import pickle
import os
from sklearn import svm
import matplotlib.pyplot as plt

def scaleMatrix(matrix, maxMin):
	x,y = matrix.shape
	out = np.zeros((x,y))
	out_range = [-1,1]
	for i in range(0,x):
		for j in range(0,y):
			out[i,j] = (matrix[i,j] - (maxMin[0,y-j-1] + maxMin[1,y-j-1])/2) / (maxMin[0,y-j-1] - maxMin[1,y-j-1])
			#scaling to [-1,1]
			out[i,j] = out[i,j] * (out_range[1] - out_range[0]) +(out_range[1] +out_range[0])/2 
	return out
	
def scaleVector(vector, maxMin):
	y = vector.size
	out = np.zeros(y)
	out_range = [-1,1]
	for j in range(0,y):
		out[j] = (vector[j] - (maxMin[0,y-j-1] + maxMin[1,y-j-1])/2) / (maxMin[0,y-j-1] - maxMin[1,y-j-1])
		#scaling to [-1,1]
		out[j] = out[j] * (out_range[1] - out_range[0]) +(out_range[1] +out_range[0])/2 
	return out

def confRate(confMatrix):
	trueGuesses = 0
	wrongGuesses = 0
	for i in range (0,7):
		for j in range (0,7):
			if i == j:
				trueGuesses += confMatrix[i][j]
			else:
				wrongGuesses += confMatrix[i][j]	
	rate = trueGuesses/float(trueGuesses+wrongGuesses) *100
	return rate

with open('featuresDataEyesOnly.pkl', 'rb') as f:
	features = pickle.load(f)

x, y = features.shape
deletedColumns = []
maximum = []
minimum = []
maxMin =  []
maximumDeleted = []
maximumDeleted = []
maxMinDeleted = []
currRate = []
confMatrix = np.zeros((7,7))
gazeDirections = ([('00.Centre/', 0), ('01.UpRight/', 1), ('02.UpLeft/', 2), 
('03.Right/', 3), ('04.Left/',4), ('05.DownRight/', 5), ('06.DownLeft/', 6)])


for i in range(y-1,-1,-1):
	#pdb.set_trace()
	maximumDeleted = np.append(maximum,features[:,i].max())
	minimumDeleted = np.append(minimum,features[:,i].min())
	if (features[:,i].max()-features[:,i].min()) < 5:
		deletedColumns = np.append(deletedColumns, i)
		features = np.delete(features,i,1)
		#pdb.set_trace()
	else:
		maximum = np.append(maximum,features[:,i].max())
		minimum = np.append(minimum,features[:,i].min())
maxMin = np.vstack([maximum, minimum])
maxMinDeleted = np.vstack([maximumDeleted, minimumDeleted])
x, y = features.shape
#pdb.set_trace()

features = scaleMatrix(features, maxMin)
#pdb.set_trace()

clf = svm.SVC(gamma = 0.01, C = 100)

test = 1
while test <38:
	featuresTrain = np.array([])
	labels = np.array([])
	for folder, label in gazeDirections :
		for j in range(1, 38):
			path = 'E:/An 4/licenta/Eye_chimeraToPublish/' + folder +str(j)
			for filename in Path(path).glob('*.jpg'):
				imgFeaturesPath = os.path.splitext(filename)[0] + 'EyesOnly.pkl'
				with open(imgFeaturesPath,'rb') as f:
					if j != test:
						featuresImg = pickle.load(f)
						featuresImg = np.delete(featuresImg,deletedColumns)
						#pdb.set_trace()
						featuresImg = scaleVector(featuresImg, maxMin)
						if featuresTrain.size:
							featuresTrain = np.vstack([featuresTrain, featuresImg]) 
							labels = np.append(labels, label)
						else: 
							featuresTrain = featuresImg
							labels = label
						#pdb.set_trace()
	clf.fit(featuresTrain,labels)
	confMatrixCurrent = np.zeros((7,7))
	for folder, label in gazeDirections :
		path = 'E:/An 4/licenta/Eye_chimeraToPublish/' + folder +str(test)
		for filename in Path(path).glob('*.jpg'):
			imgFeaturesPath = os.path.splitext(filename)[0] + 'EyesOnly.pkl'
			with open(imgFeaturesPath,'rb') as f:
				featuresImg = pickle.load(f)
				featuresImg = np.delete(featuresImg,deletedColumns)
				#pdb.set_trace()
				featuresImg = scaleVector(featuresImg, maxMin)
				#print('Prediction', clf.predict(featuresImg.reshape(1,-1)))
				#print('Label should be ', label)
				#pdb.set_trace()
				confMatrixCurrent[label, clf.predict(featuresImg.reshape(1,-1))[0]] += 1
				confMatrix[label, clf.predict(featuresImg.reshape(1,-1))[0]] += 1
				
	test += 1
	currRate = np.append(currRate, confRate(confMatrixCurrent))
	#pdb.set_trace()
rate = confRate(confMatrix)
pdb.set_trace()

