# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from matplotlib import pyplot as plt
from skimage import feature
from PIL import Image
import pdb
import glob
from pathlib import Path
import pickle
import os


def rect_to_bb(rect):

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)
	
def lbpFunction(image):

	grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	(x, y) = grayImg.shape
	lbpImg = np.zeros((x,y))
	#   ||j-1| j |j+1|
	#i-1|| 7 | 6 | 5 |
	# i || 0 | c | 4 |
	#i+1|| 1 | 2 | 3 |

	for i in range(1,x-1):
		for j in range(1,y-1):
			code = 0
			center = grayImg[i,j]
			code |= (grayImg[i-1,j-1]<=grayImg[i,j])<<7
			code |= (grayImg[i-1,j]<=grayImg[i,j])<<6
			code |= (grayImg[i-1,j+1]<=grayImg[i,j])<<5
			code |= (grayImg[i,j+1]<=grayImg[i,j])<<4
			code |= (grayImg[i+1,j+1]<=grayImg[i,j])<<3
			code |= (grayImg[i+1,j]<=grayImg[i,j])<<2
			code |= (grayImg[i+1,j-1]<=grayImg[i,j])<<1
			code |= (grayImg[i,j-1]<=grayImg[i,j])<<0
			
			lbpImg[i,j] = code

	return lbpImg

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

gazeDirections = ([('00.Centre/', 0), ('01.UpRight/', 1), ('02.UpLeft/', 2), 
('03.Right/', 3), ('04.Left/',4), ('05.DownRight/', 5), ('06.DownLeft/', 6)])
faceParts = ([('right_eye', (17, 22),((36, 42))), ('left_eye', (22, 27), (42, 48))])
features = np.array([])

for folder, label in gazeDirections :
	for j in range(1, 38):
		path = 'E:/An 4/licenta/Eye_chimeraToPublish/' + folder +str(j)
		#pdb.set_trace()
		for filename in Path(path).glob('*.jpg'):
			#pdb.set_trace()
			im = cv2.imread(str(filename))
			image = imutils.resize(im, width=500)
			#cv2.imshow("Image", image)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()	
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			#pdb.set_trace()

			rects = detector(gray, 1)
			k = 0
			roi = []

			for (i, rect) in enumerate(rects):
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)
				
				for (name, (i, j), (m,n)) in faceParts:
					clone = image.copy()
					cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			 
					for (x, y) in shape[i:j]:
						cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
					#pdb.set_trace()
					(x, y, w, h) = cv2.boundingRect(np.array([np.concatenate((shape[i:j], shape[m:n]), axis =0)]))
					roi.append(image[y:y + h, x:x + w])
					roi[k] = imutils.resize(roi[k], width=240, inter=cv2.INTER_CUBIC)

					#cv2.imshow(name, roi[k])
					#cv2.imshow("Image", clone)
					#cv2.waitKey(0)
					#cv2.destroyAllWindows()
					
					k = k+1

				#output = face_utils.visualize_facial_landmarks(image, shape)
				#cv2.imshow("Image", output)
				#cv2.waitKey(0)
				
			#pdb.set_trace()
			for i in range(0,k):
				lbpValues = []
				pdb.set_trace()
				imgwidth, imgheight, dim = roi[i].shape
				croppedHeight = int(imgheight/4)
				croppedWidth = int(imgwidth/2)
				for h in range(0, imgheight, croppedHeight):
					for j in range(0, imgwidth, croppedWidth):
						croppedImg = roi[i][j:j+croppedWidth, h:h+croppedHeight]
						lbpImg = lbpFunction(croppedImg)
						cv2.imshow("lbpImg", lbpImg/255)
						cv2.imshow("Img", croppedImg)
						cv2.waitKey(0)
						cv2.destroyAllWindows
						lbpValues = np.append(lbpValues, lbpImg.ravel())
			n, bins, patches = plt.hist(lbpValues.ravel(), density=False, bins=256, range=(0, 256))
			plt.xlabel('Local Binary Patterns')
			plt.ylabel('No of pixels')
			plt.axis([0, 256, 0, 0.03])
			plt.grid(True)
			#plt.show()
			pdb.set_trace()
			imgFeaturesPath = os.path.splitext(filename)[0] + 'Split.pkl'
			with open(imgFeaturesPath,'wb') as f:
				pickle.dump(n, f)
				
			if features.size:
				features = np.vstack([features, n]) 
			else: 
				features = n
					
			
			
#pdb.set_trace()
with open('featuresDataSplit.pkl','wb') as f:
		pickle.dump(features, f)

#pdb.set_trace()