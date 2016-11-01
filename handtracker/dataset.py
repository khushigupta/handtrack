# This file assumes that the data is available as datasets/<datasetname>/groundtruth and datasets/<datasetname>/images
# We also assume that for every image there is a corresponding grountruth image and they are present in the same order
# It will randomly create train/test split, rename the files and copy them to img/<dataset>/train and img/<dataset>/test
#																		and mask/<dataset>/train and mask/<dataset>/test
# Files have to be of the form 00000101.jpg/.png

import os
import sys
from random import shuffle
import shutil

root = "/home/khushig/claireCVPR/handtrack/"
datasetName = "GTEA/"
datasetDir = root + "datasets/" + datasetName

dataSplit = 0.8
randomize = 0

imgExt = ".jpg"
maskExt = ".png"
startFileIdx = 101 # make sure this is the same in C++ code
setPrecision = 8

print datasetDir + "groundtruth/"
if os.path.isdir(datasetDir + "groundtruth/"):
	if os.path.isdir(datasetDir + "images/"):
		imgFiles = [name for name in os.listdir(datasetDir + "images/") if os.path.isfile(os.path.join(datasetDir + "images/", name))]
		gtFiles = [name for name in os.listdir(datasetDir + "groundtruth/") if os.path.isfile(os.path.join(datasetDir + "groundtruth/", name))]
		
		assert(len(imgFiles) == len(gtFiles))
		numFiles = len(imgFiles)
		
		if randomize:
			perm = [i for i in range(numFiles)] 
			shuffle(perm)
			imgFiles = [imgFiles[i] for i in perm]
			gtFiles = [gtFiles[i] for i in perm]

		idx = int(dataSplit*numFiles)

		srcImg = datasetDir + "images/" 
		srcMask = datasetDir + "groundtruth/" 
		
		destImgTrain = root + "img/" + datasetName + "train/"
		if not os.path.exists(destImgTrain):
			os.makedirs(destImgTrain)

		destMaskTrain = root + "mask/" + datasetName + "train/"
		if not os.path.exists(destMaskTrain):
			os.makedirs(destMaskTrain)

		destImgTest = root + "img/" + datasetName + "test/"
		if not os.path.exists(destImgTest):
			os.makedirs(destImgTest)

		destMaskTest = root + "mask/" + datasetName + "test/"
		if not os.path.exists(destMaskTest):
			os.makedirs(destMaskTest)

		for i in range(numFiles):
			if i < idx:
				shutil.copy(srcImg + imgFiles[i], destImgTrain)
				os.rename(destImgTrain + imgFiles[i], destImgTrain + str(startFileIdx + i).zfill(setPrecision) + imgExt)

				shutil.copy(srcMask + gtFiles[i], destMaskTrain)
				os.rename(destMaskTrain + gtFiles[i], destMaskTrain + str(startFileIdx + i).zfill(setPrecision) + maskExt)
			else:
				shutil.copy(srcImg + imgFiles[i], destImgTest)
				os.rename(destImgTest + imgFiles[i], destImgTest + str(startFileIdx + i).zfill(setPrecision) + imgExt)

				shutil.copy(srcMask + gtFiles[i], destMaskTest)
				os.rename(destMaskTest + gtFiles[i], destMaskTest + str(startFileIdx + i).zfill(setPrecision) + maskExt)

	else:
		print "images directory does not exist .. exiting"
		sys.exit()
else:
	print "groundtruth directory does not exist .. exiting"
	sys.exit()
