# kaggle.py module

import csv
import numpy as np

#Save prediction vector in Kaggle CSV format
#Input must be a Nx1, 1XN, or N long numpy vector
def kaggleize(predictions,file):

	if(len(predictions.shape)==1):
		predictions.shape = [predictions.shape[0],1]

	ids = 1 + np.arange(predictions.shape[0])[None].T
	kaggle_predictions = np.hstack((ids,predictions))
	writer = csv.writer(open(file, 'w'))
	if predictions.shape[1] == 1:
		writer.writerow(['# id','Prediction'])
	elif predictions.shape[1] == 2:
		writer.writerow(['# id','Prediction1', 'Prediction2'])

	writer.writerows(kaggle_predictions)

