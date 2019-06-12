import numpy as np
import scipy.spatial.distance



class AnomalyDetector:
	''' Interface for anomaly detector. 
	Return either probability of being anomaly, or True for detected anomaly.'''
	def __init__(self):
		self.data = None

	def record(self, data):
		data = np.array(data)
		if self.data is None:
			self.data = data
		elif len(data.shape) == 1:
			self.data = np.append(self.data, data)
		else:
			self.data = np.concatenate((self.data, data))

	def classify(self, data, threshold, metric):
		raise NotImplementedError



class KNearestNeighbor(AnomalyDetector):
	''' The K shortest distance between query point and existing data points '''
	def __init__(self, metric='euclidean', k=1):
		# For available metric, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
		assert(metric in ['euclidean', 'cityblock', 'minkowski', 'seuclidean', 'cosine', 'correlation', 'hamming', 'chebyshev', 'canberra', 'braycurtis', 'mahalanobis', 'sokalsneath', 'wminkowski'])

		super(KNearestNeighbor, self).__init__()
		self.metric = metric
		self.k = k

	def getDistance(self, data, metric=None, k=None):
		metric = metric or self.metric
		k = k or self.k
		allDistance = scipy.spatial.distance.cdist(data, self.data, metric)
		topK = np.argpartition(allDistance, k, axis=1)[:,:k]
		kMinDistance = allDistance[np.arange(topK.shape[0])[:, None], topK]
		allMinDistance = kMinDistance.mean(axis=1)
		return allMinDistance

	def classify(self, data, threshold, metric=None, k=None):
		allMinDistance = self.getDistance(data, metric, k)
		return np.greater(allMinDistance, threshold)

