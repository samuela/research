import numpy as np
import scipy.spatial.distance

class AnomalyDetector:
	''' Interface for anomaly detector. 
	Return either probability of being anomaly, or True for detected anomaly.'''
	def __init__(self):
		self.data = None

	def fit(self, data):
		self.data = np.array(data)

	'''
	def record(self, data):
		data = np.array(data)
		if self.data is None:
			self.data = data
		elif len(data.shape) == 1:
			self.data = np.append(self.data, data)
		else:
			self.data = np.concatenate((self.data, data))
	'''

	def get_score(self, data, *args, **kwargs):
		raise NotImplementedError

	def classify(self, data, threshold, *args, **kwargs):
		score = self.get_score(data, args, kwargs)
		return np.greater(score, threshold)


class KNearestNeighbor(AnomalyDetector):
	''' The K shortest distance between query point and existing data points '''
	def __init__(self, metric='euclidean', k=1):
		# For available metric, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist
		assert(metric in ['euclidean', 'cityblock', 'minkowski', 'seuclidean', 'cosine', 'correlation', 'hamming', 'chebyshev', 'canberra', 'braycurtis', 'mahalanobis', 'sokalsneath', 'wminkowski'])

		super(KNearestNeighbor, self).__init__()
		self.metric = metric
		self.k = k

	def get_score(self, data, metric=None, k=None):
		metric = metric or self.metric
		k = k or self.k
		allDistance = scipy.spatial.distance.cdist(data, self.data, metric)
		topK = np.argpartition(allDistance, k, axis=1)[:,:k]
		kMinDistance = allDistance[np.arange(topK.shape[0])[:, None], topK]
		return kMinDistance.mean(axis=1)

class LocalOutlierFactor(AnomalyDetector):
	''' The density around an outlier object is significantly different from the
	 density around its neighbors '''
	def __init__(self, metric='euclidean', k=5):
		super(LocalOutlierFactor, self).__init__()
		self.metric = metric
		self.k = k

	def fit(self, data):
		''' Explaning LOF:
		http://www.cse.ust.hk/~leichen/courses/comp5331/lectures/LOF_Example.pdf
		Use scikit-learn at:
		https://github.com/scikit-learn/scikit-learn/blob/0.21.X/sklearn/neighbors/lof.py
		'''
		super(LocalOutlierFactor, self).fit(data)
		from sklearn.neighbors import LocalOutlierFactor as LOF
		self.lof = LOF(n_neighbors=self.k, novelty=True, contamination=0.01,
					   metric=self.metric)
		self.lof.fit(self.data)

	def get_score(self, data):
		assert(hasattr(self, 'lof'))
		return - self.lof.score_samples(data) # Lower = less abnormal


class OneClassSVM(AnomalyDetector):
	def __init__(self):
		pass


class AutoEncoder(AnomalyDetector):
	pass