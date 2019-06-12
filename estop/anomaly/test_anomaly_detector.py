import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 

def testNearestNeighbor(metric='mahalanobis'):
	from anomaly_detector import KNearestNeighbor
	detector = KNearestNeighbor(metric=metric, k=2)
	train_data = np.concatenate((
		np.random.randn(20, 2) * 8 - 20,
		np.random.randn(20, 2) * 8 + 20))
	test_data = np.random.randn(10000, 2) * 40
	detector.record(train_data)
	test_distance = detector.getDistance(test_data)
	test = plt.scatter(test_data[:,0], test_data[:,1], c=test_distance, cmap='inferno')
	train = plt.scatter(train_data[:,0], train_data[:,1], color='black')
	plt.colorbar(test)
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

if __name__ == '__main__':
	testNearestNeighbor()