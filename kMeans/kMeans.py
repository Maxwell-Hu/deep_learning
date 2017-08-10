def loadDataSet(filename):
	dataMat = []
	fr = open(filename)
	for line in fr.readlines():
		curLine = line.stripe.split('\n')
		fltLine = map(float, curLine)
		dataMat.append(fltLine)
	return dataMat

def distEclud(vecA, vecB):
	return np.sqrt(np.sum(np.power(vecA-vecB, 2)))

def randCent(dataSet, k):
	n = np.shape(dataSet)[1]
	centroids = np.mat(zeros((k,n)))
	for j in range(n):
		minJ = np.min(dataSet[:,j])
		rangeJ = float(np.max(dataSet[:,j]) - minJ)
		centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
	return centroids
