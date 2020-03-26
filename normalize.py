from sklearn import preprocessing
import numpy as np
pathSave = "Признаки\\2\\" + str("allVall") + ".txt"
X = np.loadtxt(pathSave, delimiter = " ")
normalized_X = preprocessing.normalize( X[:,0: len(X[0,:])-1])
pathSave = "Признаки\\2\\" + str("allVallNorm") + ".txt"

X[:,0: len(X[0,:])-1] = normalized_X
fileName = open(pathSave, 'w')
np.savetxt(fileName, X, fmt="%f")
fileName.close()



