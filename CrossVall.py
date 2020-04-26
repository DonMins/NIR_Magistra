import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import sklearn.model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from keras.utils import to_categorical
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def CrossValNeyro(matrix):
    row = matrix.shape[0]
    cols = matrix.shape[1]
    errors = 0
    success = 0
    for i in range(1,row):
        X_test = np.concatenate(( matrix[i,0:cols-1],  matrix[i,0:cols-1])).reshape(2,30)
        y_test = np.concatenate((matrix[i,cols-1],matrix[i,cols-1]), axis=None)
        y_test =np.array((np.uint8(y_test)))

        X_train =np.concatenate((matrix[0:i, 0:cols-1] , matrix[i + 1:row, 0:cols-1]), axis=0)
        y_train = np.concatenate((matrix[0:i, cols-1],matrix[i + 1:row, cols-1]), axis=0)
        y_train =np.array((np.uint8(y_train)))
        y_train = to_categorical(y_train)

        np.random.seed(123)
        model = Sequential()
        model.add(Dense(2000, input_dim=30, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation='softmax'))

        # Configure the model and start training
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        model.fit(X_train , y_train, epochs=1, batch_size=5, verbose=2)
        test_predictions = model.predict(X_test)
        if ( np.argmax(test_predictions[0]) == y_test[0] ):
            success = success + 1
            print("success = ", success)
        else:
            errors = errors + 1
            print("errors = ", errors)


    print("Error = " , errors)
    print("Success = " , success)


def CrossVal(matrix,model):
    row = matrix.shape[0]
    cols = matrix.shape[1]
    errors = 0
    success = 0
    for i in range(0,row):
        X_test = np.concatenate(( matrix[i,0:cols-1],  matrix[i,0:cols-1])).reshape(2,30)
        y_test = np.concatenate((matrix[i,cols-1],matrix[i,cols-1]), axis=None)
        y_test =np.array((np.uint8(y_test)))

        X_train =np.concatenate((matrix[0:i, 0:cols-1] , matrix[i + 1:row, 0:cols-1]), axis=0)
        y_train = np.concatenate((matrix[0:i, cols-1],matrix[i + 1:row, cols-1]), axis=0)
        y_train =np.array((np.uint8(y_train)))


        model.fit(X_train , y_train)
        test_predictions = model.predict(X_test)
        if (test_predictions[0] == y_test[0] ):
            success = success + 1
        else:
            errors = errors + 1


    print("Error = " , errors)
    print("Success = " , success)
    print("% " , (success*100/row))


if __name__ == '__main__':

    path1 = "Признаки\\3\\" + "allVall.txt"
    matrix = np.array(np.loadtxt(path1, delimiter=" "))
    # clf = RandomForestClassifier(random_state=1, n_estimators=100)
    # clf = KNeighborsClassifier(n_neighbors=11)
    clf = SVC( gamma='auto')


    CrossVal(matrix,clf)

