from numpy import *
import operator
from os import listdir
import time
import tensorflow as tf
import sys
from keras.models import Sequential
set_printoptions(threshold=sys.maxsize)
import pickle
import heapq

# file that hold parameters for NN version
filename='parameters/ann_params2.sav'
model_p = Sequential()
model_p = pickle.load(open(filename, 'rb'))


# distance between two points
def distance_ed(x, y):
    return ((x - y)**2).sum()


def build_kdtree(points, dim, depth=0):
    axis = depth % dim
    numberOfItems = len(points)
    middle_point = numberOfItems >> 1

    if numberOfItems > 1:
        sorted_points = sorted(points, key=lambda point: point[axis])
        return array([
            build_kdtree(sorted_points[:middle_point], dim, depth+1),
            build_kdtree(sorted_points[middle_point+1:], dim, depth+1),
            sorted_points[middle_point]
            ]
        )
    elif numberOfItems == 1:
        return array([None, None, points[0]])


# use kd-tree for finding k-nearest neighbours
def get_knn_kdtree(pointX, points, k, dim, depth=0, heap=None):
    is_root = not heap
    if is_root:
        heap = []
    if points is not None:
        dist = distance_ed(pointX, points[2])
        if len(heap) < k:
            heapq.heappush(heap, (-dist, array2string(points[2])))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, array2string(points[2])))

        axis = depth % dim

        if pointX[axis] > points[2][axis]:
            side = 1  # right side of tree
        else:
            side = 0  # left side of tree

        get_knn_kdtree(pointX, points[side], k, dim, depth + 1, heap)
        #   compare distance of given point and best with the closest point of the given point
        if distance_ed(pointX, -heap[0][0]) > abs((pointX[axis] - points[2][axis])):
            get_knn_kdtree(pointX, points[(side + 1) % 2], k, dim, depth + 1, heap)
    if is_root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors


def classify_Cosine(inX, dataset, labels, k):   # Classifier: Cosine distance formula for finding kNN
    # xyDot = dot(inX, transpose(dataset))
    xyDot = dot(dataset, inX)
    xxDot = dot(inX, inX);
    yyDot = (dataset*dataset).sum(axis=1)
    xxSqrt = sqrt(xxDot)
    yySqrt = sqrt(yyDot)
    distances = 1 - (xyDot/(xxSqrt * yySqrt))
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def classify_ED(inX, dataset, labels, k):   # Classifier: Euclidean distance formula for finding kNN
    numOfItems = dataset.shape[0]
    diffMat = tile(inX, (numOfItems, 1)) - dataset
    sqDiffMat = diffMat**2
    sumDiffMat = sqDiffMat.sum(axis=1)
    distances = sumDiffMat**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def classifier_KDtree(pointX, trainingSet, trainingLabels, k):	# Classifier: Kd-tree for finding kNN
    dim = trainingSet.shape[1]
    kd_data = build_kdtree(trainingSet, dim)
    k_nearest = get_knn_kdtree(pointX, kd_data, k, dim)
    class_count = {}
    for i in range(k):
        s = k_nearest[i][1]
        x = fromstring(s[1:-1],  dtype=float, sep=' ')
        # print(where(all(abs(trainingSet - x) < 0.0000001, axis=1)))
        index = where(all(abs(trainingSet - x) < 0.0000001, axis=1))[0][0]
        voteIlabel = trainingLabels[index]
        class_count[voteIlabel] = class_count.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def classify_NN(inX, labels, k):		# Classifier: Neural network for finding kNN
    if inX.ndim == 1:
        x_test1 = array([inX])
    sortedDistIndicies = model_p.predict(x_test1)[0].argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = int(sortedDistIndicies[i]/100)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def img2vector(filename):   # convert 32x32 image to 1x1024 vector
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def imgsInDir2mat(filesDir):    # convert all files in the specified directory to a matrix (file is 32x32 which convert to 1x1024 vector)
    hwLabels = []
    fileList = listdir(filesDir)    # list all files in the specified folder
    m = len(fileList)
    imgMat = zeros((m,1024))
    for i in range(m):
        fileName = fileList[i]
        fileStr = fileName.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        imgMat[i, :] = img2vector(filesDir + '/' + fileName)
    return imgMat, hwLabels


def handwritingClassTest():
    # First training dataset
    # traingSet, traingLabels = imgsInDir2mat('trainingDigits')
    # testSet, testLabels = imgsInDir2mat('testDigits')
    # Second training dataset
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    traingSet = mnist.train.images  # Returns np.array
    traingLabels = asarray(mnist.train.labels, dtype=int32)
    testSet = mnist.test.images  # Returns np.array
    testLabels = asarray(mnist.test.labels, dtype=int32)
    m = testSet.shape[0]
    errorCount = 0.0
    for i in range(m):
        # traingResult = classify_Cosine(testSet[i, :], traingSet, traingLabels, 3)
        # traingResult = classify_ED(testSet[i, :], traingSet, traingLabels, 3)
        # traingResult = classifier_KDtree(testSet[i, :], traingSet, traingLabels, 3)
        traingResult = classify_NN(testSet[i, :], traingLabels, 5)
        print("the classifier came back with: %d, the real answer is: %d, finished tests, %.1f%s" % (traingResult, testLabels[i], i/m*100, "%"))
        if testLabels[i] != traingResult:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(m)))
    print("\nthe accuracy is: %f" % (1-(errorCount / float(m))))


start_time = time.time()
handwritingClassTest()
print("--- %s seconds ---" % (time.time() - start_time))



