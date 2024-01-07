import numpy as np
import random

# Euclidean distance
def distance_x_points(xP, points):
    if xP.size != points.shape[1]:
        return None
    numOfItems = points.shape[0]
    diffMat = np.tile(xP, (numOfItems, 1)) - points
    sqDiffMat = diffMat ** 2
    sumDiffMat = sqDiffMat.sum(axis=1)
    distances = sumDiffMat ** 0.5
    return distances


def make_target(x_data_set, y_data_set, num_samples, dimension):
    if x_data_set.shape[0] < num_samples or num_samples % 10 != 0:
        return None

    # prepare 1000 samples, 100 for each digit
    nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    counter = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    num_each = num_samples / 10
    points = np.zeros([num_samples, dimension])
    # np.random.seed(np.random.randint(0, 9999999))
    random.seed(random.randint(0, 9999999))
    count = 0
    while count < num_samples:
        # choose randomly from data set
        inc = random.randint(0, x_data_set.shape[0]-1)
        index = int(y_data_set[inc])
        if counter[index] < num_each:
            points[int(num_each * index + nums[index])] = x_data_set[inc, :dimension]
            nums[index] += 1
            count += 1
        counter[index] += 1
    return points


def make_data_set(rows, columns):
    np.random.seed(np.random.randint(0, 9999999))
    x_data_set = np.random.rand(rows, columns)
    y_data_set = np.random.randint(0, 10, rows)
    return x_data_set, y_data_set


def make_samples(x_data_set, points):
    if x_data_set.shape[1] != points.shape[1]:
        return None

    x_size = x_data_set.shape[0]
    num_samples = points.shape[0]
    input_size = x_data_set.shape[1]

    target_distances = np.zeros([x_size, num_samples])
    target_input = np.zeros([x_size, num_samples + input_size])

    # get target distances by calculating distance between each point in data set and the selected points
    for i in range(x_size):
        target_distances[i] = distance_x_points(x_data_set[i], points)

    # normalise target distances
    max_distance = np.amax(target_distances)
    target_distances = target_distances / max_distance * 0.99 + 0.01

    target_input[:, :num_samples] = target_distances
    target_input[:, num_samples:] = x_data_set

    return target_input


# mnist = tf.contrib.learn.datasets.load_dataset("mnist")
# x_train_dataset = mnist.train.images  # Returns np.array
# y_train_dataset = np.asarray(mnist.train.labels, dtype=np.int32)
# x_test_dataset = mnist.test.images  # Returns np.array
# y_test_dataset = np.asarray(mnist.test.labels, dtype=np.int32)

num_samples = 1000
dimension = 784
x_train_dataset, y_train_dataset = make_data_set(50000, dimension)
x_test_dataset, y_test_dataset = make_data_set(10000, dimension)
points = make_target(x_train_dataset, y_train_dataset, num_samples, dimension)
# x_axis = [2, 5, 10, 50, 100, 200, 400, 600, 784]
# y_axis = [10000, 20000, 30000, 40000, 50000]
x_axis = [2, 5, 10, 50, 100, 200]
y_axis = [10000, 20000, 30000, 40000, 50000]

print('preparing train data...')
for x in x_axis:
    print(x, end=': ')
    for y in y_axis:
        print(y, end=' ')
        target_input = make_samples(x_train_dataset[:y, :x], points[:, :x])
        np.savetxt('../../media2/GroupProjectData/train_'+str(x)+'_'+str(y)+'.csv', target_input, delimiter=",")
    print()

print('preparing test data...')
for x in x_axis:
    print(x)
    target_input = make_samples(x_test_dataset[:, :x], points[:, :x])
    np.savetxt('../../media2/GroupProjectData/test_'+str(x)+'_10000.csv', target_input, delimiter=",")

