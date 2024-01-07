from keras.layers import Dense
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt



def train(train_file, test_file, input_size, output_size):
    data_set_train = np.loadtxt(train_file, delimiter=',', skiprows=1)
    data_set_test = np.loadtxt(test_file, delimiter=',', skiprows=1)

    x_train = data_set_train[:, output_size:]
    y_train = data_set_train[:, :output_size]
    x_test = data_set_test[:, output_size:]
    y_test = data_set_test[:, :output_size]

    model = Sequential()

    model.add(Dense(units=754, activation='relu', input_shape=(input_size,)))
    model.add(Dense(units=2500, activation='relu', input_shape=(input_size,)))
    model.add(Dense(units=2000, activation='relu', input_shape=(input_size,)))
    model.add(Dense(units=output_size, activation='sigmoid'))

    model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=128, epochs=15, verbose=True, validation_split=.1)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy for ' + str(input_size) + 'D with ' + str(data_set_train.shape[0]+1) + ' sample outputs')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.figure()

    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.3}')

    return accuracy


x_axis = [2, 5, 10, 50, 100, 200]
y_axis = [10000, 20000, 30000, 40000, 50000]
z_axis = np.zeros((len(y_axis), len(x_axis)))

for y in range(len(y_axis)):
    for x in range(len(x_axis)):
        print('dimension: ', x_axis[x], ', dataset: ', y_axis[y])
        file_train = '../../media2/GroupProjectData/train_' + str(x_axis[x]) + '_' + str(y_axis[y]) + '.csv'
        file_test = '../../media2/GroupProjectData/test_' + str(x_axis[x]) + '_10000.csv'
        acc = train(file_train, file_test, x_axis[x], 1000)
        z_axis[y][x] = acc


np.savetxt('../../media2/GroupProjectData/accuracy_values.txt', z_axis)
plt.show()

X, Y = np.meshgrid(x_axis, y_axis)
Z = z_axis

fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', linewidths=0.5)
ax.set_xlabel('dimension')
ax.set_ylabel('dataset size')
ax.set_zlabel('accuracy')

plt.show()
