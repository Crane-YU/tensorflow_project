from tensorflow import keras

num_classes = 10
img_rows, img_cols = 28, 28

(trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()

if keras.backend.image_data_format() == 'channels_first':
    trainX = trainX.reshape(trainX.shape[0], 1, img_rows, img_cols)
    testX = testX.reshape(testX.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols,  1)
    testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.
testX /= 255.

trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])

model.fit(trainX, trainY, batch_size=128, epochs=20, validation_data=(testX, testY))

score = model.evaluate(testX, testY)
print("Test loss:", score[0])
print("Test accuracy:", score[1])