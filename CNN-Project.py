from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import csv
import os
import string


testing = input("Type 'test' to run test cases, and anything else to run model training to retrain model: ")

if(testing=="test"):
    model = load_model('network95.h5')
    train_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory(
        'classes',
        class_mode = 'categorical')

    with open('output2.csv', 'w', newline='\n') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        spamwriter.writerow(['Input', 'Expected', 'Output'])
        directory = os.fsencode('test_set/kaggle_simpson_testset')
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            character = str(filename).split(".",1)[0]
            while character and character[-1].isalpha():
                character = character[:-1]
            test_image = image.load_img('test_set/kaggle_simpson_testset/' + str(filename), target_size= (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)
            for cls in training_set.class_indices:
                if(result[0][training_set.class_indices[cls]] == 1.0):
                    spamwriter.writerow([filename, character, cls])

else:
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    model.add(Flatten())

    model.add(Dense(units = 64, activation = 'relu'))

    model.add(Dense(units = 19, activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


    train_datagen = ImageDataGenerator(rescale = 1./255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory(
        'training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory('validation_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical')

    model.fit_generator(training_set,
    steps_per_epoch = 400,
    epochs = 80,
    validation_data = test_set,
    validation_steps = 200)

    model.save_weights("network.h5")
#model.summary()