import matplotlib
matplotlib.use('Agg')
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model , Sequential
from keras.layers import Dropout,Input,Dense,Activation,Convolution2D,MaxPooling2D,Flatten
import random
import string

characters = string.digits + string.ascii_uppercase
batch_size = 32
width, height, n_len ,n_class= 190, 80, 7, len(characters)
'''
Capcha Generator
First generate a license plate like captcha with width and height and length is just exactly same to Taiwan's law.
Define the size of X and Y.
Enable to keep asking for encoded X and y.
'''
def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        X = np.reshape(X , (batch_size , width , height , 3))
        yield X, y



'''
decoder is for changing between one-hot and character
'''
def decode(y):
    y = np.argmax(np.array(y), axis=1)[:]
    return ''.join([characters[x] for x in y])

'''
Build Model
'''

input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Convolution2D(32*2**i, (3, 3), activation="relu")(x)
    x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
model = Model(inputs=input_tensor , outputs=x)

model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])

model.fit_generator(generator=gen(), 
                    steps_per_epoch=51200, 
                    epochs=5, 
                    workers=2, 
                    pickle_safe=True, 
                    validation_data=gen(), 
                    validation_steps=1280)

#X, y = next(gen(1))
#y_pred = model.predict(X)
