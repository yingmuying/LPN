import matplotlib
matplotlib.use('Agg')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Model , Sequential
from keras.layers import Dropout,Input,Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from keras.utils.vis_utils import plot_model
from IPython.display import Image
import random
import string
import sys
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

characters = string.digits + string.ascii_uppercase
batch_size = 32
width, height, n_len ,n_class,channel= 80, 190, 7, len(characters),3
'''
Capcha Generator
First generate a license plate like captcha with width and height and length is just exactly same to Taiwan's law.
Define the size of X and Y.
Enable to keep asking for encoded X and y.
'''
def gen(batch_size=32):
    X = np.zeros((batch_size,height, width, channel), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            print('%d:'%(i+1)+random_str)
            X[i] = generator.generate_image(random_str)
            
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
#        X = np.reshape(X , (batch_size , width , height , 3))
        yield X, y

def gen1(batch_size=32):
    X = np.zeros((batch_size,width,height , channel), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=height, height=width)
    while True:
        for i in range(batch_size):
            
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
#            print('%d:'%(i+1)+random_str)
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


X, y = next(gen1(1))
plt.imshow(X[0])
plt.title(decode(y[0]))
plt.savefig('/home/wan/test.png')

'''
Build Model
'''
print('model building.....')
input_tensor = Input((width, height, channel))
x = input_tensor
for i in range(4):
    x = Convolution2D(32*2**i, (3, 3), activation="relu")(x)
    x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)

x = Flatten()(x)
x = Dropout(0.25)(x)
x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(n_len)]
model = Model(inputs=input_tensor , outputs=x)
print('model builded!!!!!!')
print('compiling.....')
model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
print('Compile complete!!!!!!!!!!!!!!!!!!!!!!!!!!')
plot_model(model, to_file="model.png", show_shapes=True)
#Image('model.png')

print('training start....')

model.fit_generator(generator=gen1(), 
                    steps_per_epoch=51200, 
                    epochs=5, 
                    workers=2, 
                    use_multiprocessing=False, 
                    validation_data=gen1(), 
                    validation_steps=1280)
print('training finished!!!!!!!!!!!!!!!!!!!!!!')
#X, y = next(gen(1))
#y_pred = model.predict(X)
