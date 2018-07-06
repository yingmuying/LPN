import matplotlib
matplotlib.use('Agg')
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
import random
import string

characters = string.digits + string.ascii_uppercase
batch_size = 32

'''
Capcha Generator
First generate a license plate like captcha with width and height and length is just exactly same to Taiwan's law.
Define the size of X and Y.
Enable to keep asking for encoded X and y.
'''

def cap_Gen():
    
    width, height, n_len ,n_class= 190, 80, 7, len(characters)
    X = np.zeros((height, width, 3), dtype=np.uint8)
    #I don't know the meaning of 3 here. maybe channel?
    y = [np.zeros((n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        random_str = ''.join([random.choice(characters) for j in range(n_len)])
        X = generator.generate_image(random_str)
        #Here comes a enumerate for lood which can output j as index and enumerate ch in list
        for j, ch in enumerate(random_str):
            y[j][:] = 0
            y[j][characters.find(ch)] = 1
        plt.imshow(X)
        plt.title(random_str)
        plt.savefig('/home/wan/png/'+random_str+'.png')
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

model = Sequential()
for i in range(4)
    model.add(Convolution2D(32*2**i, kernel_size = 3, activation='relu'))
    model.add(Convolution2D(32*2**i, kernel_size = 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(dropout(0.25))

