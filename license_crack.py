import matplotlib
matplotlib.use('Agg')
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
import random
import string

characters = string.digits + string.ascii_uppercase
batch_size = 32

'''
I want to first generate a license plate like captcha with width and height and length is just exactly same to Taiwan's law.
Then,I define the size of X and Y

'''

def captcha_Generator():
    
    width, height, n_len ,n_class= 190, 80, 7, len(characters)
    X = np.zeros((height, width, 3), dtype=np.uint8)
    #I don't know the meaning of 3 here.
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
encoder and decoder is two function converting between one-hot and number
'''

def encoder(number):
    return to_categorical(number)

def decoder(onehot):
    return np.argmax(onehot)




#for i in range(batch_size):
#    X , y=next(captcha_Generator())
