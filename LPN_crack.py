#-*- coding: utf-8 -*-
import matplotlib
from create_num import *
import sys
matplotlib.use('Agg')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import backend as K
import numpy as np
from keras.layers.convolutional import Convolution2D , Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Reshape, Lambda
from keras.layers.merge import add, concatenate 
from keras.models import Model
from keras.models import *
from keras.layers import *
from keras.utils.vis_utils import plot_model
from IPython.display import Image
from keras.layers.recurrent import GRU
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.callbacks import *
import keras.callbacks
import string , random
import opt
from optparse import OptionParser
from erode_dilate import *
from tqdm import tqdm
import keras.losses

'''
parameters
'''
parser = OptionParser()
characters = string.digits + string.ascii_uppercase
'''
remove illegel characters including 'I','O','4' in Taiwan.
'''
characters = characters.replace('I' , '')
characters = characters.replace('O' , '')
characters = characters.replace('4' , '')
#print(characters)
n_class = len(characters)

width , height =  247 , 107     
rnn_size = 128                  
n_len = 7                       #length of Taiwan License Number

opts = opt.parse_opt()          #parameters
print("Using model name:"),
print(opts.modelname)

'''
our loss function
'''

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
'''
image generator
'''
def LBNgen(n , e_n):
    n = GetRandNum(number)
    e_n = GetRandNum(english_num,False)
    #print("Generating image.....")
    #print(n,e_n)
    blankimg = Create_blank(57 * number + (number-1) * interval , 107)
    blankimg_en = Create_blank(57 * english_num + (english_num-1) * interval , 107)

    img = CombineImage(blankimg,n,"digits")
    img_en = CombineImage(blankimg_en,e_n,"english")
    #cv2.imwrite("./img.jpg" , img)
    #cv2.imwrite("./img_en" , img_en)
    combine_img = CombineTwoImage(img,img_en)
    
    combine_img = Addhat(combine_img,interval)
    (h,w) = combine_img.shape[:2]
    #print(w,w * resize_zoom ,h, h*resize_zoom)
    #cv2.imwrite("./comb.jpg" , combine_img)

    resize_img = cv2.resize(combine_img, (width ,height), interpolation=cv2.INTER_AREA)
    
    #cv2.imwrite("./resize.jpg" , resize_img)
    #print(resize_img.shape)
    return resize_img
    #cv2.waitKey(0)
    
'''
generator of model
'''

def gen(batch_size=32):
    X = np.zeros((batch_size , width, height, 3), dtype=np.uint8)
    #y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(7)]
    y = np.zeros((batch_size , n_len), dtype=np.uint8)
    #print("generating data.....")
    #print (len(y))
    
    while True:
        for i in range(batch_size):
            #print ("i = "),
            #print (i)
            n = GetRandNum(number)
            e_n = GetRandNum(english_num,False)
            random_str = n+e_n
            #X[i] = LBNgen(n , e_n)
            X[i] = np.array(LBNgen(n , e_n)).transpose(1, 0, 2)
            y[i] = [characters.find(x) for x in random_str]
            #print (y[i])
            
        #yield X , y
        yield [X, y, np.ones(batch_size)*int(conv_shape[1]-2), 
                               np.ones(batch_size)*n_len], np.ones(batch_size)
#X , y = next(gen(1))
#print (y)

def evaluate(model , batch_num=10):
    batch_acc = 0
    generator = gen()
    for i in range(batch_num):
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        ctc_decode = K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :n_len]
        if out.shape[1] == n_len:
            batch_acc += ((y_test == out).sum(axis=1) == n_len).mean()
    return batch_acc / batch_num

class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.accs = []
    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model)*100
        self.accs.append(acc)
        print
        print ('acc: %f%%'%acc)

evaluator = Evaluate()


input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

gru_1 = GRU(128, return_sequences=True, kernel_initializer="he_normal", name="gru1")(x)
gru_1b = GRU(128, go_backwards=True, kernel_initializer="he_normal", name="gru1_b", return_sequences=True)(x)

gru1_merged = add([gru_1, gru_1b])

gru_2 = GRU(128, return_sequences=True, kernel_initializer="he_normal", name="gru2")(gru1_merged)
gru_2b = GRU(128, go_backwards=True, kernel_initializer="he_normal", 
        name="gru2_b", return_sequences=True)(gru1_merged)

x = concatenate([gru_2, gru_2b])


x = Dropout(0.25)(x)
x = Dense(n_class+1, activation="softmax", kernel_initializer="he_normal")(x)
base_model = Model(inputs=input_tensor, outputs=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), 
                                  name='ctc')([x, labels, input_length, label_length])
if(opts.modelname == None):
    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out]) 
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
else:
    model = load_model(opts.modelname ,custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
    

plot_model(model, to_file="model.png", show_shapes=True)
Image('model.png')

model.fit_generator(gen(), steps_per_epoch=51200, epochs=1,
                            callbacks=[EarlyStopping(patience=10), evaluator],
                                                validation_data=gen(), validation_steps=1280)
#plot(model, to_file="model.png", show_shapes=True)
#Image('model.png')
if(opts.modelname == None):
    model.save("my_model.h5")
else:
    model.save(opts.modelname)
del model
