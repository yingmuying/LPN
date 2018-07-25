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
import datetime
import opt
from optparse import OptionParser
from erode_dilate import *
from tqdm import tqdm
import time
from keras.layers.normalization import BatchNormalization
#import keras.losses
'''
parameters
'''
#parser = OptionParser()
characters = string.digits + string.ascii_uppercase +'*'
'''
remove illegel characters including 'I','O','4' in Taiwan.
'''
#print(characters)
n_class = len(characters)

width , height =  247 , 107     
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
def LPNgen(random):
    #n = GetRandNum(number)
    #e_n = GetRandNum(english_num,False)
    #print(n,e_n)
    blankimg = Create_blank(57 * n_len + (n_len-1) * interval , 107)

    img = CombineImage(blankimg,random,"characters")
    img = Addhat(img,interval)
    (h,w) = img.shape[:2]
    if opts.erode:
        img = Erode(img , 1)
    if opts.dilate:
        img = Dilate(img , 1)
    resize_img = cv2.resize(img, (width ,height), interpolation=cv2.INTER_AREA)
    ran_str = ''.join(random)
    
    if(opts.testing==True):
        cv2.imwrite("./save_image/" + ran_str+ ".jpg" , resize_img)
    #print(resize_img.shape)
    return resize_img
    
'''
generator of model
'''

def gen(batch_size=32):
    random_len = randint(4 , 7)
    X = np.zeros((batch_size , width, height, 3), dtype=np.uint8)
    #y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(7)]
    y = np.zeros((batch_size , n_len), dtype=np.uint8)
    #print("generating data.....")
    #print (len(y))
    
    while True:
        random_len = randint(4 , 7)
        for i in range(batch_size):
            random_str = GetRandCha(random_len)
            #print(random_str)
            for k in range(n_len - random_len):
                random_str.insert(randint(0,random_len+k-1),'*')

            X[i] = np.array(LPNgen(random_str)).transpose(1, 0, 2)
            y[i] = [characters.find(x) for x in random_str]
                
        yield [X, y, np.ones(batch_size)*int(conv_shape[1]-2), 
                               np.ones(batch_size)*n_len], np.ones(batch_size)

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
'''
model building
'''
input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Conv2D(32, (3, 3), activation="relu")(x)
    x = Conv2D(32, (3, 3), activation="relu")(x)
    #BatchNormalization()
    x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)
x = Dense(32, activation='relu')(x)
gru_1 = GRU(opts.rnn_size, return_sequences=True, kernel_initializer="he_normal", name="gru1")(x)
gru_1b = GRU(opts.rnn_size, go_backwards=True, kernel_initializer="he_normal", name="gru1_b", return_sequences=True)(x)
gru1_merged = add([gru_1, gru_1b])
gru_2 = GRU(opts.rnn_size, return_sequences=True, kernel_initializer="he_normal", name="gru2")(gru1_merged)
gru_2b = GRU(opts.rnn_size, go_backwards=True, kernel_initializer="he_normal", 
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
    #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='sgd')
else:
    model = load_model(opts.modelname ,custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
    base_model = load_model("base_" +opts.modelname)

'''
print structure of model
'''

if (opts.printmodel):
    plot_model(model, to_file="model.png", show_shapes=True)
    Image('model.png')

if (opts.testing == False):
    model.fit_generator(gen(opts.batch_size), steps_per_epoch=opts.steps, epochs=opts.epochs,
            callbacks=[EarlyStopping(patience=10), evaluator],
            validation_data=gen(), validation_steps=1280)
else:
    #start = time.time()
    w , c = 0 , 0
    for i in range(20):
        print("testing......")
        print
        characters2 = characters + ' '
        [X_test, y_test, _, _], _  = next(gen(1))
        #cv2.imwrite("./save_image/test.jpg" , X_test)
        #print("shape of image:"),
        #print(X_test.shape)
        #X_test[0] = cv2.imread('ScreenShot.png').transpose(1, 0, 2)
        #cv2.imwrite('testingoutput.jpg', X_test[0])
        #print("shape of answer array:"),
        #print(y_test.shape)
        start = time.time()
        y_pred = base_model.predict(X_test)
        y_pred = y_pred[:,2:,:]
        out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :n_len]
        out = ''.join([characters[x] for x in out[0]])
        y_true = ''.join([characters[x] for x in y_test[0]])
        print("prediction-"),
        print(out),
        print("-")
        print("answer-"),
        print(y_true),
        print("-")
        
        if (out == y_true):
            print("correct prediction!")
            c=c+1
        else:
            print("Wrong....")
            w=w+1
    end = time.time()
    elapsed = end - start
    print("correction : %d"%c)
    print("wring : %d"%w)
    print "Testing time take: ", elapsed, "seconds."



if(opts.modelname == None and opts.testing == False):
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    model.save(run_name+".h5")
    base_model.save("base_"+run_name+".h5")
    print("training finish! New model saved.")

elif(opts.modelname==None and opts.testing ==True):
    print("Please input testing model name")

elif(opts.testing == True):
    print("testing finish!")

else:
    model.save(opts.modelname)
    base_model.save("base_"+opts.modelname)
    print("training finish!Old model updated")

del model
