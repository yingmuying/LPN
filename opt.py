#from optparse import OptionParser
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",default=None,type=str,dest="modelname",
            help="input the name which you want to creaete")
    parser.add_argument('--epochs',default=5 , type=int,dest = "epochs",
            help = "input the epoch number you want to train")
    parser.add_argument('--steps' , default=400 , type=int,dest = "steps" , 
            help = "num of step per epoch")
    parser.add_argument('--rnnsize' , default = 128 , type = int , dest = "rnn_size",
            help = "rnn size")
    parser.add_argument('--batchsize' , default = 128 , type=int , dest = "batch_size",
            help = "batch size")
    parser.add_argument('-e' , '--erode' , default = False , type = bool , dest = "erode",
            help = "erode")
    parser.add_argument('-d' , '--dilate' , default = False , type = bool , dest = "dilate",
            help = "dilate")
    parser.add_argument('--printmodel' , default = False , type = bool , dest = "printmodel",
            help = "print model")

    parser.add_argument('--testing' , default = False , type = bool , dest = "testing",
            help = "train or test")
    parser.add_argument('--testimage' , default = False , type = bool , dest = "testimage",
            help = "name of the image you want to test")


    args = parser.parse_args()
    return args

