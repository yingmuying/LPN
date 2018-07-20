#from optparse import OptionParser
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",default=None,type=str,dest="modelname",
            help="input the name which you want to creaete")
    parser.add_argument('--epochs',default=1 , type=int,dest = "epochs",
            help = "input the epoch number you want to train")
    parser.add_argument('--steps' , default=51200 , type=int,dest = "steps" , 
            help = "num of step per epoch")

    #(options, args) = parser.parse_args()

    args = parser.parse_args()
    return args

