#from optparse import OptionParser
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model",default=None,type=str,dest="modelname",help="input the name which you want to creaete")



    #(options, args) = parser.parse_args()

    args = parser.parse_args()
    return args

