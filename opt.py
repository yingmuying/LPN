from optparse import OptionParser

parser = OptionParser()

parser.add_option("-n","--name",dest="name",help="input the name which you want to creaete",default=None)



(options, args) = parser.parse_args()
