import csv
import getopt
import sys
from datetime import datetime

import numpy as np

from BICO.BICO import BICO
from BICO.Point import Point


def run_bico(n, d, size, p, file):
    bico = BICO(d, n, 5, p, size)
    tstart = datetime.now()
    no = 1
    with open(file, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            del row[-1]
            # print len(row)
            p = Point(np.array(map(float, row)))
            bico.insert_point(p)
            no += 1
            if no % 1000 == 0:
                print "Read point number " + str(no)
    tend = datetime.now()
    print bico.num_cfs
    print tend - tstart
    for t in bico.time:
        print t
    f = open('coreset.out', 'w')
    bico.output_coreset(f)
    f.close()


if __name__ == '__main__':
    n = 0
    d = 0
    size = 0
    p = 0
    file = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "n:d:p:s:f:")
        print opts
        print args
    except getopt.GetoptError:
        print "csv_input.py -n <number of points> -d <dimensions> -s <coreset size> -p <projections> -f <input file>"
        sys.exit(2)
    for opt, arg in opts:
        print str(opt) + " " + str(arg)
        if opt == '-n':
            n = int(arg)
        elif opt == '-d':
            d = int(arg)
        elif opt == '-s':
            size = int(arg)
        elif opt == '-p':
            p = int(arg)
        elif opt == '-f':
            file = arg
    if len(opts) != 5:
        print "Need all arguments! Got " + str(len(opts))
        print "csv_input.py -n <number of points> -d <dimensions> -s <coreset size> -p <projections> -f <input file>"
        sys.exit(2)
    run_bico(n, d, size, p, file)
