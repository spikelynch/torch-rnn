#!/usr/bin/env python

import torchrnn
import sys, re, codecs, os
import argparse, json


LENGTH = 200000

TAGS = True


parser = argparse.ArgumentParser() 
parser.add_argument("-o", "--output",  type=str, default='./', help="output directory")
parser.add_argument('-l', '--length', type=int, default=LENGTH, help="Length of output")
                       
args = parser.parse_args()


opts = {
	'-length': args.length
}


for lookahead in [ l * 10 for l in range(1, 20) ]:
	opts['-excavate'] = lookahead
	o = "{}/sample{}/".format(args.output, lookahead)
	if not os.path.exists(o):
		os.mkdir(o)
	opts['-outdir'] = o
	print(o)
	torchrnn.run_generic('excavate.lua', opts)

