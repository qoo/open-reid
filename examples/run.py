from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys


from reid import datasets

raw_dir='~/albumpair'
raw_dir = osp.expanduser(raw_dir)
dataset = datasets.create('albumpair', raw_dir)
