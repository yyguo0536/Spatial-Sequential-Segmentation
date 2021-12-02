import argparse
import os

parser = argparse.ArgumentParser(description='UNet+BDCLSTM for BraTS Dataset')

parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')

args = parser.parse_args()

print(args)
