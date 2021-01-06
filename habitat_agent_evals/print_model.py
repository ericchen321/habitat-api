import argparse
import collections

import torch

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)

args = parser.parse_args()

ckpt = torch.load(args.path, map_location="cpu")
for k in ckpt:
    print(k)

if "config" in ckpt:
    print(ckpt["config"])
