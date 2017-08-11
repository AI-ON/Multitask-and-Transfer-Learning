import argparse
import torch
import torch.nn as nn
import numpy as np

from models import beta_vae


def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)





def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Do some reinforcement learning')
    parser.add_argument('--seed', type=int, default=1,
                        metavar='SEED')
    return parser.parsee_args()

if __name__ == '__main__':
    main()
