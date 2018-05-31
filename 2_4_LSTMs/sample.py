'''
MIT License

Copyright (c) 2018 Udacity

'''

import argparse

from model import CharRNN, load_model, sample

parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('checkpoint', type=str, default=None,
                    help='initialize network from checkpoint')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='run the network on the GPU')
parser.add_argument('--num_samples', type=int, default=200,
                    help='number of samples for generating text')
parser.add_argument('--prime', type=str, default='From afar',
                    help='prime the network with characters for sampling')
parser.add_argument('--top_k', type=int, default=10,
                    help='sample from top K character probabilities')


args = parser.parse_args()

net = load_model(args.checkpoint)

print(sample(net, args.num_samples, cuda=args.gpu, top_k=args.top_k, prime=args.prime))