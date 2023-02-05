import sys

# --------------------- file load ---------------------- #
from source.utils.common import *

from mode.train import training
from mode.val import validing
from mode.test import testing


def main():
    args = parse_args()
    if args.mode == "train":
        training(args.param, args.gpus, args.checkpoint, args.dataset, args.depth, args.n_class)
    elif args.mode == "valid":
        validing()
    elif args.mode == "test":
        testing()
    else:
        raise ValueError("mode is incorrecr")

if __name__ == "__main__":
    main()