import argparse
import subprocess
from datetime import datetime


def get_args(jupyter=False, args=""):
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, nargs="+", default=[0])
    parser.add_argument("--data_dir", type=str, default="../data/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iters_to_accumulate", type=int, default=1)
    parser.add_argument("--B", type=int, default=256)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--H", type=int, default=200)
    parser.add_argument("--s_dim", type=int, default=64)
    parser.add_argument("--a_dim", type=int, default=6)
    parser.add_argument("--o_dim", type=int, default=17)
    parser.add_argument("--h_dim", type=int, default=128)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--load_epoch", type=int, default=None)

    if not jupyter:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args.split())

    args.ghash = subprocess.check_output(
        "git rev-parse --short HEAD".split()).strip().decode('utf-8')

    if args.timestamp is None:
        args.timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

    args.shuffle = not args.no_shuffle

    return args