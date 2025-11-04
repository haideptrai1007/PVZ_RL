import argparse
import pygame as pg
from source.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sp', type=int, default=20)
    parser.add_argument('--ep', type=int, default=1000)
    parser.add_argument('--fr', type=int, default=10)
    parser.add_argument('--ckpt', type=int, default=None)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vc", type=float, default=0.8)
    parser.add_argument("--ec", type=float, default=0.05)
    args = parser.parse_args()
    main(**vars(args))
    pg.quit()

