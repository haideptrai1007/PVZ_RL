import argparse
import pygame as pg
from source.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sp', type=int, default=50)
    parser.add_argument('--ep', type=int, default=20000)
    parser.add_argument('--fr', type=int, default=250)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--vc", type=float, default=0.5)
    parser.add_argument("--ec", type=float, default=0.01)
    args = parser.parse_args()
    main(**vars(args))
    pg.quit()

