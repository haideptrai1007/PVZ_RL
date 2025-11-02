import argparse
import pygame as pg
from source.main import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--frequency', type=int, default=10)
    parser.add_argument('--ckpt', type=int, default=None)
    args = parser.parse_args()

    main(args.speed, args.epochs, args.frequency, args.ckpt)
    pg.quit()
