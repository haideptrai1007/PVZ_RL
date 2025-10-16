__author__ = 'marble_xu'

from . import reinforcement

def main():
    pvz = reinforcement.PVZ_Reinforcement("./source/auto_actions.json")
    pvz.run(10)
