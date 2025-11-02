from . import reinforcement

def main(speed, loops):
    pvz = reinforcement.PVZ_Reinforcement("./source/auto_actions.json")
    pvz.run(speed, loops)