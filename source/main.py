from . import reinforcement

def main(speed=1):
    pvz = reinforcement.PVZ_Reinforcement("./source/auto_actions.json")
    pvz.run(speed)