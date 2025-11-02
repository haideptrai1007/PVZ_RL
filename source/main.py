from . import reinforcement

def main(speed, loops, update_frequency, checkpoint=None):
    pvz = reinforcement.PVZ_Reinforcement("./source/auto_actions.json")
    pvz.run(speed, loops, update_frequency, checkpoint)