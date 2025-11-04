from . import reinforcement

def main(sp, ep, fr, ckpt, save, lr, vc, ec):
    pvz = reinforcement.PVZ_Reinforcement("./source/auto_actions.json")
    pvz.run(sp, ep, fr, ckpt, save, lr, vc, ec)