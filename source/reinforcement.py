from . import tool,  constants as c
from .state import mainmenu, screen, level

import json
import pygame as pg
import numpy as np

from . import PPO
import torch

class PVZ_Reinforcement():
    def __init__(self, filePath):
        self.Control = tool.Control()
        self.state_dict = {
            c.MAIN_MENU: mainmenu.Menu(),
            c.GAME_VICTORY: screen.GameVictoryScreen(),
            c.GAME_LOSE: screen.GameLoseScreen(),
            c.LEVEL: level.Level()
        }

        self.total_sun_reward = 0

        self.Control.setup_states(self.state_dict, c.LEVEL)

        self.plants_obs = np.ones(45)
        
        with open(filePath) as file:
            self.data = json.load(file)

    def reset(self):
        self.Control = tool.Control()
        self.state_dict = {
            c.MAIN_MENU: mainmenu.Menu(),
            c.GAME_VICTORY: screen.GameVictoryScreen(),
            c.GAME_LOSE: screen.GameLoseScreen(),
            c.LEVEL: level.Level()
        }
        self.Control.setup_states(self.state_dict, c.LEVEL)
    
    # Auto Collect Star
    def __handleStar(self):
        suns = self.Control.state.sun_group.sprites()
        for sun in suns:
            self.total_sun_reward += 1
            self.Control.event_loop((sun.rect.centerx, sun.rect.bottom))
            self.Control.update()

    # Auto Pick Cards, Play Game and Control Speed Game
    def initialize(self, speed):
        self.state_dict[c.LEVEL].update_speed(speed)

        init_actions = self.data["init_actions"]
        for act in init_actions:
            self.Control.event_loop(tuple(act))
            self.Control.update()
        pg.display.update()
        self.Control.clock.tick(self.Control.fps)

        self.state_dict[c.LEVEL].speed = speed
        self.Control.fps = 60*speed

    # Get Action Space
    def valid_action_space(self):
        menubar = self.Control.state.menubar
        cards = menubar.card_list
        sun = menubar.sun_value
        curr = menubar.current_time
        action_space = [int(c.canClick(sun, curr)) for c in cards]
        return action_space[:4]
    
    # Run an action
    def step(self, plantId, gridId: list):
        if int(plantId) <= 0:
            return
        plantId -= 1
        plants = self.data["plants"]
        actions = self.data["actions"]
        w = self.data["width"]
        h = self.data["height"]

        Ctrl = self.Control

        g1 = gridId % 4
        g2 = gridId // 4

        x = int(w[0] + (g1 + 0.5)*w[2]) 
        y = int(h[0] + (g2 + 0.5)*h[2])

        Ctrl.state.update(surface=Ctrl.screen, current_time=Ctrl.current_time, mouse_pos=tuple(actions[plantId]), mouse_click=[True, False])
        Ctrl.state.addPlant((x, y))

        if isinstance(Ctrl.state, level.Level):
            Ctrl.state.menubar.setCardFrozenTime(plants[plantId])
    
    def __checkZombie(self): 
        Ctrl = self.Control

        onPending = len(Ctrl.state.zombie_list)
        onMap = 0

        for i in Ctrl.state.zombie_groups:
            onMap += len(i)        
        return onPending + onMap
    
    def get_valid_grid(self, device):
        width = self.data["width"]

        plantsGrs = self.Control.state.plant_groups
        totalPlants = 0
        matrix = np.ones((5, 4))
        for i in range(len(plantsGrs)):
            for p in plantsGrs[i]:
                grid = (np.clip(p.rect.centerx, width[0], width[1]-15) - width[0]) // width[2]
                matrix[i, grid] = 0
                totalPlants += 1

        matrix = torch.from_numpy(matrix.flatten()).float().to(device)
        return matrix, totalPlants

    # Observation
    def grid_observe(self):
        zombies = self.data["zombies"] 
        plants = self.data["plants"]
        width = self.data["width"]

        Ctrl = self.Control
        zombie_obs = np.zeros((5, 9, 5), dtype=int)
        zombiesGrs = Ctrl.state.zombie_groups

        for i in range(len(zombiesGrs)):
            for zom in zombiesGrs[i]:
                idx = zombies.index(zom.name)
                grid = (np.clip(zom.rect.centerx, width[0], width[1]-15) - width[0]) // width[2]

                zombie_obs[i][grid][idx] += 1
        
        plants_obs = np.zeros((5, 9, 4), dtype=int)
        plantsGrs = Ctrl.state.plant_groups
        
        for i in range(len(plantsGrs)):
            for p in plantsGrs[i]:
                idx = plants.index(p.name.lower())
                grid = (np.clip(p.rect.centerx, width[0], width[1]-15) - width[0]) // width[2]
                plants_obs[i][grid][idx] = 1

        return np.concatenate((zombie_obs, plants_obs), axis=2)
    
    def totalObserve(self):
        obs = self.grid_observe()
        grid_state = torch.from_numpy(obs).type(torch.float).reshape(9,5,9)

        sun_val = self.Control.state.menubar.sun_value
        action_space = [1.0] + self.valid_action_space() + [sun_val]
        action_space = np.array(action_space, dtype=float)
        context_state = torch.from_numpy(action_space).float()

        return [grid_state, context_state]

    # Run
    def run(self, speed=1, loops=1, update_frequency=10, checkpoint=None, save=True, lr=1e-5, value_coef=0.8, entropy_coef=0.001):
        agent = PPO.PPOAgent(lr=lr, value_coef=value_coef, entropy_coef=entropy_coef)
        if checkpoint:
            agent.load(checkpoint)

        episode_rewards = []
        episode_lengths = []
        zombies_killed_history = []

        for episode in range(loops):
            total_win = 0
            self.reset()
            self.initialize(speed)

            game_state = level.Level
            Ctrl = self.Control

            currZom = len(Ctrl.state.zombie_list)
            reward = 0
            old_reward = 0

            first = True
            prev = None

            episode_reward = 0
            episode_length = 0
            episode_zombie_killed = 0

            start_time = 0
            currPlants = 0

            no_action_needed = False

            while not Ctrl.done and isinstance(Ctrl.state, game_state):
                Ctrl.update()
                if first:
                    curr_state = self.totalObserve()
                    first = False 


                if (pg.time.get_ticks()-start_time) >= (6000 / speed) and isinstance(Ctrl.state, game_state) and not no_action_needed:
                    start_time = pg.time.get_ticks()
                    gridMask, totalPlants = self.get_valid_grid(agent.device)
                    if torch.count_nonzero(gridMask) == 0:
                        no_action_needed = True
                        break

                    if (currPlants - totalPlants) < 0:
                        reward -= (currPlants + totalPlants)*0.01

                    newZom = self.__checkZombie()
                    if currZom > newZom:
                        zomkill = currZom - newZom
                        episode_zombie_killed += zomkill
                        reward += zomkill * 2
                        currZom = newZom


                    _, ctx_state = curr_state
                    if torch.count_nonzero(ctx_state[1:-1]) == 0:
                        continue
                    plant_action, grid_action = agent.select_action(curr_state, gridMask)


                    if (isinstance(Ctrl.state, game_state)):
                        self.step(plant_action, grid_action)
                        next_state = self.totalObserve()

                    curr_state = next_state
                    next_reward = reward - old_reward
                    old_reward = reward

                    if prev:
                        agent.store_reward_and_done(*prev)
                        prev = None
                    prev = [next_reward, False]

                    episode_reward += next_reward
                    episode_length += 1
                
                if isinstance(Ctrl.state, game_state):
                    self.__handleStar()

                Ctrl.event_loop()
                Ctrl.update()
                pg.display.update()
                Ctrl.clock.tick(self.Control.fps)
                
            if isinstance(Ctrl.state, screen.GameLoseScreen):
                agent.store_reward_and_done(prev[0], True)
                episode_reward -= 0
            else:
                agent.store_reward_and_done(prev[0] + 10, True)
                total_win += 1
                episode_reward += 10
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            zombies_killed_history.append(episode_zombie_killed)
        
            if (episode + 1) % update_frequency == 0:
                stats = agent.update()
                agent.entropy_coef *= 0.99
                    
                avg_reward = np.mean(episode_rewards[-update_frequency:])
                avg_length = np.mean(episode_lengths[-update_frequency:])
                avg_zombies = np.mean(zombies_killed_history[-update_frequency:])
                
                print(f"Episode {episode + 1}/{loops}")
                print(f"  Total Win: {total_win}")
                print(f"  Avg Reward (last {update_frequency}): {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Avg Zombies Killed: {avg_zombies:.1f}")
                print(f"  Policy Loss: {stats['policy_loss']:.4f}")
                print(f"  Value Loss: {stats['value_loss']:.4f}")
                print(f"  Entropy: {stats['entropy']:.4f}")
                print()
        if save:
            agent.save("pvz_ppo_trained.pth")
        print("\nTraining complete!")
        print(f"Model saved to 'pvz_ppo_trained.pth'")