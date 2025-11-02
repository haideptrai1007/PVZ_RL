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
        return action_space
    
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

        g1 = gridId % 9
        g2 = gridId // 9

        x = int(w[0] + (g1 + 0.5)*w[2]) 
        y = int(h[0] + (g2 + 0.5)*h[2])

        Ctrl.state.update(surface=Ctrl.screen, current_time=Ctrl.current_time, mouse_pos=tuple(actions[plantId]), mouse_click=[True, False])
        Ctrl.state.addPlant((x, y))

        Ctrl.state.menubar.setCardFrozenTime(plants[plantId])

    
    def __checkZombie(self): 
        Ctrl = self.Control

        onPending = len(Ctrl.state.zombie_list)
        onMap = 0

        for i in Ctrl.state.zombie_groups:
            onMap += len(i)        
        return onPending + onMap
    
    def __get_valid_grid(self):
        width = self.data["width"]

        plantsGrs = self.Control.state.plant_groups
        matrix = np.ones((5, 9))
        for i in range(len(plantsGrs)):
            for p in plantsGrs[i]:
                grid = (np.clip(p.rect.centerx, width[0], width[1]-15) - width[0]) // width[2]
                matrix[i, grid] = 0
        
        matrix.flatten()

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
        
        plants_obs = np.zeros((5, 9, 8), dtype=int)
        plantsGrs = Ctrl.state.plant_groups
        
        for i in range(len(plantsGrs)):
            for p in plantsGrs[i]:
                idx = plants.index(p.name.lower())
                grid = (np.clip(p.rect.centerx, width[0], width[1]-15) - width[0]) // width[2]
                plants_obs[i][grid][idx] = 1

        return np.concatenate((zombie_obs, plants_obs), axis=2)
    
    def totalObserve(self):
        obs = self.grid_observe()
        grid_state = torch.from_numpy(obs).type(torch.float).reshape(13,5,9)

        sun_val = self.Control.state.menubar.sun_value
        action_space = [1] + self.valid_action_space() + [sun_val]
        context_state = torch.tensor(action_space, dtype=torch.float)

        return [grid_state, context_state]

    # Run
    def run(self, speed=1, loops=1):
        agent = PPO.PPOAgent()

        episode_rewards = []
        episode_lengths = []
        zombies_killed_history = []

        for _ in range(loops):
            self.Control.setup_states(self.state_dict, c.LEVEL)
            self.initialize(speed)

            game_state = level.Level
            Ctrl = self.Control

            currZom = len(Ctrl.state.zombie_list)
            currSun = 0
            count = 0
            reward = 0
            old_reward = 0

            first = True
            prev = None

            episode_reward = 0
            episode_length = 0

            while not Ctrl.done and isinstance(Ctrl.state, game_state):
                Ctrl.update()
                if first:
                    curr_state = self.totalObserve()
                    first = False 


                if (pg.time.get_ticks()) >= (1500 / speed) * count: # Underdeveloped
                    count += 1
                    if prev:
                        agent.store_reward_and_done(*prev)
                        prev = None

                    gridMask = self.__get_valid_grid()

                    # Reward
                    if self.total_sun_reward > currSun:
                        reward += self.total_sun_reward - currSun
                        currSun = self.total_sun_reward

                    newZom = self.__checkZombie()
                    if currZom > newZom:
                        reward += (currZom - newZom)*5
                        currZom = newZom

                    plant_action, grid_action = agent.select_action(curr_state, gridMask)
       
                    self.step(plant_action, grid_action)

                    curr_state = self.totalObserve()
                    print(curr_state[1])
                    reward = reward - old_reward
                    prev = [reward, False]

                    episode_reward += reward
                    episode_length += 1






                    

      



                elif (isinstance(Ctrl.state, game_state)):
                    self.__handleStar()

                Ctrl.event_loop()
                Ctrl.update()
                pg.display.update()
                Ctrl.clock.tick(self.Control.fps)
                
            if isinstance(Ctrl.state, screen.GameLoseScreen):
                agent.store_reward_and_done(prev[0] - 100, True)
                episode_reward -= 100
            else:
                agent.store_reward_and_done(prev[0] + 100, True)
                episode_reward += 100
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)