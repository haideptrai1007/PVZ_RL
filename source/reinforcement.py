from . import tool, constants as c
from .state import mainmenu, screen, level

import json
import pygame as pg
import numpy as np

class PVZ_Reinforcement():
    def __init__(self, filePath):
        self.Control = tool.Control()
        self.state_dict = {
            c.MAIN_MENU: mainmenu.Menu(),
            c.GAME_VICTORY: screen.GameVictoryScreen(),
            c.GAME_LOSE: screen.GameLoseScreen(),
            c.LEVEL: level.Level()
        }

        self.Control.setup_states(self.state_dict, c.LEVEL)

        self.plants_obs = np.zeros((5, 9, 8))
        
        with open(filePath) as file:
            self.data = json.load(file)

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

    # Auto Collect Star
    def __handleStar(self):
        suns = self.Control.state.sun_group.sprites()
        for sun in suns:
            self.Control.event_loop((sun.rect.centerx, sun.rect.bottom))
    
    # Get Action Space
    def valid_action_space(self):
        menubar = self.Control.state.menubar
        cards = menubar.card_list
        sun = menubar.sun_value
        curr = menubar.current_time
        action_space = [int(c.canClick(sun, curr)) for c in cards]
        return action_space
    
    def step(self, plantId, gridId):
        pass

            
    
    # Observation
    def observe(self):
        zombies = self.data["zombies"] 
        width = self.data["width"]

        Ctrl = self.Control
        zombie_obs = np.zeros((5, 9, 5), dtype=int)
        zombiesGrs = Ctrl.state.zombie_groups

        for i in range(len(zombiesGrs)):
            for zom in zombiesGrs[i]:
                idx = zombies.index(zom.name)
                grid = (np.clip(zom.rect.centerx, width[0], width[1]-1) - width[0]) // width[2]

                zombie_obs[i][grid][idx] += 1

        return np.concatenate((zombie_obs, self.plants_obs), axis=2)

    def run(self, speed=1):
        self.initialize(speed)

        game_state = level.Level
        Ctrl = self.Control

        time = 0
        while not Ctrl.done and isinstance(Ctrl.state, game_state):
            if time % 1080:
                self.valid_action_space()
            time += 1
            
            self.__handleStar()
            self.Control.event_loop()
            self.Control.update()
            pg.display.update()
            self.Control.clock.tick(self.Control.fps)


        if isinstance(Ctrl.state, screen.GameLoseScreen):
            print("Game Over")
        else:
            print("Game Win")