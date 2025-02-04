import time
import numpy as np
from mushroom_rl.core import Core


class ChallengeCore(Core):
    def __init__(self, agent=None, mdp=None, *args, **kwargs):
        if mdp is None:
            return

        super().__init__(agent, mdp, *args, **kwargs)
        self.prev_action = None

    def run(self, render=True):
        while True:
            if render:
                self.mdp.render()

            action = None

            state, reward, done, _ = self.mdp.step(action)

            if done:
                print("simulation ended")
                self.mdp.reset()