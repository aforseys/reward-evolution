#Adapted from Gurobi examples and from 
#https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/envs/grids/grid_environment.py

import gymnasium 
from gymnasium import spaces
import pygame
import numpy as np
from reward_machines.reward_machines.rm_environment import RewardMachineEnv

class GridWorldEnv(gymnasium.Env):
   # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4} 

    def __init__(self, size=8):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "a_loc": spaces.Box(0, size - 1, shape=(2,), dtype=int), #agent location
                "B_loc": spaces.Box(0, size - 1, shape=(2,), dtype=int), #target locations
                "C_loc": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "h_loc": spaces.Box(0, size - 1, shape=(2,), dtype=int) #hazard location
            }
        ) 
        
        #We have 3 propositions, corresponding to if either target or the hazard has been reached
        self.propositions = {"B": False, "C": False, "h": False}

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)
        self._obstacle_locations = [np.array([3,2]), np.array([3,3]), np.array([4,2]), np.array([4,1])]

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

#         assert render_mode is None or render_mode in self.metadata["render_modes"]
#         self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"a_loc": self._agent_location, "B_loc": self._target_locations[0], "C_loc": self._target_locations[1], "h_loc": self._hazard_location}
    
    def get_events(self):
        
        #If agent at either target, update propositions 
        #In this environment once a proposition holds, no turning off 
        if np.array_equal(self._agent_location, self._target_locations[0]):
            self.propositions["B"]=True 
        if np.array_equal(self._agent_location, self._target_locations[1]):
            self.propositions["C"]=True
        if np.array_equal(self._agent_location, self._hazard_location):
            self.propositions["h"]=True

        #Return string of true propositions (string is how they represent and evaluate props)
        return "".join([prop for prop in self.propositions if self.propositions[prop]]) 

    def _get_info(self):
        #feature set, only using distance to hazard 
        return {
            "distance_B": np.linalg.norm(
                self._agent_location - self._target_locations[0], ord=1)}
#            ),
#             "distance_C": np.linalg.norm(
#                 self._agent_location - self._target_locations[1], ord=1
#             ),
#             "distance_hazard": np.linalg.norm(
#                 self._agent_location - self._hazard_location, ord=1)
#         } 
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #set target locations
        self._target_locations = [np.array([4,6]),np.array([2,2])]
        
        #set hazard location
        self._hazard_location = np.array([1,4])
        
        #set agent location (for now initialized same every time)
        self._agent_location = np.array([0,0])
        
        #need to reset propositions at start 
        self.propositions = {"B": False, "C": False, "h": False}

        # Choose the agent's location uniformly at random 
        # such that it doesn't interfere with targets 
#         self._agent_location = self._target_locations[0]
#         while np.sum([np.array_equal(self._agent_location, t) for t in self._target_locations]) >= 1:
#         #(np.array_equal(self.target1_location, self.agent_location) or np.array_equal(self.target2_locaiton, self.agent_location)):
#             self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

       # propositions = self._get_props()
        observation = self._get_obs()
        info = self._get_info()

#         if self.render_mode == "human":
#             self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        # Move agent to state as long as not obstacle
        if not np.any(np.all(self._agent_location+direction == self._obstacle_locations, axis=1)):
            
            # We use `np.clip` to make sure we don't leave the grid
            self._agent_location = np.clip(
                self._agent_location+direction, 0, self.size - 1
            )
        
        #Make observation
        observation = self._get_obs()

        # Get required feature info (distance to hazard) 
        info = self._get_info()
        
        # # Receive reward based on RM state
        reward = 0 #all reward comes from RM 
        terminated = False 
        
        # terminated = False 
        # if propositions["h"]:
        #     terminated=True
        #     reward = -100
        # elif propositions["B"] and propositions["C"]:
        #     terminated = True 
        #     reward = 100 
        # elif propositions["B"]: #already reached reward 1
        #     reward = -1
        # else: #haven't reached either target 
        #     reward = -1 + -20/(info["distance_hazard"] + 0.001) #add small buffer to prevent division by zero 
                
     #   terminated = np.array_equal(self._agent_location, self._target_location[0])
     #   reward = 1 if terminated else 0  # Binary sparse rewards
        
       
#         if self.render_mode == "human":
#             self._render_frame()

        return observation, reward, terminated, info
        #return observation, reward, terminated, False, info #not sure why the False was here, not good reason online 

#     def render(self):
#         if self.render_mode == "rgb_array":
#             return self._render_frame()

#     def _render_frame(self):
#         if self.window is None and self.render_mode == "human":
#             pygame.init()
#             pygame.display.init()
#             self.window = pygame.display.set_mode((self.window_size, self.window_size))
#         if self.clock is None and self.render_mode == "human":
#             self.clock = pygame.time.Clock()

#         canvas = pygame.Surface((self.window_size, self.window_size))
#         canvas.fill((255, 255, 255))
#         pix_square_size = (
#             self.window_size / self.size
#         )  # The size of a single grid square in pixels

#         # First we draw the targets
#         pygame.draw.rect(
#             canvas,
#             (255, 0, 255),
#             pygame.Rect(
#                 pix_square_size * self._target_locations[0],
#                 (pix_square_size, pix_square_size),
#             ),
#         )
#         pygame.draw.rect(
#             canvas,
#             (0, 255, 0),
#             pygame.Rect(
#                 pix_square_size * self._target_locations[1],
#                 (pix_square_size, pix_square_size),
#             ),
#         )
        
#         #Now draw hazard
#         pygame.draw.rect(
#             canvas,
#             (255, 0, 0),
#             pygame.Rect(
#                 pix_square_size * self._hazard_location,
#                 (pix_square_size, pix_square_size),
#             ),
#         )
        
#         # Now we draw the agent
#         pygame.draw.circle(
#             canvas,
#             (0, 0, 255),
#             (self._agent_location + 0.5) * pix_square_size,
#             pix_square_size / 3,
#         )
        
#         #Now we draw the obstacles
#         for o in self._obstacle_locations: 
#             pygame.draw.rect(
#                 canvas,
#                 (128, 128, 128),
#                 pygame.Rect(
#                     pix_square_size * o,
#                     (pix_square_size, pix_square_size),
#                 ), 
#             )

#         # Finally, add some gridlines
#         for x in range(self.size + 1):
#             pygame.draw.line(
#                 canvas,
#                 0,
#                 (0, pix_square_size * x),
#                 (self.window_size, pix_square_size * x),
#                 width=3,
#             )
#             pygame.draw.line(
#                 canvas,
#                 0,
#                 (pix_square_size * x, 0),
#                 (pix_square_size * x, self.window_size),
#                 width=3,
#             )

#         if self.render_mode == "human":
#             # The following line copies our drawings from `canvas` to the visible window
#             self.window.blit(canvas, canvas.get_rect())
#             pygame.event.pump()
#             pygame.display.update()

#             # We need to ensure that human-rendering occurs at the predefined framerate.
#             # The following line will automatically add a delay to keep the framerate stable.
#             self.clock.tick(self.metadata["render_fps"])
#         else:  # rgb_array
#             return np.transpose(
#                 np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
#             )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class GridWorldRMEnv(RewardMachineEnv):
    def __init__(self, env, rm_files):
        super().__init__(env, rm_files)

    def render(self, mode='human'):
        if mode == 'human':
            # commands
            str_to_action = {"w":3,"d":0,"s":1,"a":2} #could be wrong, double check 

            # play the game!
            done = True
            while True:
                if done:
                    print("New episode --------------------------------")
                    obs, info = self.reset()
                    print("Current task:", self.rm_files[self.current_rm_id])
                    self.env.show()
                    print("Features:", obs)
                    print("RM state:", self.current_u_id)
                    print("Events:", self.env.get_events())

                print("\nAction? (WASD keys or q to quite) ", end="")
                a = input()
                print()
                if a == 'q':
                    break
                # Executing action
                if a in str_to_action:
                    obs, rew, done, _ = self.step(str_to_action[a])
                    self.env.show()
                    print("Features:", obs)
                    print("Reward:", rew)
                    print("RM state:", self.current_u_id)
                    print("Events:", self.env.get_events())
                else:
                    print("Forbidden action")
        else:
            raise NotImplementedError           
            
class ObstacleRMEnv(GridWorldRMEnv):
    def __init__(self, rm_files):
       # rm_files = ["./reward_machines/envs/grids/reward_machines/rm1.txt"]
        env = GridWorldEnv()
        super().__init__(env,rm_files)
            