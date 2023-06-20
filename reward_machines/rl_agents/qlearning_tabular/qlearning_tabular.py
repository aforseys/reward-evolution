#Adapted from:
#https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb
#https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/rl_agents/deepq/deepq.py

import numpy as np
import random
# import matplotlib.pyplot as plt
# from matplotlib import colors

# def visualize(Q):

#     data = np.random.rand(10, 10) * 20

#     # create discrete colormap
#     cmap = colors.ListedColormap(['red', 'blue'])
#     bounds = [0,10,20]
#     norm = colors.BoundaryNorm(bounds, cmap.N)

#     fig, ax = plt.subplots()
#     ax.imshow(data, cmap=cmap, norm=norm)

#     # draw gridlines
#     ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
#     ax.set_xticks(np.arange(0, 8, 1));
#     ax.set_yticks(np.arange(0, 8, 1));

#     plt.show()

def get_best_action(Q,s, actions):
    qmax = max(Q[s].values())
    best_actions = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best_actions)
        
def q_learning_tabular(env, total_timesteps=500000, lr=1e-4, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, use_crm=True, print_freq=100, q_init=2):
    """
    Q-Learning algorithm for off-policy TD control using linear function approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    
    Args:
        env: ----- RMEnv (Wrapper?) 
        total_timesteps: Number of timesteps to train for. 
      #  estimator: Action-Value function estimator
      #  num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    #initialize environment and rewards 
    episode_rewards = [0]
    obs,info = env.reset()
    env_obs, rm_obs = obs 
     
    actions = list(range(env.action_space.n))
    
    Q = {}
    
    for t in range(1,total_timesteps+1): #assume that can't start in terminal state? try to take step w/o checking if done 
        
        #if hazard locations, account for more than one and turn all to tuples (so can be part of dict index) 
        #consider changing whole thing so only agent location indexes 
        env_state = []
        for key,val in rm_obs["features"].items():
            if key == "h_loc":
                env_state.append(tuple([tuple(loc) for loc in val]))
            else:
                env_state.append(tuple(val))
        env_state = tuple(env_state) 
            
        #env_state = tuple([tuple(val) for val in list(rm_obs["features"].values())])
        rm_state = tuple(rm_obs["rm-state"])
        
        #problem: hazard is now a list of locations, where those locations are np arrays,
        #therefore turning list into tuple doesn't get rid of the np array hazard locations 
        
        if (env_state, rm_state) not in Q: Q[(env_state, rm_state)] = dict([(a,q_init) for a in actions])
            
        action = random.choice(actions) if random.random() < epsilon else get_best_action(Q, (env_state,rm_state), actions)
    
        new_obs, r, done, new_info = env.step(action) #includes generating counterfactual experiences  
    
#         if np.all(new_obs[1]["features"]["a_loc"] == np.array([4,6])):
#             print("action selected", action)
#             print(type(action))
#             print("Reached")
#             print(new_obs)
#             print(r)
#             print(done)
#             print(new_info["crm-experience"][0][-1])
        
        if use_crm: 
            experiences = new_info["crm-experience"] #returns a list of counterfactual experiences generated per each RM state 
#             print("length of experiences", experiences)
        else: 
            experiences = [(obs, action, r, new_obs, done)]
            
        #take step to update for the state-action value fcn of each state
        for _obs, _action, _r, _new_obs, _done in experiences:
            _env_obs, _rm_obs = _obs
            #_env_state = tuple(_env_obs["features"].values())
            #_rm_state = tuple(_rm_obs["rm-state"]) #this is STARTING RM state
            
            #messy, get rid of, has to change hazard locs into tuples  
            _env_state = []
            for key,val in _rm_obs["features"].items():
                if key == "h_loc":
                    _env_state.append(tuple([tuple(loc) for loc in val]))
                else:
                    _env_state.append(tuple(val))
            _env_state = tuple(_env_state) 

            _rm_state = tuple(_rm_obs["rm-state"])
            
            _new_env_obs, _new_rm_obs = _new_obs
#             _new_env_state = tuple(_new_env_obs["features"].values())
#             _new_rm_state = tuple(_new_rm_obs["rm-state"]) #this is ENDING RM state (different if transitioned) 
            
            #messy, get rid of, has to change hazard locs into tuples  
            _new_env_state = []
            for key,val in _new_rm_obs["features"].items():
                if key == "h_loc":
                    _new_env_state.append(tuple([tuple(loc) for loc in val]))
                else:
                    _new_env_state.append(tuple(val))
                    
            _new_env_state = tuple(_new_env_state) 
            _new_rm_state = tuple(_new_rm_obs["rm-state"])
            
            if (_env_state, _rm_state) not in Q: Q[(_env_state, _rm_state)]  = dict([(a,q_init) for a in actions])
                #models[_rm_state] = Estimator(env,info)#specific info not important, just need length
                
            if _done: 
                target = _r #no future rewards to add 
            else: 
                if (_new_env_state, _new_rm_state) not in Q: Q[(_new_env_state, _new_rm_state)] = dict([(a,q_init) for a in actions])
                   # models[_new_rm_state] = Estimator(env,info)#specific info not important, just need length
                
                #update estimator with received reward and estimated future reward (from estimator of ending RM state) 
                predicted_next_q = Q[(_new_env_state, _new_rm_state)][get_best_action(Q, (_new_env_state,_new_rm_state), actions)] #UPDATE 
                target = _r + discount_factor* predicted_next_q #take greedy action (off-policy not following e-greedy)
            
            #think info correct here (not changing between rm states), not new_info because updating initial state before step 
            Q[(_env_state,_rm_state)][action] += lr* (target - Q[(_env_state, _rm_state)][action])
            
        obs, info = new_obs, new_info 
        env_obs, rm_obs = new_obs 
        episode_rewards[-1] += r #Add to reward of current episode (at end of list)
        
        if done: 
            obs,info = env.reset()
            env_obs, rm_obs = obs 
            episode_rewards.append(0) #Add new episode to end of list
        
        #Print for debugging purposes  
        if t%print_freq ==0:
            num_episodes = len(episode_rewards) 
#             print("The model:",Q)
#             print("Last action taken", action)
            print("Step {}/{} @ Episode {} ({})".format(t, total_timesteps, num_episodes, episode_rewards[-1]))
            
            #visualize grid
            
   
    return Q
        