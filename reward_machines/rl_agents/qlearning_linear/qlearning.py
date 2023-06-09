#Adapted from:
#https://github.com/dennybritz/reinforcement-learning/blob/master/FA/Q-Learning%20with%20Value%20Function%20Approximation%20Solution.ipynb
#https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/rl_agents/deepq/deepq.py

import numpy as np

class Estimator():
    """
    Linear action-value function approximator. 
    """
    
    def __init__(self, env):    
        
        self.env = env         
        self.action_space = env.action_space
        self.num_state_features = len(env._get_info) #NOTE: Calls internal function 
        
        #will be redefining features for each state,action pair 
        self.num_state_action_features = len(self.action_space)*num_state_features  
        
        #Initialize feature weights (Best way for features to be initialized?)
        stdv = 1. / numpy.sqrt(self.num_state_action_features+1)
        self.model = np.random.uniform(-stdv,stdv,(self.num_state_action_features+1,1))
 
            
    def featurize_state(self, info, env_action):
        """
        Returns the featurized representation for a state. Assumes "info" gives feature values. 
        Features defined as function of state AND ACTIONS (because esitmating action-value fcn). 
        """
        #initialize features with constant term added for scalar adjustment 
        featurized = np.zeros(self.num_state_action_features+1) 
        featurized[-1]=1 
     
        #actions represented by an integer (should start at 0 and only increase, therefore can index using) 
        featurized[env_action:env_action+self.num_state_features] = info.values()
        
        return featurized
    
    def predict(self, info, a=None):
        """
        Makes value function predictions.
        
        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for
            
        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a list of predictions for all actions
            in the environment where pred[i] is the prediction for action i.
            
        """
        if not a: 
            return [np.dot(np.transpose(featurize_state(info,a)),self.model) for a in self.action_space]
        else:
            features = self.featurize_state(info,a)
            return np.dot(np.transpose(features),self.model) 
    
    def update(self, info, a, y, lr):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(info,a)
        
        #take one step of size learning rate (loss fcn = 1/2 * l2 norm of target & actual) 
        model = model + lr*(y - np.dot(np.transpose(features), model))*features
        

def get_epsilon_greedy_action(estimator, epsilon):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.
    
    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array.
    
    """
    
    A = np.ones(len(env.action_space), dtype=float) * epsilon / len(estimator.action_space)
    q_values = estimator.predict(info)
    best_action = np.argmax(q_values)
    A[best_action] += (1.0 - epsilon) 
    return A
        
def q_learning_linear(env, total_timesteps=500000, lr=1e-4, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0, use_crm=True, print_freq=100):
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
    #initialize q function approximations for each state in the RM
    models = {}
    for rm_state in env.reward_machines[0].U:  #assuming only 1 RM we are learning a policy for 
        models[rm_state] = Estimator(env)
       
    episode_rewards = []
    obs,info = env.reset()
    
    for t in total_timesteps: #assume that can't start in terminal state? try to take step w/o checking if done 

        _, rm_state = obs["rm-features"] #assuming single rm 
        env_action = get_epsilon_greedy_action(models[rm_state],info) #assuming not in terminal state 
        reset = False 
        new_obs, r, done, new_info = env.step(env_action) #includes generating counterfactual experiences 
        
        if use_crm: 
            experiences = info["crm_experience"] #returns a list of counterfactual experiences generated per each RM state  
        else: 
            experiences = [(new_obs, r, done, new_info)]
            
        #take step to update for the state-action value fcn of each state
        for _obs, _action, _r, _new_obs, _done in experiences:
            _, _rm_state = _obs["rm-features"] #this is STARTING RM state
            _, _new_rm_state = _new_obs["rm-features"] #this is ENDING RM state (different if transitioned) 
            if _done: 
                target = _r #no future rewards to add 
            else: 
                #update estimator with received reward and estimated future reward (from estimator of ending RM state) 
                predicted_next_q = models[_new_rm_state].predict(new_info) #think calculating with new info is correct... 
                target = _r + discount_factor* np.max(predicted_next_q) #take greedy action (off-policy not following e-greedy)
            
            #think info correct here (not changing between rm states), not new_info because updating initial state before step 
            models[_rm_state].update(info, env_action, target, lr) 
            
        obs, info = new_obs, new_info
        episode_rewards[-1] += r #Add to reward of current episode (at end of list)
        
        if done: 
            obs,info = env.reset()
            episode_rewards.append(0) #Add new episode to end of list
        
        #Print for debugging purposes  
        if t%print_freq ==0:
            num_episodes = len(episode_rewards) 
            print("Step {}/{} @ Episode {} ({})".format(t, total_timesteps, num_episodes, episode_rewards[-1]))
        
        
        
        