# Contributors
# 2602515   Taboka Dube
# 2541693   Wendy Maboa
# 2596852   Liam Brady
# 2333776   Refiloe Mopeloa

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import random
from collections import defaultdict, deque

class GridWorldMDP:
    def __init__(self, size=4):
        self.size = size
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.action_effects = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        self.obstacles = set()
        
        self.initial_state = (3, 0)
        self.goal_state = (0, 0)
        self.current_state = self.initial_state
    
    def reset(self):
        self.current_state = self.initial_state
        return self.current_state
    
    def is_valid_state(self, state):
        row, col = state
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if state in self.obstacles:
            return False
        return True
    
    def step(self, action):
        if self.current_state == self.goal_state:
            return self.current_state, 0, True
        
        row_change, col_change = self.action_effects[action]
        new_row = self.current_state[0] + row_change
        new_col = self.current_state[1] + col_change
        new_state = (new_row, new_col)
        
        if self.is_valid_state(new_state):
            self.current_state = new_state
        else:
            new_state = self.current_state
        
        if self.current_state == self.goal_state:
            reward = 20
            done = True
        else:
            reward = -1
            done = False
        
        return self.current_state, reward, done
    
    def get_all_states(self):
        states = []
        for row in range(self.size):
            for col in range(self.size):
                if self.is_valid_state((row, col)):
                    states.append((row, col))
        return states

def policy_evaluation_inplace(env, policy, gamma=1.0, theta=0.01, max_iterations=10000):
    V = defaultdict(float)
    V[env.goal_state] = 0
    iterations = 0
    
    while iterations < max_iterations:
        delta = 0
        iterations += 1
        
        for state in env.get_all_states():
            #skip goal state
            if state == env.goal_state:
                continue
                
            v_old = V[state]
            new_v = 0
            
            #uniform random policy
            for action in range(len(env.actions)):
                #simulate action
                row_change, col_change = env.action_effects[action]
                new_row = state[0] + row_change
                new_col = state[1] + col_change
                next_state = (new_row, new_col)
                
                #check if valid state
                if not env.is_valid_state(next_state):
                    next_state = state
                
                #reward (20 for goal -1 for entering state)
                if next_state == env.goal_state:
                    reward = 20 - 1
                else:
                    reward = -1
                
                #Bellman equation
                next_value = V[next_state]
                
                contribution = policy[state][action]*(reward + gamma*next_value)
                new_v += contribution
            
            #update value function in-place
            V[state] = new_v
            delta = max(delta, abs(v_old - new_v))
        
        #check for convergence
        if delta < theta:
            break
    
    return V, iterations

def policy_evaluation_two_array(env, policy, gamma=1.0, theta=0.01, max_iterations=10000):
    V_old = defaultdict(float)
    #set goal to be 0
    V_old[env.goal_state] = 0
    iterations = 0
    
    while iterations < max_iterations:
        V_new = defaultdict(float)
        V_new[env.goal_state] = 0
        delta = 0
        iterations += 1
        
        for state in env.get_all_states():
            #skip goal state
            if state == env.goal_state:
                continue
                
            new_v = 0
            
            #uniform random policy (each action)
            for action in range(len(env.actions)):
                #simulate the action
                row_change, col_change = env.action_effects[action]
                new_row = state[0] + row_change
                new_col = state[1] + col_change
                next_state = (new_row, new_col)
                
                #check if valid state
                if not env.is_valid_state(next_state):
                    next_state = state
                
                #reward
                if next_state == env.goal_state:
                    reward = 20 - 1
                else:
                    reward = -1
                
                #Bellman equation
                next_value = V_old[next_state]
                contribution = policy[state][action]*(reward + gamma*next_value)
                new_v += contribution
            
            #update value function
            V_new[state] = new_v
            delta = max(delta, abs(V_old[state] - new_v))
        
        #update old value function
        V_old = V_new.copy()
        
        #check for convergence
        if delta < theta:
            break
    
    return V_old, iterations

def run_experiments():
    env = GridWorldMDP(size=4)
    
    #uniform random policy
    policy = defaultdict(list)
    for state in env.get_all_states():
        if state == env.goal_state:
            policy[state] = [0, 0, 0, 0]  #no actions at goal state
        else:
            policy[state] = [0.25, 0.25, 0.25, 0.25]  #uniform random policy - each state has 1/4 chance to be picked
    
    #heatmap for gamma = 1
    V_inplace, iterations_1 = policy_evaluation_inplace(env, policy, gamma=0.9999999)
    
    #convert value function to 2D array for heatmap
    value_grid = np.zeros((env.size, env.size))
    for state in env.get_all_states():
        value_grid[state] = V_inplace[state]
    
    #create heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(value_grid, cmap='viridis')
    plt.colorbar(im, label='Value')
    plt.title('Value Function Heatmap (gamma approximately 1)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    for i in range(env.size):
        for j in range(env.size):
            text = plt.text(j, i, f'{value_grid[i, j]:.1f}', ha="center", va="center", color="w")
    
    plt.tight_layout()
    plt.savefig('heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    #generate discount rates
    gammas = np.logspace(-0.2, 0, num=20)
    
    inplace_iterations = []
    two_array_iterations = []
    
    for i, gamma in enumerate(gammas):
        #don't use values too close to 1
        if gamma > 0.999:
            gamma_adj = 0.999
        else:
            gamma_adj = gamma
            
        _, iter_inplace = policy_evaluation_inplace(env, policy, gamma=gamma_adj)
        _, iter_two_array = policy_evaluation_two_array(env, policy, gamma=gamma_adj)
        
        inplace_iterations.append(iter_inplace)
        two_array_iterations.append(iter_two_array)
    
    #create combined plot
    plt.figure(figsize=(10, 6))
    plt.plot(gammas, inplace_iterations, 'o-', label='In-place', linewidth=2)
    plt.plot(gammas, two_array_iterations, 's-', label='Two-array', linewidth=2)
    plt.xlabel('Discount Rate (gamma)')
    plt.ylabel('Iterations to Convergence')
    plt.title('Policy Evaluation Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('convergence_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nResults:",
          f"\nDiscount rates: {gammas}",
          f"\nIn-place iterations: {inplace_iterations}",
          f"\nTwo-array iterations: {two_array_iterations}")
    
    return inplace_iterations, two_array_iterations, gammas

if __name__ == "__main__":
    run_experiments()