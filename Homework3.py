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
    def __init__(self):
        self.size = 7
        self.actions = ['Up', 'Down', 'Left', 'Right']
        self.action_effects = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        self.obstacles = set()
        for col in range(6):
            self.obstacles.add((2, col))
        
        self.initial_state = (6, 0)
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
        
        if self.current_state == self.goal_state:
            reward = 20 - 1
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

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.num_actions = len(env.actions)
    
    def select_action(self, state):
        return random.choice(range(self.num_actions))
    
    def run_episode(self, max_steps=50):
        state = self.env.reset()
        trajectory = [state]
        total_reward = 0
        
        for step in range(max_steps):
            action = self.select_action(state)
            state, reward, done = self.env.step(action)
            trajectory.append(state)
            total_reward += reward
            
            if done:
                break
        
        return trajectory, total_reward

class GreedyAgent:
    def __init__(self, env, value_function):
        self.env = env
        self.value_function = value_function
    
    def select_action(self, state):
        if state == self.env.goal_state:
            return 0
        
        best_action = 0
        best_value = float('-inf')
        
        for action in range(len(self.env.actions)):
            row_change, col_change = self.env.action_effects[action]
            new_row = state[0] + row_change
            new_col = state[1] + col_change
            next_state = (new_row, new_col)
            
            if not self.env.is_valid_state(next_state):
                next_state = state
            
            if next_state == self.env.goal_state:
                reward = 20 - 1
            else:
                reward = -1
            
            expected_value = reward + self.value_function.get(next_state, float('-inf'))
            
            if expected_value > best_value:
                best_value = expected_value
                best_action = action
        
        return best_action
    
    def run_episode(self, max_steps=50):
        state = self.env.reset()
        trajectory = [state]
        total_reward = 0
        
        for step in range(max_steps):
            action = self.select_action(state)
            state, reward, done = self.env.step(action)
            trajectory.append(state)
            total_reward += reward
            
            if done:
                break
        
        return trajectory, total_reward

def compute_optimal_value_function(env):
    value_function = {}
    distances = {}
    queue = deque([(env.goal_state, 0)])
    distances[env.goal_state] = 0
    visited = {env.goal_state}
    
    while queue:
        current_state, current_dist = queue.popleft()
        
        for action in range(len(env.actions)):
            row_change, col_change = env.action_effects[action]
            prev_row = current_state[0] - row_change
            prev_col = current_state[1] - col_change
            prev_state = (prev_row, prev_col)
            
            if (env.is_valid_state(prev_state) and prev_state not in visited):
                distances[prev_state] = current_dist + 1
                visited.add(prev_state)
                queue.append((prev_state, current_dist + 1))
    
    for state in env.get_all_states():
        if state in distances:
            optimal_steps = distances[state]
            value_function[state] = 20 - optimal_steps
        else:
            value_function[state] = float('-inf')
    
    return value_function

def run_experiments(num_runs=20):
    random.seed(42)
    np.random.seed(42)
    
    env = GridWorldMDP()
    optimal_values = compute_optimal_value_function(env)
    
    random_agent = RandomAgent(env)
    greedy_agent = GreedyAgent(env, optimal_values)
    
    random_returns = []
    greedy_returns = []
    sample_random_trajectory = None
    sample_greedy_trajectory = None
    
    for run in range(num_runs):
        trajectory, total_return = random_agent.run_episode()
        random_returns.append(total_return)
        if run == 0:
            sample_random_trajectory = trajectory
        
        trajectory, total_return = greedy_agent.run_episode()
        greedy_returns.append(total_return)
        if run == 0:
            sample_greedy_trajectory = trajectory
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    agents = ['Random Agent', 'Greedy Agent']
    means = [np.mean(random_returns), np.mean(greedy_returns)]
    stds = [np.std(random_returns), np.std(greedy_returns)]
    
    bars = ax1.bar(agents, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('Average Return')
    ax1.set_title(f'Agent Performance Comparison (Averaged over {num_runs} runs)')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('agent_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 7))
    
    grid1 = np.zeros((env.size, env.size))
    for obs in env.obstacles:
        grid1[obs] = -1
    grid1[env.goal_state] = 2
    grid1[env.initial_state] = 1
    
    colors = ['white', 'lightblue', 'lightgreen', 'black']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    ax2.imshow(grid1, cmap=cmap, vmin=-1, vmax=2)
    
    if sample_random_trajectory:
        for i in range(len(sample_random_trajectory) - 1):
            r1, c1 = sample_random_trajectory[i]
            r2, c2 = sample_random_trajectory[i + 1]
            if (r1, c1) != (r2, c2):
                ax2.arrow(c1, r1, c2-c1, r2-r1, head_width=0.1, head_length=0.1, 
                        fc='red', ec='red', alpha=0.7, linewidth=2)
    
    ax2.set_title(f'Random Agent (Steps: {len(sample_random_trajectory)-1}, Return: {random_returns[0]})')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(env.size))
    ax2.set_yticks(range(env.size))
    
    grid2 = np.zeros((env.size, env.size))
    for obs in env.obstacles:
        grid2[obs] = -1
    grid2[env.goal_state] = 2
    grid2[env.initial_state] = 1
    
    ax3.imshow(grid2, cmap=cmap, vmin=-1, vmax=2)
    
    if sample_greedy_trajectory:
        for i in range(len(sample_greedy_trajectory) - 1):
            r1, c1 = sample_greedy_trajectory[i]
            r2, c2 = sample_greedy_trajectory[i + 1]
            if (r1, c1) != (r2, c2):
                ax3.arrow(c1, r1, c2-c1, r2-r1, head_width=0.1, head_length=0.1, 
                        fc='red', ec='red', alpha=0.7, linewidth=2)
    
    ax3.set_title(f'Greedy Agent (Steps: {len(sample_greedy_trajectory)-1}, Return: {greedy_returns[0]})')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(env.size))
    ax3.set_yticks(range(env.size))
    
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Start'),
        plt.Rectangle((0,0),1,1, facecolor='lightgreen', label='Goal'),
        plt.Rectangle((0,0),1,1, facecolor='black', label='Obstacle'),
        plt.Rectangle((0,0),1,1, facecolor='white', edgecolor='black', label='Empty'),
        plt.Line2D([0], [0], color='red', lw=2, label='Trajectory')
    ]
    fig2.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=5)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_experiments(num_runs=20)