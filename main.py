import numpy as np
import random as rd

class Grid:
    def __init__(self, size):
        self.size = size
        self.grid = [[0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0]]
        self.threshold = 0.01
        self.discount = 1
        self.goal = [0,0]
        self.start = [3,0]
        self.current_position = self.start
        self.possible_moves = []
        self.reward = 0
        self.terminate = False
        self.V = {}
        for i in range(size):
            for j in range(size):
                self.V[i,j] = 1
        self.V[self.goal[0],self.goal[1]] = 0
        
        def get_possible_moves(current_position):
            possible_moves = []
            # north
            if (current_position[0] - 1 >= 0):
                self.possible_moves.append("N")
            # east
            if (current_position[1] + 1 <= 3):
                self.possible_moves.append("E")
            # south
            if (current_position[0] + 1 <= 3):
                self.possible_moves.append("S")
            # west
            if (current_position[0] - 1 >= 0):
                self.possible_moves.append("W")
            return possible_moves
            
        def move():
            possible_moves = get_possible_moves(self.current_position)
            current_move = self.possible_moves[rd.randint(0,len(self.possible_moves)-1)]
            match current_move:
                case "N":
                    self.current_position[0] -= 1
                case "E":
                    self.current_position[1] += 1
                case "S":
                    self.current_position[0] += 1
                case "W":
                    self.current_position[1] -= 1
            self.reward -= 1
            if current_move == self.goal:
                self.reward += 20
                self.terminate = True
            
        def compute_V(s):
            pass
        def bellman(reward, s0, s1):
            v = self.V[s0]
            sum_actions = 0
            sum_rewards = 0
            possible_moves = get_possible_moves(s0)
            for a in possible_moves:
                
            # self.V[s0] = 

            
     