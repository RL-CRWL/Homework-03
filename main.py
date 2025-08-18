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
        
        def get_possible_moves():
            self.possible_moves = []
            # north
            if (self.current_position[0] - 1 >= 0):
                self.possible_moves.append("N")
            # east
            if (self.current_position[1] + 1 <= 3):
                self.possible_moves.append("E")
            # south
            if (self.current_position[0] + 1 <= 3):
                self.possible_moves.append("S")
            # west
            if (self.current_position[0] - 1 >= 0):
                self.possible_moves.append("W")
            
        def move():
            get_possible_moves()
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
                

            
     