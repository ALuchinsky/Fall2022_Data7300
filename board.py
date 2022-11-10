import numpy as np

class Board:
    def __init__(self, n=3):
        self.n = n
        self.board = self.initBoard(n)
        self.masks = self.initMasks()
        
        
    def initBoard(self, n=3):
        self.board = [[0 for i in range(self.n)] for j in range(self.n)]
        return self.board
    
    def initMasks(self):
        board = self.board
        masks = []
        # rows
        for I in range(len(board)):
            masks  = masks + [[(i,j) for i in range(len(board)) for j in range(len(board[i])) if i==I]]  
        # columns
        for J in range(len(board)):
            masks  = masks + [[(i,j) for i in range(len(board)) for j in range(len(board[i])) if j==J]]
        # diagonals
        masks  = masks + [[(i,j) for i in range(len(board)) for j in range(len(board[i])) if i==j]]
        masks  = masks + [[(i,j) for i in range(len(board)) for j in range(len(board[i])) if i==len(board)-j-1]]
        return masks
    
    def getMoves(self):
        board = self.board
        return [[i,j] for i in range(len(board)) 
                    for j in range(len(board[i])) 
                    if board[i][j]==0]
    
    def getWinner(self):
        board = self.board
        masks = self.masks
        for M in masks:
            masked_board = np.array([board[i][j] for (i,j) in M])
            candidate = masked_board[0]
            if np.all(masked_board == candidate) and candidate != 0:
                return candidate
        if len(self.getMoves())==0:
            # It's a draw
            return 0
        else:
            # game is not finished yet
            return -1
        
    def movesToBoard(self, moves):
        for move in moves:
            player = move[0]
            coords = move[1]
            self.board[coords[0]][coords[1]] = player
        return self.board

    def printBoard(self, printWinner=True):
        board = self.board
        markers = dict({0:' ', 1:"X", 2:"\x1b[31m0\x1b[0m"})
        print("    ", end="")
        for j in range(len(board[0])):
            print(j, end=" ");
        print("")
        for i in range(len(board)):
            print(i," |", end="")
            for j in range(len(board[i])):
                  print(markers[board[i][j]], end=" ")
            print("|")
        if printWinner:
            print("Winner:", self.getWinner())
