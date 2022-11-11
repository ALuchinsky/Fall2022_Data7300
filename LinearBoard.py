import numpy as np

class LinearBoard:
    def __init__(self, n=3):
        self.n = n
        self.board = np.zeros(self.n*self.n, dtype = 'int')
        self.initMasks()
        
    def initMasks(self):
        n = self.n
        masks = np.empty((0,n), dtype = 'int')
        # rows
        for I in range(n):
            masks = np.vstack([masks, [self.ij_to_ind(i,j) for i in range(n) for j in range(n) if i==I]])
        # columns
        for J in range(n):
            masks = np.vstack([masks, [self.ij_to_ind(i,j) for i in range(n) for j in range(n) if j==J]])
        # diagonals
        masks = np.vstack([masks, [self.ij_to_ind(i,j) for i in range(n) for j in range(n) if i==j]])
        masks = np.vstack([masks, [self.ij_to_ind(i,j) for i in range(n) for j in range(n) if i==n-j-1]])
        self.masks = np.array(masks)
        return masks

    
    def ij_to_ind(self, i, j):
        return i*self.n + j
    def ind_to_ij(ind):
        return [ind//self.n, i % self.n]
    
    def getMoves(self):
        board = self.board
        return [i for i in range(len(board)) if board[i]==0]

    def getWinner(self):
        for m in self.masks:
            w = np.unique(self.board[m])
            if len(w)==1 and w[0]>0:
                return w[0]
        if np.sum(self.board==0)==0:
            return 0
        return -1
   
    def printBoard(self, printWinner=True):
        board = self.board.reshape([self.n, self.n])
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
        # if printWinner:
        #     print("Winner:", self.getWinner())
        

        
