import numpy as np

def simulate(board, player_to_move=1, max_moves = 1000):
    n_moves = 0;
    while n_moves < max_moves:
        n_moves = n_moves + 1
        moves = board.getMoves()
        if len(moves)==0:
            return board.board, board.getWinner();
        next_move = moves[np.random.randint(0, len(moves))]
        board.board[next_move] = player_to_move
        w = board.getWinner()
        if(w>0):
            return board.board, w
        player_to_move = 3-player_to_move
    return board.board, -1


class Board:
    def __init__(self, n=3):
        self.n = n
        self.board = self.initBoard(n)
        self.masks = self.initMasks()
        
        
    def initBoard(self, n=3):
        self.board = np.array([[0 for i in range(self.n)] for j in range(self.n)])
        return self.board
    
    def copy(b):
        newB = Board()
        newB.board = b.board.copy()
        newB.masks = b.masks
        return newB

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
    
    def boardToMoves(self):
        board = self.board
        return [ (board[i][j], [i,j])  for i in range(len(board)) for j in range(len(board[i])) if board[i][j] >0]

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
        

        