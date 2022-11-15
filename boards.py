import numpy as np

def find_best_MC_move(board, masks, mc_trials = 1000, max_moves = 100):
    player_to_move = get_player_to_move(board)
    next_player = 3 - player_to_move
    print("Finding best move for player ", player_to_move, "next one will be ", next_player)
    moves = boards.getMoves(board)
    print("There are ", len(moves), " possible moves")
    moves_stats = np.zeros(len(moves))
    for iMove in tqdm.tqdm(range(len(moves))):
        bb = board.copy()
        bb[ moves[iMove]] = player_to_move # making test move
        # MC the play with the opponent starting
        boards_collection = np.tile(bb, (mc_trials,1))
        mc = np.apply_along_axis(lambda b: boards.simulate(b, masks, player_to_move=next_player, max_moves=max_moves)[1], 1, boards_collection)
        moves_stats[iMove] = np.mean(mc==player_to_move)
    i_best_move = np.argmax(moves_stats)
    return moves[i_best_move], player_to_move, sorted([list(m) for m in zip(moves, moves_stat)], key= lambda r: -r[1])


def get_player_to_move(board):
    return 1 if np.sum(board==1) == np.sum(board==2) else 2

def getMoves(board):
    return [i for i in range(len(board)) if board[i]==0]    

def getWinner(board, masks):
    for m in masks:
        w = np.unique(board[m])
        if len(w)==1 and w[0]>0:
            return w[0]
    if np.sum(board==0)==0:
        return 0
    return -1

def ij_to_ind(i, j, n):
    return i*n + j
def ind_to_ij(ind, n):
    return [ind//n, i % n]

def initMasks(n):
    masks = np.empty((0,n), dtype = 'int')
    # rows
    for I in range(n):
        masks = np.vstack([masks, [ij_to_ind(i, j, n) for i in range(n) for j in range(n) if i==I]])
    # columns
    for J in range(n):
        masks = np.vstack([masks, [ij_to_ind(i,j, n) for i in range(n) for j in range(n) if j==J]])
    # diagonals
    masks = np.vstack([masks, [ij_to_ind(i,j, n) for i in range(n) for j in range(n) if i==j]])
    masks = np.vstack([masks, [ij_to_ind(i,j, n) for i in range(n) for j in range(n) if i==n-j-1]])
    return np.array(masks)


def simulate(board_, masks, player_to_move=1, max_moves = 1000):
    board = board_.copy()
    n_moves = 0;
    while n_moves < max_moves:
        n_moves = n_moves + 1
        moves = getMoves(board)
        if len(moves)==0:
            return board, getWinner(board, masks);
        next_move = moves[np.random.randint(0, len(moves))]
        board[next_move] = player_to_move
        w = getWinner(board, masks)
        if(w>0):
            return board, w
        player_to_move = 3-player_to_move
    return board, -1


def printBoard(b, n, printWinner=True, markers = dict({0:' ', 1:"X", 2:"\x1b[31m0\x1b[0m"})):
    board = b.reshape([n, n])
    print("    ", end="")
    for j in range(len(board[0])):
        print(j, end=" ");
    print("")
    for i in range(len(board)):
        print(i," |", end="")
        for j in range(len(board[i])):
              print(markers[board[i][j]], end=" ")
        print("|")


class LinearBoard:
    def __init__(self, n=3):
        self.n = n
        self.board = np.zeros(self.n*self.n, dtype = 'int')
        self.masks = self.initMasks()
        
    def initMasks(self):
        return initMasks(self.n)
    
    def ij_to_ind(self, i, j):
        return ij_to_ind(i,j,self.n)
    
    def ind_to_ij(ind):
        return ind_to_ij(ind, self.n)
    
    def getMoves(self):
         return getMoves(self.board)
        

    def getWinner(self):
        return getWinner(self.board, self.masks)
   
    def printBoard(self, printWinner=True):
        printBoard(self.board, self.n)
        

        


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


