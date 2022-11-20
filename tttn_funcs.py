import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import tqdm

N = 5;
print("Simple ",N,"x",N, "game")

masks = []

def initBoard():
    global masks, N
    return np.zeros((N,N),'int')

def interesting_move(board, r0, c0, d=1):
    if not board[r0, c0]==0:
        return False
    window = board[ max(0, r0-d):min(r0+d+1, board.shape[0]), max(0, c0-d):min(c0+d+1, board.shape[1])]
    if np.alltrue(window==0):
        return False
    return True


def getMoves(board, d=1):
    dim1, dim2 = board.shape
    moves = np.array(
        [[i,j] for i in range(dim1) for j in range(dim2) if interesting_move(board, i, j, d=d)]
    )
    if len(moves)==0:
        moves = np.array(
            [[i,j] for i in range(len(board)) 
                for j in range(len(board[i])) 
                if board[i][j]==0]
        )
    return moves


#
def initMasks(board):
    global masks
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

masks = initMasks(initBoard())

def getWinner(board):
    global masks
    for M in masks:
        masked_board = np.array([board[i][j] for (i,j) in M])
        candidate = masked_board[0]
        if np.all(masked_board == candidate) and candidate != 0:
            return candidate
    if len(getMoves(board))==0:
        # It's a draw
        return 0
    else:
        # game is not finished yet
        return -1

def printBoard(board, markers = dict({0:' ', 1:"X", 2:"\x1b[31m0\x1b[0m"})):
    for i in range(len(board)):
        print("|", end="")
        for j in range(len(board[i])):
              print(markers[board[i][j]], end=" ")
        print("|")
    print("Winner:", getWinner(board))
    
def printMoves(moves):
    printBoard(movesToBoard(moves))

def movesToBoard(moves):
    board = initBoard()
    for move in moves:
        player = move[0]
        coords = move[1]
        board[coords[0]][coords[1]] = player
    return board

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.backend import reshape
from keras.utils.np_utils import to_categorical

def getModel():
    board = initBoard()
    numCells = len(board)*len(board[0])
    outcomes = 3
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(numCells, )))
    model.add(Dropout(0.2))
    model.add(Dense(125, activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(outcomes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model

def gamesToWinLossData(games, train_ratio = 0.8):
    board = initBoard()
    X = []
    y = []
    for game in games:
        winner = getWinner(movesToBoard(game))
        for move in range(len(game)):
            X.append(movesToBoard(game[:(move + 1)]))
            y.append(winner)

    X = np.array(X).reshape((-1, len(board)*len(board[0])))
    y = to_categorical(y)
    
    # Return an appropriate train/test split
    trainNum = int(len(X) * train_ratio)
    return (X[:trainNum], X[trainNum:], y[:trainNum], y[trainNum:])


def bestMove(board, model, player, rnd=0):
    scores = []
    moves = getMoves(board)
    
    # Make predictions for each possible move
    for i in range(len(moves)):
        future = np.array(board)
        future[moves[i][0]][moves[i][1]] = player
        prediction = model.predict(future.reshape((-1, len(board)*len(board[0]))), verbose = None)[0]
        if player == 1:
            winPrediction = prediction[1]
            lossPrediction = prediction[2]
        else:
            winPrediction = prediction[2]
            lossPrediction = prediction[1]
        drawPrediction = prediction[0]
        if winPrediction - lossPrediction > 0:
            scores.append(winPrediction - lossPrediction)
        else:
            scores.append(drawPrediction - lossPrediction)

    # Choose the best move with a random factor
    bestMoves = np.flip(np.argsort(scores))
    for i in range(len(bestMoves)):
        if random.random() * rnd < 0.5:
            return moves[bestMoves[i]]

    # Choose a move completely at random
    return moves[random.randint(0, len(moves) - 1)]

def gameStats(games, player=1):
    stats = {"win": 0, "loss": 0, "draw": 0}
    for game in games:
        result = getWinner(movesToBoard(game))
        if result == -1:
            continue
        elif result == player:
            stats["win"] += 1
        elif result == 0:
            stats["draw"] += 1
        else:
            stats["loss"] += 1
    
    winPct = stats["win"] / len(games) * 100
    lossPct = stats["loss"] / len(games) * 100
    drawPct = stats["draw"] / len(games) * 100

    print("Results for player %d:" % (player))
    print("Wins: %d (%.1f%%)" % (stats["win"], winPct))
    print("Loss: %d (%.1f%%)" % (stats["loss"], lossPct))
    print("Draw: %d (%.1f%%)" % (stats["draw"], drawPct))
    
def simulateGame(p1=None, p2=None, rnd=0, d=1, max_moves = None, start_history = None, player_to_move_ = 1):
    if start_history is None:
        history = []
    else:
        history = start_history.copy()
    board = movesToBoard(history)
    playerToMove = player_to_move_
    n_moves = 0
    while getWinner(board) == -1:
        n_moves += 1
        if max_moves is not None and n_moves>max_moves:
            break
        # Chose a move (random or use a player model if provided)
        move = None
        if playerToMove == 1 and p1 != None:
            move = bestMove(board, p1, playerToMove, rnd)
        elif playerToMove == 2 and p2 != None:
            move = bestMove(board, p2, playerToMove, rnd)
        else:
            moves = getMoves(board, d=d)
            # print("moves=", moves)
            # print("[moves]=", len(moves))
            move = moves[random.randint(0, len(moves) )]
        
        # Make the move
        # print("Making move ", move)
        board[move[0]][move[1]] = playerToMove
        
        # Add the move to the history
        history.append((playerToMove, move))
        
        # Switch the active player
        playerToMove = 1 if playerToMove == 2 else 2
        
    return history

def gen_tournament(p1=None, p2=None, n=100, rnd=0.9, d=1, tqdm_disable = False):
    t_games = []
    n_fault = 0
    for _ in tqdm.tqdm(range(n), disable = tqdm_disable):
        try:
            moves = simulateGame(p1=p1, p2=p2, rnd=rnd, d=1)
            winner = getWinner(movesToBoard(moves))
            t_games = t_games + [(winner, moves)]
        except:
            n_fault = n_fault + 1
    return t_games, n_fault

def getCNNModel():
    board = initBoard()
    numCells = len(board)*len(board[0])
    num_rows = len(board)
    outcomes = 3
    CNNmodel = keras.models.Sequential()
    CNNmodel.add( keras.layers.InputLayer(input_shape=(numCells, )))
    CNNmodel.add( keras.layers.Reshape( target_shape = (num_rows, num_rows, 1)))
    CNNmodel.add( keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = "same"))
    CNNmodel.add( keras.layers.Reshape( target_shape = (1,5*5*32)))
    CNNmodel.add( keras.layers.BatchNormalization())
    CNNmodel.add( keras.layers.Dense(100, activation = "relu"))
    CNNmodel.add( keras.layers.BatchNormalization())
    CNNmodel.add( keras.layers.Dense(20, activation = "relu"))
    CNNmodel.add( keras.layers.BatchNormalization())
    CNNmodel.add( keras.layers.Dense(3, activation = "relu"))
    CNNmodel.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    CNNmodel.add( keras.layers.Reshape(target_shape=(3,)))
    # training the model
    # CNNmodel.compile(loss = keras.losses.categorical_crossentropy,
    #           optimizer = keras.optimizers.SGD(lr = 0.01),
    #           metrics =['accuracy'])
    CNNmodel.compile(optimizer='adam',
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
#    [CNNmodel.input_shape, CNNmodel.output_shape]
    return CNNmodel