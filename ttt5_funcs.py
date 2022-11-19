import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

def initBoard():
    board = [
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]
    ]
    return board
#
def getMoves(board):
    return [[i,j] for i in range(len(board)) 
                for j in range(len(board[i])) 
                if board[i][j]==0]
#
def initMasks(board):
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
def getWinner(board, masks=masks):
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

def simulateGame(p1=None, p2=None, rnd=0):
    history = []
    board = initBoard()
    playerToMove = 1
    
    while getWinner(board) == -1:
        
        # Chose a move (random or use a player model if provided)
        move = None
        if playerToMove == 1 and p1 != None:
            move = bestMove(board, p1, playerToMove, rnd)
        elif playerToMove == 2 and p2 != None:
            move = bestMove(board, p2, playerToMove, rnd)
        else:
            moves = getMoves(board)
            move = moves[random.randint(0, len(moves) )]
        
        # Make the move
        board[move[0]][move[1]] = playerToMove
        
        # Add the move to the history
        history.append((playerToMove, move))
        
        # Switch the active player
        playerToMove = 1 if playerToMove == 2 else 2
        
    return history

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