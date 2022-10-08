from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .HexLogic import Board
import numpy as np

class HexGame(Game):
    square_content = {
        -1: "X",
        0: ".",
        1: "O"
    }

    @staticmethod
    def getPiece(piece):
        return HexGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n

    def transpose_act(self, action):
        x, y = (action//self.n, action%self.n)
        return y*self.n + x

    def getNextState(self, canonical_board, player, canonical_action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        action = canonical_action if player == 1 else self.transpose_act(canonical_action)
        b = Board(self.n)
        b.pieces = np.copy(canonical_board)
        move = (action//self.n, action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0 for _ in range(self.getActionSize())]
        # unfortunately "board" is a np.Array and not a Board
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves()

        #print(board)
        #print(f"legalMoves = {legalMoves}")
        for x, y in legalMoves:
            valids[self.n*x+y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player, verbose=False):
        b = Board(self.n)
        b.pieces = np.copy(board)
        assert player == 1 or player == -1
        if b.has_path(player):
            return 1
        elif b.has_path(-player):
            if verbose:
                print(b)
                print(f"Player {-player} has won? {b.has_path(-player)}")
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state-and-transpose if player==-1
        assert player == 1 or player == -1
        return player * (board if player == 1 else board.transpose())

    # This function does a lot more than just get the symmetries... It generates the data to be used in the training examples
    def getSymmetries(self, board, pi):
        sym = [(board, pi)]
        piBoard = np.reshape(pi[:], (self.n, self.n))

        xBoard = np.fliplr(board)
        xPi = np.fliplr(piBoard)
        yBoard = np.flipud(xBoard)
        yPi = np.flipud(xPi)
        sym += [(yBoard, list(yPi.ravel()))]
        return sym

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def matrixRepresentation(board_str, n):
        return np.frombuffer(board_str, dtype='<i8').reshape((n, n))

    def display(self, board):
        b = Board(self.n)
        b.pieces = np.copy(board)
        print(b)
        return
