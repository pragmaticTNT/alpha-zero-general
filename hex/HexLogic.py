from enum import Enum
import numpy as np

'''
Author:
Date:
Board class.
Board data:
  1=white, -1=black, 0=empty
  Hexes are stored and manipulated as (x,y) tuples.
  x is the row, y is the column.
'''

class Board():

    # list of all 6 directions on the board, as (x,y) offsets
    __directions = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n):
        "Set up initial board configuration."

        self.n = n
        # print(self.n)
        # Create the empty board array.
        self.pieces = [[0 for _ in range(self.n)] for _ in range(self.n)]
        # print(self.pieces)

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def get_legal_moves(self):
        """Returns all the legal moves for the given color.
        1 for white, -1 for black
        """
        moves = []  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x, y] == 0:
                    moves.append((x,y))
        return moves

    def has_legal_moves(self, color):
        for y in range(self.n):
            for x in range(self.n):
                if self[x, y] == 0:
                    return True
        return False

    def neighbour(self, index):
        x, y = index//self.n, index%self.n
        n = ((x + i, y + j) for (i,j) in self.__directions)
        return [x*self.n + y for (x,y) in n if 0 <= x < self.n and 0 <= y < self.n]

    def has_path_between(self, source, sink, player):
        stack = source[:]
        visited = [False for _ in range(self.n*self.n)]
        while len(stack) != 0:
            idx = stack.pop()
            x, y = idx//self.n, idx%self.n
            if self[x,y] == player:
                if idx in sink:
                    return True
                stack += [n for n in self.neighbour(idx) if not visited[idx]]
                visited[idx] = True
        return False

    # Always from the perspective of the first player. If second player, transpose board
    def has_path(self, player):
        source = []
        sink = []
        assert player == 1 or player == -1
        if player == 1:
            source += [i for i in range(self.n)]
            sink += [i+self.n*(self.n-1) for i in range(self.n)]
        elif player == -1:
            source += [i*self.n for i in range(self.n)]
            sink += [i*self.n-1 for i in range(1, self.n+1)]
        #print(f"Sink: {sink}\nSource: {source}\nPlayer: {player}")
        return self.has_path_between(source, sink, player)

    def execute_move(self, index, color):
        #print(f"index = {index}")
        assert self.pieces[index] == 0
        self.pieces[index] = color

    def __repr__(self):
        hex_content = {
                -1: 'X',
                0: '.',
                1: 'O'
        }
        rep_pieces = ""
        for i in range(self.n):
            rep_pieces += ' '*i
            rep_pieces += "  ".join([hex_content[self.pieces[i,j]] for j in range(self.n)])
            rep_pieces += '\n'
        return rep_pieces

def test():
    win_board = [
            [1, -1, 1, -1, -1],
            [1, -1, 1, 1, 1],
            [1, -1, 1, -1, -1],
            [-1, -1, -1, -1, 1],
            [1, -1, 1, -1, 1]
    ]
    board = np.array(win_board)
    b = Board(5)
    b.pieces = board
    print(b)
    print(f"Has path 1: {b.has_path(1)}")
    print(f"Has path -1: {b.has_path(-1)}")
    return

if __name__ == "__main__":
    test()
