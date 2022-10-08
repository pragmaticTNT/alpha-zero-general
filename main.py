import logging

import sys
from MCTS import MCTS
import numpy as np

import coloredlogs

from Coach import Coach
##from othello.OthelloGame import OthelloGame as Game
##from othello.pytorch.NNet import NNetWrapper as nn
from hex.HexGame import HexGame as Game
from hex.HexLogic import Board
from hex.pytorch.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

def load_nn(g):
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')
    return nnet

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(6)

    nnet = load_nn(g)

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

def interactive_protocol_loop():
    while command := input():
        command = command.split()
        log.info(command)
        if command[0] != "start":
            raise "Unexpected command"
        
        g = Game(int(command[1]))
        log.info(g.n)
        board = Board(g.n)
        curPlayer = 1
        for action in command[2:]:
            int_action = int(action)
            x, y = int_action//g.n, int_action%g.n 
            board.execute_move((x,y), curPlayer)
            curPlayer = -curPlayer

        nnet = load_nn(g)
        args = dotdict({'numMCTSSims': 800, 'cpuct':1.0})
        mcts = MCTS(g, nnet, args)
        player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
        action = player(g.getCanonicalForm(np.array(board.pieces), curPlayer))
        action = action if curPlayer == 1 else g.transpose_act(action)
        print(action)

if __name__ == "__main__":
    interactive_protocol_loop()
    # main()
