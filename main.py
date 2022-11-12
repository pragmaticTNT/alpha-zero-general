import logging
import os

import sys
from MCTS import MCTS
import numpy as np

import coloredlogs

from Coach import Coach
from MCTS import MCTS
from Arena import Arena
import numpy as np

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
    'arenaCompare': 100,        # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'learning': False,

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

    if args.learning:
        if args.load_model:
            log.info("Loading 'trainExamples' from file...")
            c.loadTrainExamples()

        log.info('Starting the learning process 🎉')
        c.learn()
    else:
        log.info('Running tournament')
        tournament(c, 5, 1, 305)

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
# Generator for pairs of valid filename

def tournament(coach, ncomparisons: int, minNetId: int, maxNetId: int):
    checkpoint = 'temp/'
    for i in range(maxNetId, minNetId-1, -1):
        newFilename = coach.getCheckpointFile(i)
        newFilePath = os.path.join(checkpoint, newFilename)
        if os.path.exists(newFilePath):
            log.info(f'{newFilename} exists')
            icomp = 1
            for j in range(i-1, minNetId-1, -1):
                if icomp > ncomparisons:
                    break
                oldFilename = coach.getCheckpointFile(j)
                oldFilePath = os.path.join(checkpoint, oldFilename)
                if os.path.exists(oldFilePath):
                    log.info(f'... comparing against {oldFilename} [{icomp}]')
                    result = compareNets(coach.game, oldFilename, newFilename)
                    writePgn(result, j, i)
                    icomp += 1

def compareNets(game: Game, oldNetFilename: str, newNetFilename: str) -> tuple[int, int, int]:
    onet = nn(game)
    onet.load_checkpoint(folder=args.checkpoint, filename=oldNetFilename)
    omcts = MCTS(game, onet, args)

    nnet = nn(game)
    nnet.load_checkpoint(folder=args.checkpoint, filename=newNetFilename)
    nmcts = MCTS(game, nnet, args)

    arena = Arena(lambda x: np.argmax(omcts.getActionProb(x, temp=0)),
                  lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), game)

    # Returns triples of integers: (owins, nwins, draws)
    return arena.playGames(args.arenaCompare)

def writePgn(results:tuple[int, int, int], oldId: int, newId: int, filename:str='out.pgn'):
    filler = "1. e4\n\n"
    with open(filename, 'a') as f:
        owins, nwins, draws = results
        for _ in range(owins):
            f.write(f'[White "Net {oldId}"]\n[Black "Net {newId}"]\n[Result "1-0"]\n{filler}')
        for _ in range(nwins):
            f.write(f'[White "Net {oldId}"]\n[Black "Net {newId}"]\n[Result "0-1"]\n{filler}')
        for _ in range(draws):
            f.write(f'[White "Net {oldId}"]\n[Black "Net {newId}"]\n[Result "1/2-1/2"]\n{filler}')

if __name__ == "__main__":
    interactive_protocol_loop()
    # main()
