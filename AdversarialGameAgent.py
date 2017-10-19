from pacmanGame import *

import sys
import random
sys.path.insert(0, ".\\aima-python-master")
from utils import *

from collections import namedtuple

infinity = float('inf')
GameState = namedtuple('GameState', 'to_move, utility, pos')

class PacmanAdversarialGameProblem:

    def __init__(self,game):
        self.game=game
        self.maxDist=game.MAZE_WIDTH+game.MAZE_HEIGHT

    def actions(self, state): #figure out all legal moves based on current pos
        (x,y)=state.pos
        moves=[]
        if not self.game.isWall(x-1,y):
            moves.append(x-1,y)
        if not self.game.isWall(x+1,y):
            moves.append(x+1,y)
        if not self.game.isWall(x,y-1):
            moves.append(x,y-1)
        if not self.game.isWall(x,y+1):
            moves.append(x,y+1)
        return moves

    def result(self, state, move):
        return GameState(to_move=state.to_move, utility=self.compute_utility(state.to_move), pos=move)
        # to_move is the player

    def utility(self, state):
        return state.utility

    def terminal_test(self, state,step):
        if(state.to_move == "Pacman"):
            return (step == self.game.ghostPos[0])
        else:
            return (step == self.game.pacmanPos[0])#ghost is playing


    def compute_utility(self, to_move):
        length=len(self.game.capsulePos)+len(self.game.foodPos)
        return -length if to_move == "Pacman" else length
#needs -length to return smaller number for the terms of the game
    def evalForPacman(self,state):
        distance=manhattanDistance(state.to_move,self.game.ghostPos[0])
        if(distance <= 2):
            return distance
        else:
            return(compute_utility(state.to_move))
        #improve in homework

class PacmanAdvGameAgent():
    def __init__(self,problem):
        self.problem=problem
        self.state=GameState(to_move="Pacman", utility=self.problem.compute_utility("Pacman"),
                             pos=self.problem.game.pacmanPos[0])

    def get_action(self):
        self.state = GameState(to_move="Pacman", utility=self.problem.compute_utility("Pacman"),
                               pos=self.problem.game.pacmanPos[0])
        step=alphabeta_search(self.state,self.problem,d=2, cutoff_test=None,
                                eval_fn=self.problem.evalForPacman)
        prev=self.problem.game.pacmanPos[0]
        action=getDirection(prev,step)
        if(self.problem.terminal_test(self.state, step)):
            collision = True
        else:
            collision = False
        return collision, action, step


class GhostGameAgent():
    def __init__(self, problem,index=0):
        self.problem = problem
        self.state = GameState(to_move="Ghost",
                               utility=self.problem.compute_utility("Ghost"),
                               pos=self.problem.game.ghostPos[0])
        self.index=index

    def get_action(self,gAgent):
       # step = alphabeta_search(self.state, self.problem, d=2, cutoff_test=None, eval_fn=None)
        game=self.problem.game

        if (gAgent == "random"):
            step = randomMove(self.actions(game.ghostPos[self.index]))
        else:
            step = betterThanRandomMove(self.actions(game.ghostPos[self.index]),game)
        prev = game.ghostPos[self.index]
        action = getDirection(prev, step)
        if (self.problem.terminal_test(self.state, step)):
            collision = True
        else:
            collision = False
        return (collision, action, step)
        # else:
        #     raise NotImplementedError

    def actions(self, pos):
        (x, y) = pos
        moves = []
        game=self.problem.game
        if not game.isWall(x - 1, y):
            moves.append((x - 1, y))
        if not game.isWall(x + 1, y):
            moves.append((x + 1, y))
        if not game.isWall(x, y - 1):
            moves.append((x, y - 1))
        if not game.isWall(x, y + 1):
            moves.append((x, y + 1))
        return moves

def minmanhattanDistance(listPos, state):
    (x1, y1) = state;
    distanses=[(abs(x1 - x2) + abs(y1 - y2))
                for (x2, y2) in listPos]
    return min(distanses)


def randomMove(moves):
    return random.choice(moves)

def betterThanRandomMove(moves,game):
    i=random.randint(0,9)
    if i<5:
        return random.choice(moves)
    else:
        ds=[manhattanDistance(pos,game.pacmanPos[0]) for pos in moves]
        return moves[ds.index(min(ds))]

def alphabeta_search(state, problem, d=4, cutoff_test=None, eval_fn=None):
    """Search problem to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = state.to_move

    # Functions used by alphabeta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -infinity
        for a in problem.actions(state):
            v = max(v, min_value(problem.result(state, a),
                                 alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = infinity
        for a in problem.actions(state):
            v = min(v, max_value(problem.result(state, a),
                                 alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or
                   (lambda state, depth: depth > d))
    #or problem.terminal_test(state
    eval_fn = eval_fn or (lambda state: problem.utility(state, player))
    best_score = -infinity
    beta = infinity
    best_action = None
    choices = []  # if more than one best, add it here
    for a in problem.actions(state):
        v = min_value(problem.result(state, a), best_score, beta, 1)
        choices.append((a, v))

    best_value = max(choices, key=lambda item: item[1])
    bests = [p for p in choices if best_value[1] == p[1]]
    # list = [a for a in bests if a[1] == v]
    if bests:
        (best_action, best_score) = random.choice(bests)
    return best_action

def minimax_decision(state, problem):
    """Given a state in a problem, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = problem.to_move(state)

    def max_value(state):
        if problem.terminal_test(state):
            return problem.utility(state, player)
        v = -infinity
        for a in problem.actions(state):
            v = max(v, min_value(problem.result(state, a)))
        return v

    def min_value(state):
        if problem.terminal_test(state):
            return problem.utility(state, player)
        v = infinity
        for a in problem.actions(state):
            v = min(v, max_value(problem.result(state, a)))
        return v

    # Body of minimax_decision:
    return argmax(problem.actions(state),
                  key=lambda a: min_value(problem.result(state, a)))
