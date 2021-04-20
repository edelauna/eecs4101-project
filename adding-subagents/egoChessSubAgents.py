from math import inf
import time
import random
import shelve
from importlib import reload

import pdb

import cellular
reload(cellular)
import qlearn_mod_random as qlearn # to use the alternative exploration method
#import qlearn # to use standard exploration method
reload(qlearn)

directions = 8
reset_board = True
lookdist = 2
lookcells = []
for i in range(-lookdist,lookdist+1):
    for j in range(-lookdist,lookdist+1):
        if (abs(i) + abs(j) <= lookdist) and (i != 0 or j != 0):
            lookcells.append((i,j))

def pickRandomLocation():
    while 1:
        x = random.randrange(world.width)
        y = random.randrange(world.height)
        cell = world.getCell(x, y)
        if not (cell.wall or len(cell.agents) > 0):
            return cell


class Cell(cellular.Cell):
    wall = False

    def colour(self):
        if self.wall:
            return 'blue'
        else:
            return 'white'

    def load(self, data):
        if data == 'X':
            self.wall = True
        else:
            self.wall = False


class WhiteKing(cellular.Agent):
    cell = None
    score = 0
    colour = 'grey'

    def update(self):
        self.old_cells = []
        self.old_cells.append(self.cell)
        if reset_board:
            self.cell = world.getCell(5, 6)
            return
        cell = self.old_cells[0]
        if cell != blkKing.cell and cell != pawn.cell:
            self.goTowards(blkKing.cell)
            while cell == self.cell:
                self.goInDirection(random.randrange(directions))
                if self.cell == pawn.cell:
                    world.agents.remove(pawn)

class Player(cellular.Agent):
    count = 0
    reset = False
    def update(self):
        self.old_cells = []
        self.old_cells.append(blkKing.cell)
        self.old_cells.append(pawn.cell)
        global reset_board
        if reset_board:
            if self.count == 0:
                pawn.update()
                self.setAgentAttr(pawn)
                self.count += 1
                self.reset = True
            else:
                blkKing.update()
                self.setAgentAttr(blkKing)
                self.count = 0
                self.reset = False
                reset_board = False
        else:
            # get q values from pieces
            q_pawn = pawn.update()
            if q_pawn is None: q_pawn = [-inf]
            q_mouse = blkKing.update()
            if q_mouse is None: q_mouse = [-inf]
            dec = 0
            if max(q_pawn) == max(q_mouse):
                dec = random.random()
            elif max(q_pawn) > max(q_mouse):
                dec = 1
            else: dec = 0         
            if dec < 0.5:
                pawn.cell = self.old_cells[1]
                pawn.lastState = None
                self.setAgentAttr(blkKing)                
            else:
                if not reset_board:
                    blkKing.cell = self.old_cells[0]
                blkKing.lastState = None
                self.setAgentAttr(pawn)
    
    def setAgentAttr(self, agent):
        self.colour = agent.colour
        self.cell = agent.cell

class Pawn(cellular.Agent):
    colour = 'brown'

    def __init__(self):
        self.ai = None
        # Action 0 for forward motion and 5 for diagonal
        self.ai = qlearn.QLearn(actions=[4,5],
                                alpha=0.1, gamma=0.9, epsilon=0.1)
        self.lastState = None
        self.lastAction = None

    def update(self):
        global reset_board
        # calculate the state of the surrounding cells
        state = self.calcState()

        if reset_board:
            if self.cell == whtKing.old_cells[0]:
                reward = 50
                if self.lastState is not None:
                    self.ai.learn(self.lastState, self.lastAction, reward, state)
            self.cell = world.getCell(8,2)
            # Add back the pawn if removed by "Cat"
            if self not in world.agents:
                world.addAgent(self, cell=world.getCell(8,2))
            return
        
        # asign a reward of -1 by default
        reward = -1

        # observe the reward and update the Q-value
        if self.cell == whtKing.cell:
            reward = -100
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, state)
            self.lastState = None
            return
        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, state)

        print("Pawn_state:")
        print(state)
        cell = self.cell
        action,q = self.ai.chooseAction(state, return_q=True)
        self.lastState = state
        self.lastAction = action
        if action == 4:
            self.goInDirection(action, [whtKing,blkKing])
        else:
                # looking diagonally
                cell = self.cell.neighbour[5]
                if cell == whtKing.cell:
                    self.goInDirection(action)
        return q
        
    def calcState(self):
        def cellvalue(cell):
            if whtKing.cell is not None and (cell.x == whtKing.cell.x and
                                         cell.y == whtKing.cell.y):
                return 3
            elif blkKing.cell is not None and (cell.x == blkKing.cell.x and
                                              cell.y == blkKing.cell.y):
                return 2
            else:
                return 1 if cell.wall else 0

        return tuple([cellvalue(self.world.getWrappedCell(self.cell.x + j, self.cell.y + i))
                      for i,j in lookcells])


class BlackKing(cellular.Agent):
    colour = 'black'

    def __init__(self):
        self.ai = None
        self.ai = qlearn.QLearn(actions=list(range(directions)),
                                alpha=0.1, gamma=0.9, epsilon=0.05)
        self.eaten = 0
        self.fed = 0
        self.lastState = None
        self.lastAction = None

    def update(self):
        global reset_board
        if reset_board:
            self.cell = world.getCell(6,1)
            return
        # calculate the state of the surrounding cells
        state = self.calcState()
        # asign a reward of -1 by default
        reward = -1

        # observe the reward and update the Q-value
        if self.cell == whtKing.cell:
            self.eaten += 1
            reward = -100
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, state)
            self.lastState = None
            reset_board = True
            return

        # Moved to pawn agent
        if pawn.cell == whtKing.cell:
            self.fed += 1 #tracker for display
            reset_board = True
            return
        #     Learning Moved to Pawn Agen
        #     self.fed += 1
        #     reward = 50
        #     if self.lastState is not None:
        #         self.ai.learn(self.lastState, self.lastAction, reward, state)
        #     self.lastState = None
        #     reset_board = True
        #     return

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, state)

        # Choose a new action and execute it
        state = self.calcState()
        print(state)
        cell = self.cell
        t_actions = []
        while cell == self.cell:
            (action,q) = self.ai.chooseAction(state, return_q=True)
            self.lastState = state
            self.lastAction = action
            self.goInDirection(action, [pawn,whtKing])
            if cell == self.cell:
                t_actions.append(action)
                self.ai.actions.remove(action)
            else: 
                self.ai.actions.extend(t_actions)
                self.ai.actions.sort()
        return q

    def calcState(self):
        def cellvalue(cell):
            if whtKing.cell is not None and (cell.x == whtKing.cell.x and
                                         cell.y == whtKing.cell.y):
                return 3
            elif pawn.cell is not None and (cell.x == pawn.cell.x and
                                              cell.y == pawn.cell.y):
                return 2
            else:
                return 1 if cell.wall else 0

        return tuple([cellvalue(self.world.getWrappedCell(self.cell.x + j, self.cell.y + i))
                      for i,j in lookcells])

blkKing = BlackKing()
whtKing = WhiteKing()
pawn = Pawn()
player = Player()

world = cellular.World(Cell, directions=directions, filename='/Users/edela/source/repos/basic_reinforcement_learning/worlds/eecs4401-2.txt')
world.age = 0

world.addAgent(whtKing, cell=world.getCell(5, 6))
world.addAgent(pawn, cell=world.getCell(8,2))
world.addAgent(blkKing, cell=world.getCell(6, 1))
world.addAgent(player, cell=world.getCell(8, 2))
world.main_agents.append(whtKing)
world.main_agents.append(player)

epsilonx = (0,100000)
epsilony = (0.1,0)
epsilonm = (epsilony[1] - epsilony[0]) / (epsilonx[1] - epsilonx[0])

endAge = world.age + 102400

while world.age < endAge:
    world.update()

    '''if world.age % 100 == 0:
        blkKing.ai.epsilon = (epsilony[0] if world.age < epsilonx[0] else
                            epsilony[1] if world.age > epsilonx[1] else
                            epsilonm*(world.age - epsilonx[0]) + epsilony[0])'''

s = "{:d}, e: {:0.2f}, W: {:d}, L: {:d}".format(world.age, blkKing.ai.epsilon, blkKing.fed, blkKing.eaten)
print(s)
blkKing.eaten = 0
blkKing.fed = 0
whtKing.cell=world.getCell(5, 6)
pawn.cell=world.getCell(8,2)
blkKing.cell=world.getCell(6, 1)
world.display.activate(size=30)
world.display.delay = 1
while 1:
    world.update(blkKing.fed, blkKing.eaten)
    print(len(blkKing.ai.q)) # print the amount of state/action, reward 
                          # elements stored
    import sys
    bytes = sys.getsizeof(blkKing.ai.q)
    print("Bytes: {:d} ({:d} KB)".format(bytes, bytes//1024))