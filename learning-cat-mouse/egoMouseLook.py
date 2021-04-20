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


class Cat(cellular.Agent):
    cell = None
    score = 0
    colour = 'grey'

    def update(self):
        cell = self.cell
        if cell != mouse.cell and cell != cheese.cell:
            self.goTowards(mouse.cell)
            while cell == self.cell:
                self.goInDirection(random.randrange(directions))
                if self.cell == cheese.cell:
                    world.agents.remove(cheese)


class Cheese(cellular.Agent):
    colour = 'brown'

    def update(self):
        # looking diagonally
        cell = self.cell.neighbour[5]
        if cell == cat.cell:
            self.goInDirection(5)
        else: 
            cheese.cell = world.getCell(8,2)


class Mouse(cellular.Agent):
    colour = 'black'

    def __init__(self):
        self.ai = None
        self.ai = qlearn.QLearn(actions=list(range(directions)),
                                alpha=0.1, gamma=0.9, epsilon=0.1)
        self.eaten = 0
        self.fed = 0
        self.lastState = None
        self.lastAction = None

    def update(self):
        # calculate the state of the surrounding cells
        state = self.calcState()
        # asign a reward of -1 by default
        reward = -1

        # observe the reward and update the Q-value
        if self.cell == cat.cell:
            self.eaten += 1
            reward = -100
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, state)
            self.lastState = None

            #self.cell = pickRandomLocation()
            self.cell = world.getCell(6, 1)
            cat.cell = world.getCell(5, 6)
            if len(world.agents) < 3:
                world.addAgent(cheese, cell=world.getCell(8,2))
            return

        if cheese.cell == cat.cell:
            self.fed += 1
            reward = 50
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, state)
            self.lastState = None
            self.cell = world.getCell(6,1)
            cat.cell = world.getCell(5,6)
            return

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, state)

        # Choose a new action and execute it
        state = self.calcState()
        print(state)
        cell = self.cell
        t_actions = []
        while cell == self.cell:
            action = self.ai.chooseAction(state)
            self.lastState = state
            self.lastAction = action
            self.goInDirection(action, cheese)
            if cell == self.cell:
                t_actions.append(action)
                self.ai.actions.remove(action)
            else: 
                self.ai.actions.extend(t_actions)
                self.ai.actions.sort()


    def calcState(self):
        def cellvalue(cell):
            if cat.cell is not None and (cell.x == cat.cell.x and
                                         cell.y == cat.cell.y):
                return 3
            elif cheese.cell is not None and (cell.x == cheese.cell.x and
                                              cell.y == cheese.cell.y):
                return 2
            else:
                return 1 if cell.wall else 0

        return tuple([cellvalue(self.world.getWrappedCell(self.cell.x + j, self.cell.y + i))
                      for i,j in lookcells])

mouse = Mouse()
cat = Cat()
cheese = Cheese()

world = cellular.World(Cell, directions=directions, filename='/Users/edela/source/repos/basic_reinforcement_learning/worlds/eecs4401.txt')
world.age = 0

world.addAgent(cat, cell=world.getCell(5, 6))
world.addAgent(cheese, cell=world.getCell(8,2))
world.addAgent(mouse, cell=world.getCell(6, 1))

epsilonx = (0,100000)
epsilony = (0.1,0)
epsilonm = (epsilony[1] - epsilony[0]) / (epsilonx[1] - epsilonx[0])

endAge = world.age + 1000

while world.age < endAge:
    world.update()

    '''if world.age % 100 == 0:
        mouse.ai.epsilon = (epsilony[0] if world.age < epsilonx[0] else
                            epsilony[1] if world.age > epsilonx[1] else
                            epsilonm*(world.age - epsilonx[0]) + epsilony[0])'''

s = "{:d}, e: {:0.2f}, W: {:d}, L: {:d}".format(world.age, mouse.ai.epsilon, mouse.fed, mouse.eaten)
print(s)
mouse.eaten = 0
mouse.fed = 0
cat.cell=world.getCell(5, 6)
cheese.cell=world.getCell(8,2)
mouse.cell=world.getCell(6, 1)
world.display.activate(size=30)
world.display.delay = 1
while 1:
    world.update(mouse.fed, mouse.eaten)
    print(len(mouse.ai.q)) # print the amount of state/action, reward 
                          # elements stored
    import sys
    bytes = sys.getsizeof(mouse.ai.q)
    print("Bytes: {:d} ({:d} KB)".format(bytes, bytes//1024))