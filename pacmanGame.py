import sys, math
import pygame

FPS = 20
BLANK = None

WALL_RADIUS = 30
WALL_WIDTH=4
PAC_SIZE=int(WALL_RADIUS/2)
FOOD_SIZE=int(WALL_RADIUS/10)
CAPSULE_SIZE=int(WALL_RADIUS/4)

BASICFONTSIZE = 20

XMARGIN = int(WALL_RADIUS/2)
YMARGIN = int(WALL_RADIUS/2)

#                 R    G    B
BLACK =         (  0,   0,   0)
WHITE =         (255, 255, 255)
BRIGHTBLUE =    (  0,  50, 255)
DARKTURQUOISE = (  3,  54,  73)
GREEN =         (  0, 204,   0)
YELLOW=         (255, 255,   0)
PINK=           (255, 105, 180)
LIGHTBLUE=      (135, 206, 250)
RED=            (255,   0,   0)
LIGHTPINK=      (255, 182, 193)

BUTTONCOLOR = WHITE
BUTTONTEXTCOLOR = BLACK
BGCOLOR = DARKTURQUOISE
PACCOLOR = YELLOW
FOODCOLOR=LIGHTPINK
CAPSULECOLOR=WHITE
TEXTCOLOR = YELLOW
WALLCOLOR = BRIGHTBLUE
BASICFONTSIZE = 20

BUTTONCOLOR = WHITE
BUTTONTEXTCOLOR = BLACK
MESSAGECOLOR = WHITE

class PacGame:
    def __init__(self,filename):
        maze_file = open(filename)
        self.pacmanPos = []
        self.capsulePos = []
        self.foodPos=[]
        self.ghostPos=[]
        strS = [line.strip() for line in maze_file]
        self.strS=strS
        # Create the constants (go ahead and experiment with different values)
        self.walls = [[False for i in range(len(strS))] for j in range(len(strS[0]))]

        self.MAZE_WIDTH = len(strS[0])
        self.MAZE_HEIGHT = len(strS)
        self.WINDOWWIDTH = self.MAZE_WIDTH * WALL_RADIUS
        self.WINDOWHEIGHT = self.MAZE_HEIGHT * WALL_RADIUS + BASICFONTSIZE * 2

    def genMaze(self):

        for x in range(0, self.MAZE_HEIGHT):
            for y in range(0, self.MAZE_WIDTH):
                if self.strS[x][y] == '%':
                    self.walls[y][x] = True
                elif self.strS[x][y] == 'P':
                    self.pacmanPos.append((y, x))
                elif self.strS[x][y] == '.':
                    self.capsulePos.append((y, x))
                elif self.strS[x][y]==' ':
                    self.foodPos.append((y,x))
                elif self.strS[x][y]=='G':
                    self.ghostPos.append((y,x))


    def drawWall(self, DISPLAYSURF):
        wallColor=WALLCOLOR
        width=WALL_WIDTH
        walls=self.walls
        for xNum in range(self.MAZE_WIDTH):
            for yNum in range(self.MAZE_HEIGHT):
                if (walls[xNum][yNum] == True):
                    pos = (xNum, yNum)
                    nIsWall = self.isWall(pos[0], pos[1] - 1)
                    wIsWall = self.isWall(pos[0] - 1, pos[1])
                    sIsWall = self.isWall(pos[0], pos[1] + 1)
                    eIsWall = self.isWall(pos[0] + 1, pos[1])
                    neIsWall = self.isWall(pos[0] + 1, pos[1] - 1)
                    nwIsWall = self.isWall(pos[0] - 1, pos[1] - 1)
                    seIsWall = self.isWall(pos[0] + 1, pos[1] + 1)
                    swIsWall = self.isWall(pos[0] - 1, pos[1] + 1)
                    posMargin = add(pos, (.5, .5))
                    if (nIsWall and (not (neIsWall and nwIsWall and eIsWall and wIsWall))):
                        pygame.draw.line(DISPLAYSURF, wallColor,
                                         add(mul(posMargin, WALL_RADIUS), (0, WALL_RADIUS * .5 * (-1))),
                                         mul(posMargin, WALL_RADIUS), width)
                    if (sIsWall and (not (seIsWall and swIsWall and eIsWall and wIsWall))):
                        pygame.draw.line(DISPLAYSURF, wallColor, mul(posMargin, WALL_RADIUS),
                                         add(mul(posMargin, WALL_RADIUS), (0, WALL_RADIUS * .5)), width)
                    if (wIsWall and (not (swIsWall and nwIsWall and nIsWall and sIsWall))):
                        pygame.draw.line(DISPLAYSURF, wallColor,
                                         add(mul(posMargin, WALL_RADIUS), (WALL_RADIUS * .5 * (-1), 0)),
                                         mul(posMargin, WALL_RADIUS), width)
                    if (eIsWall and (not (seIsWall and neIsWall and nIsWall and sIsWall))):
                        pygame.draw.line(DISPLAYSURF, wallColor, mul(posMargin, WALL_RADIUS),
                                         add(mul(posMargin, WALL_RADIUS), ((WALL_RADIUS * .5), 0)), width)

                    if (eIsWall and nIsWall and (not neIsWall)):
                        pygame.draw.line(DISPLAYSURF, wallColor, mul(posMargin, WALL_RADIUS),
                                         add(mul(posMargin, WALL_RADIUS), ((WALL_RADIUS * .5), 0)),
                                         width)  # draw east wall
                        pygame.draw.line(DISPLAYSURF, wallColor,
                                         add(mul(posMargin, WALL_RADIUS), (0, WALL_RADIUS * .5 * (-1))),
                                         mul(posMargin, WALL_RADIUS), width)  # draw northwall
                    if (eIsWall and sIsWall and (not seIsWall)):
                        pygame.draw.line(DISPLAYSURF, wallColor, mul(posMargin, WALL_RADIUS),
                                         add(mul(posMargin, WALL_RADIUS), ((WALL_RADIUS * .5), 0)),
                                         width)  # draw east wall
                        pygame.draw.line(DISPLAYSURF, wallColor, mul(posMargin, WALL_RADIUS),
                                         add(mul(posMargin, WALL_RADIUS), (0, WALL_RADIUS * .5)), width)
                    if (wIsWall and sIsWall and (not swIsWall)):
                        pygame.draw.line(DISPLAYSURF, wallColor,
                                         add(mul(posMargin, WALL_RADIUS), (WALL_RADIUS * .5 * (-1), 0)),
                                         mul(posMargin, WALL_RADIUS), width)  # draw west wall
                        pygame.draw.line(DISPLAYSURF, wallColor, mul(posMargin, WALL_RADIUS),
                                         add(mul(posMargin, WALL_RADIUS), (0, WALL_RADIUS * .5)), width)
                    if (wIsWall and nIsWall and (not nwIsWall)):
                        pygame.draw.line(DISPLAYSURF, wallColor,
                                         add(mul(posMargin, WALL_RADIUS), (WALL_RADIUS * .5 * (-1), 0)),
                                         mul(posMargin, WALL_RADIUS), width)  # draw west wall
                        pygame.draw.line(DISPLAYSURF, wallColor,
                                         add(mul(posMargin, WALL_RADIUS), (0, WALL_RADIUS * .5 * (-1))),
                                         mul(posMargin, WALL_RADIUS), width)  # draw north wall

    def isWall(self, x, y):
        if x < 0 or y < 0:
            return False
        if x >= len(self.walls) or y >= len(self.walls[0]):
            return False
        return self.walls[x][y]

    def drawCapsule(self,surf):
        for i in range(0, len(self.capsulePos)):
            (x,y)=add(self.capsulePos[i],(0.5, 0.5))

            posScreen=mul((x,y),WALL_RADIUS)
            pygame.draw.circle(surf, CAPSULECOLOR, posScreen, CAPSULE_SIZE, 0)

    def drawFoods(self, surf):
        for i in range(0, len(self.foodPos)):
            (x, y) = add(self.foodPos[i], (0.5, 0.5))

            posScreen = mul((x, y), WALL_RADIUS)
            pygame.draw.circle(surf, FOODCOLOR, posScreen, FOOD_SIZE, 0)

    def nextDirectionIsValid(self, direction, pos):
        (x, y) = pos
        if direction == 'North':
            if ((y - 1 >= 0) and (not self.walls[x][y - 1])):
                return (True, (x, y - 1))
        elif direction == 'South':
            if ((y + 1 < self.MAZE_HEIGHT) and (not self.walls[x][y + 1])):
                return (True, (x, y + 1))
        elif direction == 'West':
            if ((x - 1 >= 0) and (not (self.walls[x - 1][y]))):
                return (True, (x - 1, y))
        elif direction == 'East':
            if ((x + 1 < self.MAZE_WIDTH) and (not self.walls[x + 1][y])):
                return (True, (x + 1, y))
        return (False,pos)

def degreesToRadians(deg):
    return deg / 180.0 * math.pi

def  add(x, y):
    return (x[0] + y[0], x[1] + y[1])

def sub(x,y):
    return (x[0]-y[0],x[1]-y[1])

def mul(pos,r):
    return (int(pos[0]*r), int(pos[1]*r))

def manhattanDistance(xy1, xy2):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def getDirection(prev,step):
    if(sub(step,prev)==(1,0)):
        dire="East"
    elif (sub(step,prev)==(-1,0)):
        dire="West"
    elif (sub(step,  prev) == (0, -1)):
        dire = "North"
    elif (sub(step, prev) == (0, 1)):
        dire = "South"
    else:
        dire=None
    return dire
