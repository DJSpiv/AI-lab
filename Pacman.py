from pacmanGame import *
import pygame.gfxdraw
# Create the constants (go ahead and experiment with different values)

class Pacman:
    def __init__(self,pacmanPos, index, color,radius,size,walls,mw,mh):
        self.color=color
        self.radius=radius
        self.size=size
        self.pos=pacmanPos
        self.walls=walls
        self.maze_width=mw
        self.maze_height=mh

    def drawPacman(self,surf,direction="West"): #size=1 open widely
        (x, y) = self.pos[0];
        pos = x - int(x) + y - int(y)
        width = WALL_RADIUS/2 + WALL_RADIUS * math.sin(math.pi * pos)
        delta = width

        if (direction == 'West'):
            startRad = degreesToRadians(180+delta)
            endRad =degreesToRadians(180-delta)
        elif (direction == 'North'):
            startRad = degreesToRadians(90+delta)
            endRad =  degreesToRadians(90-delta)
        elif (direction == 'South'):
            startRad = degreesToRadians(270+delta)
            endRad = degreesToRadians(270-delta)
        else:
            startRad = degreesToRadians(delta)
            endRad =degreesToRadians(360-delta)

        rect = (x*self.radius*2, y* self.radius*2, self.radius * 2, self.radius * 2)
        if (self.size==0):
            #pygame.gfxdraw.arc(surf, int(x*self.radius*2), int(y* self.radius*2), self.radius, startRad, endRad, self.color)
            pygame.draw.arc(surf, self.color, rect, startRad, endRad, self.radius)
        else:
            #pygame.gfxdraw.arc(surf, int(x * self.radius * 2), int(y * self.radius * 2), self.radius, startRad+.5, endRad-.5,self.color)
            pygame.draw.arc(surf, self.color, rect, startRad+.5, endRad-.5, self.radius)

    def makeMove(self,direction):
        #(x,y)=self.pos.pop(0)
        (x, y) = self.pos.pop(0)
        self.direction=direction
        if direction=='North':
            if((y-1>=0) and (not self.walls[x][y-1])):
                y=y-1
        if direction == 'South':
            if ((y + 1 < self.maze_height) and (not self.walls[x][y+1])):
                y =y+1

        if direction == 'West':
            if ((x - 1 >= 0) and (not (self.walls[x-1][y]))):
                x=x-1
        if direction == 'East':
            if ((x + 1 <self.maze_width) and (not self.walls[x+1][y])):
                x = x + 1
        self.pos.append((x,y))
        return (x,y)

