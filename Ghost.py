import random
from pacmanGame import *


GHOST_SHAPE = [
    ( 0,    -0.3 ),
    ( 0.25, -0.75 ),
    ( 0.5,  -0.3 ),
    ( 0.75, -0.75 ),
    ( 0.75, 0.5 ),
    ( 0.5,  0.75 ),
    (-0.5,  0.75 ),
    (-0.75, 0.5 ),
    (-0.75, -0.75 ),
    (-0.5,  -0.3 ),
    (-0.25, -0.75 )
  ]
GHOST_SIZE=0.65
GHOST_OFFSET=0.1*WALL_RADIUS
class Ghost:
    def __init__(self,game,color,index=0):
        self.index=index
        self.game=game
        maze_width=game.MAZE_WIDTH
        maze_height=game.MAZE_HEIGHT
        self.walls=game.walls
        if len(game.ghostPos)==0:  #no ghost in maze, then generate one randomly
           raise ValueError("No ghost found in the maze!")
        self.color=color

    def drawGhost(self,surf, direction=None, index=0):
        (x,y)=add(self.game.ghostPos[0],(.5, .5))
        coords = []
        w_r=WALL_RADIUS*GHOST_SIZE
        offset=WALL_RADIUS*(1-GHOST_SIZE)
        for (x1, y1) in GHOST_SHAPE:
            offsetX=x*offset
            offsetY=y*offset
            screen_x=int((x-x1)*w_r+offsetX)
            screen_y=int((y-y1)*w_r+offsetY)
            coords.append((screen_x,screen_y))

        offsetX=x*offset
        offsetY=y*offset
        dx_left=int((x-.3/1.5)*w_r+offsetX)
        dx_right=int((x+.5/1.5)*w_r+offsetX)
        dy=int((y-.3/1.5)*w_r+offsetY)
        dx_left_pupil = int((x-.35/1.5) * w_r+offsetX)
        dx_right_pupil = int((x+.55/1.5) * w_r+offsetX)
        dy_pupil = int((y-.25/1.5) * w_r+offsetY)
        leftEye = (dx_left,dy)
        rightEye = (dx_right,dy)
        leftPupil = (dx_left_pupil,dy_pupil)
        rightPupil = (dx_right_pupil,dy_pupil)
        pygame.draw.polygon(surf, self.color, coords)
        pygame.draw.circle(surf, WHITE, leftEye, int(WALL_RADIUS*GHOST_SIZE*.4), 0)
        pygame.draw.circle(surf, WHITE, rightEye, int(WALL_RADIUS*GHOST_SIZE*.4), 0)
        pygame.draw.circle(surf, BLACK, leftPupil, int(WALL_RADIUS*GHOST_SIZE*.22), 0)
        pygame.draw.circle(surf, BLACK, rightPupil, int(WALL_RADIUS*GHOST_SIZE*.22), 0)


