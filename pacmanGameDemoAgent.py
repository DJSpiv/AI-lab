#using AIMA-PYTHON to solve pacman game

import pygame, sys, random, math

from pygame.locals import *
from Pacman import *
from pacmanGame import *
from Ghost import *
from AdversarialGameAgent import PacmanAdvGameAgent,PacmanAdversarialGameProblem,GhostGameAgent

def main():
    global FPSCLOCK, DISPLAYSURF, BASICFONT, SCORE_SURF, SCORE_RECT
    global walls, capsulePos,game,pacman,score, scoreText
    PACMANPLAYER = True

    score=0
    filename = ".\\layouts\\tinyMaze2.lay"
    filename=".\\layouts\\smallClassic.lay"
  #  filename = ".\\layouts\\MediumClassic.lay"
  #  filename = ".\\layouts\\minimaxClassic.lay"
    if(len(sys.argv)>1):
        print(sys.argv[1])
        gAgent=sys.argv[1]
    else:
        gAgent="random"
    game=PacGame(filename)
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((game.WINDOWWIDTH, game.WINDOWHEIGHT))

    strS=game.strS
    pygame.display.set_caption('PacMan')
    BASICFONT = pygame.font.Font('freesansbold.ttf', BASICFONTSIZE)
    BASICFONT = pygame.font.Font('freesansbold.ttf', BASICFONTSIZE)

    # Store the option buttons and their rectangles in OPTIONS.
    scoreText='SCORE:'+str(score)
    SCORE_SURF, SCORE_RECT = makeText(scoreText, TEXTCOLOR, BGCOLOR, 10, game.WINDOWHEIGHT - 40)
    DISPLAYSURF.blit(SCORE_SURF, SCORE_RECT)
    game.genMaze()
    walls=game.walls
    game.drawWall(DISPLAYSURF)
    pacman=Pacman(game.pacmanPos,0,PACCOLOR,PAC_SIZE,0,walls,game.MAZE_WIDTH,game.MAZE_HEIGHT)
    pacman.drawPacman(DISPLAYSURF)
    game.drawCapsule(DISPLAYSURF)
    game.drawFoods(DISPLAYSURF)
    ghost=Ghost(game,PINK,0)
    ghost.drawGhost(DISPLAYSURF)
    problem=PacmanAdversarialGameProblem(game)
    pacmanAdvAgent = PacmanAdvGameAgent(problem)
    ghostAgent=GhostGameAgent(problem)
    pygame.time.wait(1000)
    slideTo=None
    stop=False
    while not stop:# main game loop
        for event in pygame.event.get():  # event handling loop
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        scores=0
        while(game.capsulePos or game.foodPos):
            for event in pygame.event.get():  # event handling loop
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
            if PACMANPLAYER:
                (collision, pac_action, new_pos) = pacmanAdvAgent.get_action()
                if collision:
                    stop=True
                    break;
                currentPos=game.pacmanPos.pop(0)
                game.pacmanPos.append(new_pos)
                if new_pos in game.capsulePos:
                    index=game.capsulePos.index(new_pos)
                    game.capsulePos.pop(index)
                elif new_pos in game.foodPos:
                    index = game.foodPos.index(new_pos)
                    game.foodPos.pop(index)
                slideTo = pac_action
                if slideTo:
                    slideAnimation(pacman, currentPos, slideTo, "Ok", 8)  #

                scores+=1
                pygame.display.update()
                FPSCLOCK.tick(FPS)
            else: #ghost play
                (collision,action, new_pos) = ghostAgent.get_action(gAgent)
                if collision:
                    stop=True
                    break;
                currentPos=game.ghostPos.pop(0)
                game.ghostPos.append(new_pos)
                slideTo = action
                if slideTo:
                    slideAnimation(ghost,currentPos,slideTo, "Ok", 8)  #

                pygame.display.update()
                FPSCLOCK.tick(FPS)
            scoreText = 'SCORE:' + str(scores)
            SCORE_SURF, SCORE_RECT = makeText(scoreText, TEXTCOLOR, BGCOLOR, 10, game.WINDOWHEIGHT - 40)
            DISPLAYSURF.blit(SCORE_SURF, SCORE_RECT)
            PACMANPLAYER= (not PACMANPLAYER)
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def makeText(text, color, bgcolor, top, left):
    # create the Surface and Rect objects for some text.
    textSurf = BASICFONT.render(text, True, color, bgcolor)
    textRect = textSurf.get_rect()
    textRect.topleft = (top, left)
    return (textSurf, textRect)

def slideAnimation(spirit, currentPos,direction, message, animationSpeed):
    baseSurf = DISPLAYSURF.copy()
    (x,y)=currentPos
    xTop,yTop=x * PAC_SIZE* 2, y*PAC_SIZE*2
    pygame.draw.rect(DISPLAYSURF, BGCOLOR, (xTop,yTop, PAC_SIZE * 2, PAC_SIZE * 2))
#    spirit.makeMove(direction)#pop the initial pos in makeMove
    baseSurf = DISPLAYSURF.copy()
    if(type(spirit) is Ghost):
        spirit.drawGhost(baseSurf,direction)
    else:
        spirit.drawPacman(baseSurf,direction)  # open widely
    game.drawCapsule(baseSurf)
    game.drawFoods(baseSurf)
    DISPLAYSURF.blit(baseSurf, (0, 0))
    pygame.display.update()
    FPSCLOCK.tick(FPS)

if __name__ == '__main__':
    main()
