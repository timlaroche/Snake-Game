import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import random
import numpy as np
import cv2
import os

width = 500
height = 500

cols = 25
rows = 20

class Snake_Env(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, server):
    if server == True:
      os.environ["SDL_VIDEODRIVER"] = "dummy"
    self.win = pygame.display.set_mode((width,height))
    self.s = snake((255,0,0), (10,10))
    self.s.addCube()
    self.snack = cube(self.randomSnack(rows,self.s), color=(0,255,0))
    self.flag = True
    self.clock = pygame.time.Clock()
    self.action_space = spaces.Discrete(4)
    # self.observation_space = spaces.Box(low = 0, high = 255, shape = (100, 100, 3), dtype=np.uint8)
    self.observation_space = spaces.Box(low = 0, high = 255, shape = (100, 100, 3), dtype=np.uint8)


  def step(self, action):
    reward = 0
    # pygame.time.delay(50)
    # self.clock.tick(10)
    self.s.move(action)
    headPos = self.s.head.pos
    if headPos[0] >= 20 or headPos[0] < 0 or headPos[1] >= 20 or headPos[1] < 0:
        # print("Score:", len(self.s.body))
        # self.s.reset((10, 10))
        reward -= 10
        self.flag = False

    if self.s.body[0].pos == self.snack.pos:
        self.s.addCube()
        self.snack = cube(self.randomSnack(rows,self.s), color=(0,255,0))
        reward += 1
        
    for x in range(len(self.s.body)):
        if self.s.body[x].pos in list(map(lambda z:z.pos, self.s.body[x+1:])):
            # print("Score:", len(self.s.body))
            # self.s.reset((10,10))
            reward -=10
            self.flag = False
            break

    return self.get_observation(), reward, not self.flag, {}

  def reset(self):
    # print(f"reset called! {self.flag}")
    self.s.reset((10, 10))
    self.flag = True
    return self.get_observation()

  def render(self, mode='human'):
    self.clock.tick(10)
    self.redrawWindow()

  def get_actions(self):
    for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                keys = pygame.key.get_pressed()

                for key in keys:
                    if keys[pygame.K_LEFT]:
                      return 1
                    elif keys[pygame.K_RIGHT]:
                      return 2
                    elif keys[pygame.K_UP]:
                      return 3
                    elif keys[pygame.K_DOWN]:
                      return 4

  def redrawWindow(self):
    self.win.fill((0,0,0))
    self.s.draw(self.win)
    self.snack.draw(self.win)
    pygame.display.update()
    pass

  def get_observation(self):
    surf = pygame.surfarray.array3d(pygame.display.get_surface())
    return cv2.resize(surf, (100, 100)) # resize to 100x100
    # return self.snack.pos

  def randomSnack(self, rows, item):
      positions = item.body

      while True:
          x = random.randrange(1,rows-1)
          y = random.randrange(1,rows-1)
          if len(list(filter(lambda z:z.pos == (x,y), positions))) > 0:
                 continue
          else:
                 break

      return (x,y)

class cube():
    rows = 20
    w = 500
    def __init__(self, start, dirnx=1, dirny=0, color=(255,0,0)):
        self.pos = start
        self.dirnx = dirnx
        self.dirny = dirny # "L", "R", "U", "D"
        self.color = color

    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos  = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)
            

    def draw(self, surface, eyes=False):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]
        
        pygame.draw.rect(surface, self.color, (i*dis+1,j*dis+1,dis-2,dis-2))
        if eyes:
            centre = dis//2
            radius = 3
            circleMiddle = (i*dis+centre-radius,j*dis+8)
            circleMiddle2 = (i*dis + dis -radius*2, j*dis+8)
            pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
            pygame.draw.circle(surface, (0,0,0), circleMiddle2, radius)    

class snake():
    body = []
    turns = {}
    
    def __init__(self, color, pos):
        #pos is given as coordinates on the grid ex (1,5)
        self.color = color
        self.head = cube(pos)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
    
    def move(self, action):
      if action == 1:
          self.dirnx = -1
          self.dirny = 0
          self.turns[self.head.pos[:]] = [self.dirnx,self.dirny]
      elif action == 2:
          self.dirnx = 1
          self.dirny = 0
          self.turns[self.head.pos[:]] = [self.dirnx,self.dirny]
      elif action == 3:
          self.dirny = -1
          self.dirnx = 0
          self.turns[self.head.pos[:]] = [self.dirnx,self.dirny]
      elif action == 4:
          self.dirny = 1
          self.dirnx = 0
          self.turns[self.head.pos[:]] = [self.dirnx,self.dirny]
        
      for i, c in enumerate(self.body):
          p = c.pos[:]
          if p in self.turns:
              turn = self.turns[p]
              c.move(turn[0], turn[1])
              if i == len(self.body)-1:
                  self.turns.pop(p)
          else:
              c.move(c.dirnx,c.dirny)
        
        
    def reset(self,pos):
        self.head = cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(cube((tail.pos[0]-1,tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(cube((tail.pos[0]+1,tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(cube((tail.pos[0],tail.pos[1]-1)))
        elif dx == 0 and dy == -1:
            self.body.append(cube((tail.pos[0],tail.pos[1]+1)))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy
    
    def draw(self, surface):
        for i,c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)