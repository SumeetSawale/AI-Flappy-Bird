import neat.config
import pygame
import neat
import time
import os
import random
pygame.font.init()

HEIGHT = 800
WIDTH = 500
GRAVITY = 3
VELOCITY = 5

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load("images/bird1.png")),
             pygame.transform.scale2x(pygame.image.load("images/bird2.png")),
             pygame.transform.scale2x(pygame.image.load("images/bird3.png"))]

PIPE_IMG = pygame.transform.scale2x(pygame.image.load("images/pipe.png"))

BASE_IMG = pygame.transform.scale2x(pygame.image.load("images/base.png"))

BG_IMG = pygame.transform.scale2x(pygame.image.load("images/bg.png"))

SCORE_FONT = pygame.font.SysFont("sans", 40)


class Bird :
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.img_count = 0
        self.image = self.IMGS[0]

    def jump(self) :
        # Upward velocity of 10.5
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self) :

        # Time since we last tapped
        self.tick_count += 1

        # s = ut + (1/2)at^2
        d = self.velocity*self.tick_count + .5*GRAVITY*self.tick_count**2 

        # Terminal velocity
        if d > 16 :
            d = 16

        # While moving up move little more
        if d < 0 :
            d -= 2

        # updating y cord
        self.y += d

        if d < 0 or self.y < self.height + 50 :
            if self.tilt < self.MAX_ROTATION :
                self.tilt = self.MAX_ROTATION
        
        else :
            if self.tilt > -90 :
                self.tilt -= self.ROT_VEL
        
    def draw(self, win) :
        self.img_count = (self.img_count + 1) % (3 * self.ANIMATION_TIME)
        self.image = self.IMGS[self.img_count//self.ANIMATION_TIME]

        if self.tilt < -80 :
            self.image = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2
    
        rotated_img = pygame.transform.rotate(self.image, self.tilt)
        new_rect = rotated_img.get_rect(center=self.image.get_rect(topleft=(self.x, self.y)).center)

        win.blit(rotated_img, new_rect.topleft)

    def get_mask(self) :
        return pygame.mask.from_surface(self.image)
    

class Pipe :
    GAP = 200
    # VELOCITY = 5

    def __init__(self, x):
        self.x = x
        self.height = 0

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()
    
    def set_height(self) :
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self) :
        self.x -= VELOCITY

    def draw(self, win) :
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird) :
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if (b_point or t_point) :
            return True
        return False
    

class Base :
    # VELOCITY = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self) :
        self.x1 -= VELOCITY
        self.x2 -= VELOCITY

        if self.x1 + self.WIDTH < 0 :
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0 :
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win) :
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def draw_window(win, birds, pipes, base, score, gen) :
    win.blit(BG_IMG, (0, 0))
    
    for pipe in pipes :
        pipe.draw(win)

    text = SCORE_FONT.render("Score: " + str(score), 1, (255,255,255))
    win.blit(text, (WIDTH - 10 - text.get_width(), 10))

    text = SCORE_FONT.render("Gen: " + str(gen), 1, (255,255,255))
    win.blit(text, (10, 10))

    base.draw(win)

    for bird in birds :
        bird.draw(win)
    
    pygame.display.update()

GEN = 0
def main(genomes, config) :
    global GEN
    GEN += 1

    nets = []
    ge = []
    birds = []

    for _, g in genomes :
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.fitness = 0
        ge.append(g)

    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIDTH, HEIGHT))

    clock = pygame.time.Clock()

    score = 0
    flag = True
    while flag :
        clock.tick(30)
        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                flag = False
                pygame.quit()
                quit()

        pipe_idx = 0
        if len(birds) > 0 :
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width() :
                pipe_idx = 1
        
        else :
            flag = False
            break

        for idx, bird in enumerate(birds) :
            bird.move()
            ge[idx].fitness += 0.1

            output = nets[idx].activate((bird.y, abs(bird.y - pipes[pipe_idx].height), abs(bird.y - pipes[pipe_idx].bottom)))

            if output[0] > 0.5 :
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes :
            for idx, bird in enumerate(birds) :
                if pipe.collide(bird) :
                    ge[idx].fitness -= 1
                    birds.pop(idx)
                    nets.pop(idx)
                    ge.pop(idx)

                if not pipe.passed and pipe.x < bird.x :
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0 :
                rem.append(pipe)

            pipe.move()
        
        if add_pipe :
            score += 1
            for g in ge :
                g.fitness += 5
            pipes.append(Pipe(600))

        for r in rem :
            pipes.remove(r)

        for idx, bird in enumerate(birds) :
            if bird.y + bird.image.get_height() >= 730 or bird.y < 0 :
                birds.pop(idx)
                nets.pop(idx)
                ge.pop(idx)
        
        if score > 10 :
            break
        
        base.move()
        draw_window(win, birds, pipes, base, score, GEN)



def run(config_path) :
    config = neat.config.Config(neat.DefaultGenome, 
                                neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet, 
                                neat.DefaultStagnation, 
                                config_path)
    
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)

if __name__ == "__main__" :
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_file.txt")
    run(config_path)