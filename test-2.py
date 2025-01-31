import itertools
import pickle
import random
import sys
import pygame
import numpy as np
from datetime import datetime
from timeit import default_timer as timer
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Constants
CELL_SIZE = 10
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
STC_COLOR = (255, 255, 0)
RTC_COLOR = (255, 0, 0)
SLIDER_COLOR = (200, 200, 200)
HANDLE_COLOR = (100, 100, 100)

# Simulation Parameters
running = False
days_elapsed = 0
delta_t = 1 / 12
frame_rate = 10
t_max = 1000
dt = 1 / 12
INITIAL_SIZE = 50
MAX_SIZE = 400
EXPANSION_SIZE = 5
param_cct = 24
P_MAX = 15
param_stem = P_MAX + 1
param_potm = 1

# System configuration
sys_nruns = 3
sys_t_run = []
sys_visu_partial = False
sys_visu_end = True
sys_print = True
sys_report = False
sys_save = False

# Vectors and matrices
vect_deat = np.zeros(t_max + 1)
vect_prol = np.zeros(t_max + 1)
vect_potm = np.zeros(t_max + 1)
vect_stem = np.zeros(t_max + 1)
vect_deat[:round(0.5 * t_max)] = 0.01 * dt
vect_deat[round(0.5 * t_max):] = 0.01 * dt
vect_prol[:] = (24 / param_cct * dt)
vect_potm[:round(0.4 * t_max)] = 10 * dt
vect_potm[round(0.4 * t_max):] = 10 * dt
vect_stem[:] = 0.1

neighbors = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=int)
neighbors_success = np.array([0, 0, 0], dtype=np.int64)
neighbors_fail = np.array([1, 0, 0], dtype=np.int64)
neighbors_permutation = list(itertools.permutations(np.arange(0, 8, dtype=int)))
neighbors_all = np.array(neighbors_permutation[:])

np.random.seed(6)
random.seed(6)

# Pygame setup
pygame.init()
screen_width, screen_height = 600, 600
screen = pygame.display.set_mode((screen_width, screen_height + 100))  # Increased height for controls
pygame.display.set_caption("Tumor Growth Simulation")
clock = pygame.time.Clock()

# Control positions
start_button = pygame.Rect(10, screen_height + 10, 80, 30)
stop_button = pygame.Rect(100, screen_height + 10, 80, 30)
reset_button = pygame.Rect(190, screen_height + 10, 80, 30)
speed_slider = pygame.Rect(300, screen_height + 25, 200, 20)

# Data structures
vect_pop_stem = np.empty((sys_nruns, t_max + 1), dtype=np.int64)
vect_pop_reg = np.empty((sys_nruns, t_max + 1), dtype=np.int64)
vect_rad = np.empty((sys_nruns, t_max + 1), dtype=np.int64)
vect_tumor_snapshots = [None] * sys_nruns

class Slider:
    def __init__(self, rect, min_val, max_val, initial_val):
        self.rect = rect
        self.min = min_val
        self.max = max_val
        self.val = initial_val
        self.dragging = False
        
    def draw(self, surface):
        pygame.draw.rect(surface, SLIDER_COLOR, self.rect)
        handle_x = self.rect.x + (self.val - self.min)/(self.max - self.min) * self.rect.width
        pygame.draw.circle(surface, HANDLE_COLOR, (int(handle_x), self.rect.centery), 10)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if self.dragging and event.type == pygame.MOUSEMOTION:
            x = max(self.rect.left, min(event.pos[0], self.rect.right))
            self.val = self.min + (x - self.rect.left)/self.rect.width * (self.max - self.min)
            return True
        return False

speed_control = Slider(speed_slider, 1, 60, frame_rate)

class TumorSimulation:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.current_run = 0
        self.running = False
        self.completed_runs = 0
        self.grid, self.tumor_center = create_initial_matrix(INITIAL_SIZE)
        self.tumor = self.grid.copy()
        self.tumor_n = INITIAL_SIZE
        self.tumor_rad = [0]
        self.days_elapsed = 0
        self.t = 0
        self.snapshots = [None] * (t_max + 1)
        self.snapshots[0] = self.tumor.copy()
        self.start_time = 0

    def update(self):
        if self.t < t_max:
            self.t += 1
            self.tumor, new_rad = main_ca(self.tumor, self.tumor_rad[-1], self.t, self.tumor_center)
            self.tumor_rad.append(new_rad)
            
            if new_rad >= self.tumor_center - 1:
                self.tumor, self.tumor_n, self.tumor_center = create_extension(
                    self.tumor, self.tumor_n, self.tumor_center, EXPANSION_SIZE)
            
            self.days_elapsed += delta_t
            self.snapshots[self.t] = self.tumor.copy()
            return False
        else:
            pop_tot, pop_stem, pop_reg = calc_pop(self.tumor, param_stem)
            vect_pop_stem[self.current_run] = pop_stem
            vect_pop_reg[self.current_run] = pop_reg
            vect_rad[self.current_run] = np.array(self.tumor_rad)
            vect_tumor_snapshots[self.current_run] = self.snapshots
            
            self.current_run += 1
            if self.current_run < sys_nruns:
                self.reset()
                self.running = True
                return False
            else:
                return True

# [Keep all previous functions identical except draw_buttons and draw_info]

def draw_buttons():
    pygame.draw.rect(screen, (0, 255, 0), start_button)
    pygame.draw.rect(screen, (255, 0, 0), stop_button)
    pygame.draw.rect(screen, (0, 0, 255), reset_button)
    font = pygame.font.SysFont(None, 24)
    screen.blit(font.render("Start", True, BLACK), (start_button.x + 10, start_button.y + 5))
    screen.blit(font.render("Stop", True, BLACK), (stop_button.x + 10, stop_button.y + 5))
    screen.blit(font.render("Reset", True, BLACK), (reset_button.x + 10, reset_button.y + 5))
    
    # Draw speed slider label
    speed_label = font.render(f"Speed: {int(speed_control.val)} FPS", True, BLACK)
    screen.blit(speed_label, (speed_slider.x, speed_slider.y - 20))

def draw_info(days):
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Days: {days:.2f} | Run: {sim.current_run + 1}/{sys_nruns}", True, BLACK)
    screen.blit(text, (screen_width - 250, screen_height + 15))

# [Keep all other functions identical]

sim = TumorSimulation()

# Main loop
while True:
    current_frame_rate = int(speed_control.val)
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if start_button.collidepoint(event.pos):
                sim.running = True
                sim.start_time = timer()
            elif stop_button.collidepoint(event.pos):
                sim.running = False
            elif reset_button.collidepoint(event.pos):
                sim.reset()
            else:
                speed_control.handle_event(event)
        elif event.type == pygame.MOUSEMOTION:
            speed_control.handle_event(event)
    
    if sim.running and sim.completed_runs < sys_nruns:
        complete = sim.update()
        if complete:
            sim.completed_runs = sys_nruns
            sys_t_run.append(timer() - sim.start_time)
            
            # Post-simulation analysis
            pop_stem_mean, pop_stem_std = calc_mean(vect_pop_stem)
            pop_reg_mean, pop_reg_std = calc_mean(vect_pop_reg)
            tumor_rad_mean, tumor_rad_std = calc_mean(vect_rad)
            rep_idx = calc_index_representative(vect_pop_reg, pop_reg_mean,
                                              vect_pop_stem, pop_stem_mean,
                                              vect_rad, tumor_rad_mean)
            
            if sys_visu_end:
                see_end()
                see_snapshots()
                see_chances()
            
            if sys_print:
                report_general()
            
            if sys_report:
                save_files()
            
            sim.running = False

    # Drawing
    screen.fill(WHITE)
    draw_grid(sim.tumor)
    draw_buttons()
    draw_info(sim.days_elapsed)
    speed_control.draw(screen)
    pygame.display.flip()
    clock.tick(current_frame_rate)