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
screen = pygame.display.set_mode((screen_width, screen_height + 50))
pygame.display.set_caption("Tumor Growth Simulation")
clock = pygame.time.Clock()
start_button = pygame.Rect(10, screen_height + 10, 80, 30)
stop_button = pygame.Rect(100, screen_height + 10, 80, 30)
reset_button = pygame.Rect(190, screen_height + 10, 80, 30)

# Data structures
vect_pop_stem = np.empty((sys_nruns, t_max + 1), dtype=np.int64)
vect_pop_reg = np.empty((sys_nruns, t_max + 1), dtype=np.int64)
vect_rad = np.empty((sys_nruns, t_max + 1), dtype=np.int64)
vect_tumor_snapshots = [None] * sys_nruns

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
            # Pass current tumor_center to main_ca
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
            
            # Increment run counter before checking completion
            self.current_run += 1
            
            if self.current_run < sys_nruns:
                # Reset for next run without resetting current_run
                self.t = 0
                self.days_elapsed = 0
                self.grid, self.tumor_center = create_initial_matrix(INITIAL_SIZE)
                self.tumor = self.grid.copy()
                self.tumor_n = INITIAL_SIZE
                self.tumor_rad = [0]
                self.snapshots = [None] * (t_max + 1)
                self.snapshots[0] = self.tumor.copy()
                return False
            else:
                return True

def create_initial_matrix(size):
    tumor = np.zeros((size, size), dtype=np.int64)
    center = size // 2
    tumor[center, center] = P_MAX + 1
    return tumor, center

def draw_grid(tumor):
    rows, cols = tumor.shape
    cell_w = screen_width // cols
    cell_h = screen_height // rows

    for row in range(rows):
        for col in range(cols):
            cell = tumor[row, col]
            if cell == 0:
                color = WHITE
            elif cell == P_MAX + 1:
                color = STC_COLOR
            else:
                intensity = int(255 * (cell / P_MAX))
                color = (intensity, 0, 0)
            pygame.draw.rect(screen, color, (col*cell_w, row*cell_h, cell_w, cell_h))

def draw_buttons():
    pygame.draw.rect(screen, (0, 255, 0), start_button)
    pygame.draw.rect(screen, (255, 0, 0), stop_button)
    pygame.draw.rect(screen, (0, 0, 255), reset_button)
    font = pygame.font.SysFont(None, 24)
    screen.blit(font.render("Start", True, BLACK), (start_button.x + 10, start_button.y + 5))
    screen.blit(font.render("Stop", True, BLACK), (stop_button.x + 10, stop_button.y + 5))
    screen.blit(font.render("Reset", True, BLACK), (reset_button.x + 10, reset_button.y + 5))

def draw_info(days):
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Days: {days:.2f} | Run: {sim.current_run + 1}/{sys_nruns}", True, BLACK)
    screen.blit(text, (screen_width - 250, screen_height + 15))

def order_sweep(tumor_backup):
    i, j = np.nonzero(tumor_backup[1:-1, 1:-1])
    cord = np.transpose([i, j]) + 1
    order = np.arange(len(cord))
    np.random.shuffle(order)
    return cord, order

def calc_chance(cord, order):
    cord_n = len(cord)
    return (np.random.rand(cord_n), np.random.rand(cord_n), np.random.rand(cord_n))

def calc_free_spots(A, i, j):
    neighbors_order = neighbors_all[np.random.randint(len(neighbors_all))]
    for n in neighbors_order:
        ii, jj = neighbors[n]
        if A[i + ii, j + jj] == 0:
            neighbors_success[1], neighbors_success[2] = ii, jj
            return neighbors_success
    return neighbors_fail

def calc_rad(i, j, radius, nc):
    return max(np.sqrt((nc - i)**2 + (nc - j)**2), radius)

def main_ca(tumor, tumor_rad_new, t_step, tumor_center):  # Add tumor_center as parameter
    # Keep existing code but use passed tumor_center
    cord, order = order_sweep(tumor)
    tumor_death, tumor_prolif, tumor_migr = calc_chance(cord, order)
    
    # Rest of the function remains the same
    for n in order:
        i, j = cord[n]
        cell = tumor[i][j]
        if cell == 0: continue
        
        if cell < param_stem and tumor_death[n] <= vect_deat[t_step]:
            tumor[i][j] = 0
            continue
            
        if tumor_prolif[n] <= vect_prol[t_step] or tumor_migr[n] <= vect_potm[t_step]:
            spots = calc_free_spots(tumor, i, j)
            if spots[0] == 0:
                ni, nj = i + spots[1], j + spots[2]
                if tumor_prolif[n] <= vect_prol[t_step]:
                    if cell == param_stem:
                        tumor[ni][nj] = param_stem if random.random() <= vect_stem[t_step] else P_MAX
                    else:
                        tumor[i][j] -= 1
                        tumor[ni][nj] = tumor[i][j]
                else:
                    tumor[ni][nj] = cell
                    tumor[i][j] = 0
                tumor_rad_new = calc_rad(ni, nj, tumor_rad_new, tumor_center)  # Now uses passed center
    
    return tumor, tumor_rad_new

def calc_pop(tumor, _):
    stem = np.sum(tumor == param_stem)
    total = np.sum(tumor > 0)
    return total, stem, total - stem

def create_extension(tumor, n, nc, n_plus):
    aux_B = np.zeros((n, n_plus), dtype=int)
    aux_C = np.zeros((n_plus, 2*n_plus + n), dtype=int)
    tumor = np.concatenate((aux_B, tumor, aux_B), axis=1)
    tumor = np.concatenate((aux_C, tumor, aux_C), axis=0)
    return tumor, n + 2*n_plus, (n + 2*n_plus) // 2

def calc_mean(vector):
    return np.mean(vector, axis=0), np.std(vector, axis=0)

def calc_index_representative(reg, reg_mean, stem, stem_mean, rad, rad_mean):
    ste = (stem[:, -1] - stem_mean[-1])**2 / stem_mean[-1]**2
    reg = (reg[:, -1] - reg_mean[-1])**2 / reg_mean[-1]**2
    rad = (rad[:, -1] - rad_mean[-1])**2 / rad_mean[-1]**2
    return np.argmin(0.5*reg + 0.3*ste + 0.2*rad)

def see_end():
    plt.style.use('default')
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2)
    
    ax1 = fig.add_subplot(gs[:, 0])
    # Fix the colormap creation
    cmap = LinearSegmentedColormap.from_list('cells', ['black', 'red'])  # Now properly imported
    cmap.set_under('white')
    cmap.set_over('yellow')
    
    # Rest of the function remains the same
    ax1.imshow(vect_tumor_snapshots[rep_idx][-1], cmap=cmap, vmin=1, vmax=P_MAX)
    ax1.set_title(f'Final Tumor (Day {t_max*dt:.1f})')
    
    t_vec = np.arange(t_max+1) * dt
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(t_vec, pop_stem_mean, label='STC')
    ax2.fill_between(t_vec, pop_stem_mean-pop_stem_std, pop_stem_mean+pop_stem_std, alpha=0.2)
    ax2.semilogy(t_vec, pop_reg_mean, label='RTC')
    ax2.fill_between(t_vec, pop_reg_mean-pop_reg_std, pop_reg_mean+pop_reg_std, alpha=0.2)
    ax2.legend()
    ax2.set_title('Population Dynamics')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_vec, tumor_rad_mean)
    ax3.fill_between(t_vec, tumor_rad_mean-tumor_rad_std, tumor_rad_mean+tumor_rad_std, alpha=0.2)
    ax3.set_title('Tumor Radius')
    plt.tight_layout()
    plt.show()

def see_snapshots():
    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    times = np.linspace(0, t_max, 8, dtype=int)
    for ax, t in zip(axs.flat, times):
        ax.imshow(vect_tumor_snapshots[rep_idx][t], cmap='Reds')
        ax.set_title(f'Day {t*dt:.1f}')
    plt.tight_layout()
    plt.show()

def see_chances():
    fig, ax = plt.subplots()
    t_vec = np.arange(t_max+1) * dt
    ax.plot(t_vec, vect_deat*100, label='Death')
    ax.plot(t_vec, vect_prol*100, label='Proliferation')
    ax.plot(t_vec, vect_potm*100, label='Migration')
    ax.set_yscale('log')
    ax.legend()
    plt.show()

def report_general():
    print(f"\n=== Simulation Report ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===")
    print(f"Runs: {sys_nruns} | Total time: {np.sum(sys_t_run):.2f}s")
    print(f"Final Populations (mean ± std):")
    print(f"STC: {pop_stem_mean[-1]:.0f} ± {pop_stem_std[-1]:.0f}")
    print(f"RTC: {pop_reg_mean[-1]:.0f} ± {pop_reg_std[-1]:.0f}")
    print(f"Radius: {tumor_rad_mean[-1]:.1f} ± {tumor_rad_std[-1]:.1f} units\n")

def save_files():
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    with open(f"tumor_sim_{date_str}.pkl", 'wb') as f:
        pickle.dump({
            'parameters': (param_cct, P_MAX, param_stem, param_potm),
            'pop_stem': vect_pop_stem,
            'pop_reg': vect_pop_reg,
            'radius': vect_rad
        }, f)
    print(f"Saved simulation data to tumor_sim_{date_str}.pkl")

sim = TumorSimulation()

# Main loop
while True:
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
    pygame.display.flip()
    clock.tick(frame_rate)