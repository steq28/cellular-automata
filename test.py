import itertools
import pickle
import random
import sys
from datetime import datetime
from timeit import default_timer as timer

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pygame
from matplotlib.colors import LinearSegmentedColormap

# Keep original constants
CELL_SIZE = 10
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
STC_COLOR = (255, 255, 0)
RTC_COLOR = (255, 0, 0)

OXYGEN_DIFFUSION_RATE = 0.1  # D in diffusion equation
OXYGEN_CONSUMPTION_RATE = 0.05  # γ in consumption term
OXYGEN_SOURCE = 0.2  # Background oxygen level
BLOOD_VESSEL_RADIUS = 3  # Radius of central blood vessel
HYPOXIC_THRESHOLD = 0.15  # Oxygen level for hypoxia effects

# Simulation Parameters
running = False
days_elapsed = 0
delta_t = 1 / 12
frame_rate = 10  # Frames per second
population_dynamics = {"total": [], "stem": [], "regular": []}
t_max=1000 #Total ammount of steps dt
dt = 1/12 # Time step size (fraction of a day) 

# Initial Grid Size

INITIAL_SIZE = 50
MAX_SIZE = 400  # Maximum size of the matrix
EXPANSION_SIZE = 5  # Number of rows/columns to add when expanding the matrix

# Model parameters
param_cct = 24 #cell cicle time (hours)
P_MAX = 15 #proliferation potential of regular tumor cell (number of divisions unitl death + 1)
param_stem = P_MAX+1 #stem cells have superior proliferation potential (and dont die)
param_potm = 1 #migration potential in cell width  per day
vect_deat,vect_prol,vect_potm,vect_stem = (np.empty(t_max+1) for i in range(4)) #create empty vectors for time-variable chances
vect_deat[:round(0.5*t_max)]=0.01*dt; vect_deat[round(0.5*t_max):]=0.01*dt #chance of death changing w/ time
vect_prol[:] =  (24/param_cct*dt) # Chance of proliferation 
vect_potm[:round(0.4*t_max)] = 10*dt; vect_potm[round(0.4*t_max):] = 10*dt; #Chance of migration changing w/ time
vect_stem[:] = 0.1 #Probability of creating a daughter stem cell

#System configuration
sys_nruns=3
sys_t_run = [] #Number of random runs and vector for runtimes
sys_visu_partial=False; sys_visu_end=True; sys_print=True; sys_report=False; sys_save=False; #Control of plot/print generation and file saving

vect_pop_stem = np.empty((sys_nruns,t_max+1), dtype=np.int64)  #creation of empty vectors for populations and radius
vect_pop_reg = np.empty((sys_nruns,t_max+1), dtype=np.int64)
vect_rad = np.empty((sys_nruns,t_max+1), dtype=np.int64)
vect_tumor_snapshots = [None]*(sys_nruns) #Creating an empty list for receiving all snapshots

neighbors = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],dtype=int) #Coordinates for all 8 possible neighbors
neighbors_success = np.array([0, 0, 0], dtype=np.int64) #returning vector of check_spots when a free spot is found
neighbors_fail = np.array([1, 0, 0], dtype=np.int64) #returning vector of check_spots when no free spots are found
neighbors_permutation = list(itertools.permutations(np.arange(0,8, dtype=int))) #all possible combinations of 0-7
neighbors_all = np.array(neighbors_permutation[:]) #transforming tuple into an array

np.random.seed(6) #use only when trying to control results for verification purposes
random.seed(6)

# Initialize Pygame
pygame.init()
screen_width, screen_height = 600, 600
screen = pygame.display.set_mode((screen_width, screen_height + 50))  # Extra space for controls/info
pygame.display.set_caption("Tumor Growth: Dynamic Matrix Expansion")
clock = pygame.time.Clock()
start_button = pygame.Rect(10, screen_height + 10, 80, 30)
stop_button = pygame.Rect(100, screen_height + 10, 80, 30)
reset_button = pygame.Rect(190, screen_height + 10, 80, 30)

def create_initial_matrix(size):
    """Create an initial tumor matrix with a single clonogenic stem cell."""
    tumor = np.zeros((size, size), dtype=np.int64)
    center = size // 2
    tumor[center, center] = P_MAX + 1  # Clonogenic stem cell
    return tumor, center

def draw_grid():
    """Draw the grid on the screen."""
    rows, cols = grid.shape
    cell_width = screen_width // cols
    cell_height = screen_height // rows

    for row in range(rows):
        for col in range(cols):
            cell = grid[row, col]
            if cell == 0:
                color = WHITE
            elif cell == P_MAX + 1:  # Stem Cell
                color = STC_COLOR
            elif 1 <= cell <= P_MAX:  # Regular cells
                intensity = int(255 * (cell / P_MAX))  # Intensity based on proliferation potential
                color = (intensity, 0, 0)  # Shades of red
            pygame.draw.rect(screen, color, (col * cell_width, row * cell_height, cell_width, cell_height))

    alpha_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    rows, cols = oxygen.shape
    cell_w = screen_width // cols
    cell_h = screen_height // rows

    alpha_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
    rows, cols = oxygen.shape
    cell_w = screen_width // cols
    cell_h = screen_height // rows
    
    for row in range(rows):
        for col in range(cols):
            oxy = oxygen[row,col]
            alpha = int(oxy * 100)
            pygame.draw.rect(alpha_surface, (0, 0, 255, alpha), 
                            (col*cell_w, row*cell_h, cell_w, cell_h))
    
    screen.blit(alpha_surface, (0,0))

    # # Draw grid lines
    # for x in range(0, screen_width, cell_width):
    #     pygame.draw.line(screen, BLACK, (x, 0), (x, screen_height))
    # for y in range(0, screen_height, cell_height):
    #     pygame.draw.line(screen, BLACK, (0, y), (screen_width, y))

def draw_buttons():
    """Draw control buttons."""
    pygame.draw.rect(screen, (0, 255, 0), start_button)
    pygame.draw.rect(screen, (255, 0, 0), stop_button)
    pygame.draw.rect(screen, (0, 0, 255), reset_button)

    font = pygame.font.SysFont(None, 24)
    screen.blit(font.render("Start", True, BLACK), (start_button.x + 10, start_button.y + 5))
    screen.blit(font.render("Stop", True, BLACK), (stop_button.x + 10, stop_button.y + 5))
    screen.blit(font.render("Reset", True, BLACK), (reset_button.x + 10, reset_button.y + 5))

def draw_info():
    """Display simulation information."""
    font = pygame.font.SysFont(None, 24)
    info_text = f"Days Elapsed: {days_elapsed:.2f}"
    screen.blit(font.render(info_text, True, BLACK), (screen_width - 150, screen_height + 15))

def reset_grid():
    """Reset the grid and population dynamics."""
    global grid, days_elapsed, population_dynamics, tumor_center
    grid, tumor_center = create_initial_matrix(INITIAL_SIZE)
    days_elapsed = 0
    population_dynamics = {"total": [], "stem": [], "regular": []}

def create_tumor(param_stem):
    """From given param_stem creates and returns the tumor matrix""" 
    tumor_n = INITIAL_SIZE  # Use INITIAL_SIZE instead of hardcoded 11
    tumor_center = tumor_n // 2
    tumor = np.zeros((tumor_n, tumor_n), dtype=np.int64)
    tumor[tumor_center][tumor_center] = param_stem
    return tumor, tumor_n, tumor_center
def order_sweep(tumor_backup):
    """ Returns a random sweeping order to accesss cord (rows, columns) for tumor_backup matrix"""
    i,j = np.matrix.nonzero(tumor_backup[1:-1,1:-1]) #returns all coordinates i,j where there is a tumor cell
    cord = np.transpose([i,j])+1 #transpose to get an array of coordinates i,j for all cells
    order = np.arange(0,len(cord)) #create a vector going from 0 to N (number of cells)
    np.random.shuffle(order) #Randomly change the vector "order" (i.e. 0, 1,2.. -> 77, 34, ...)
    return cord, order #returns the coordinates indicating where there are cells and a random order to sweep them
def calc_chance(cord, order):
    """ Calculate order-sized arrays with random chances """
    cord_n = len(cord) #get number of tumor cells
    tumor_death = np.random.rand(cord_n); tumor_prolif = np.random.rand(cord_n);  tumor_migr = np.random.rand(cord_n); #calculate vectors of random chances for each ocasion and cell
    return tumor_death, tumor_prolif, tumor_migr #return chance arrays

neighbors_permutation = list(itertools.permutations(np.arange(0, 8, dtype=int)))
neighbors_all = np.array(neighbors_permutation[:])
np.random.shuffle(neighbors_all)  # Shuffle once at the start

def calc_free_spots(A, i, j):
    """Picks free neighbours of cell at i,j from matrix A and returns a random one as 1,i,j"""
    # Randomly pick one of the pre-shuffled neighbor orders
    neighbors_order = neighbors_all[np.random.randint(0, len(neighbors_all))]
    
    # Iterate through the selected neighbor order
    for n in neighbors_order:
        ii, jj = neighbors[n]  # Get coordinates
        if A[i + ii, j + jj] == 0:  # Check if it is empty
            neighbors_success[1], neighbors_success[2] = ii, jj  # If yes, get coordinates of the empty space
            return neighbors_success  # Return coordinates of the free spot
    
    return neighbors_fail


def calc_rad(i, j, radius):
    """Calculate maximum distance from original center"""
    global original_center  # This should be defined in your initialization
    dx = i - original_center[0]
    dy = j - original_center[1]
    return max(np.sqrt(dx**2 + dy**2), radius)

# Replace original functions with optimized versions
def main_ca(tumor, tumor_rad_new, oxygen):
    cord, order = order_sweep(tumor)
    tumor_death, tumor_prolif, tumor_migr = calc_chance(cord, order)
    
    for n in order:
        i, j = cord[n]
        cell_value = tumor[i][j]
        current_oxygen = oxygen[i,j]
        
        # Modify probabilities based on oxygen level
        oxygen_factor = current_oxygen / HYPOXIC_THRESHOLD
        hypoxia_factor = max(0, 1 - oxygen_factor)
        
        # Adjusted probabilities
        adj_death = chance_death * (1 + 2*hypoxia_factor)
        adj_proliferation = chance_proliferation * min(1, oxygen_factor)
        adj_migration = chance_migration * (1 + hypoxia_factor)
        
        # Death check
        if cell_value < param_stem and tumor_death[n] <= adj_death:
            tumor[i][j] = 0
            continue
            
        # Proliferation and migration
        if tumor_prolif[n] <= adj_proliferation or tumor_migr[n] <= adj_migration:
            spots_cache = calc_free_spots(tumor, i, j)
            if spots_cache[0] == 0:  # Free adjacent spots exist
                new_i = i + spots_cache[1]
                new_j = j + spots_cache[2]
                
                if tumor_prolif[n] <= adj_proliferation:
                    # Proliferation case
                    if cell_value == param_stem:
                        tumor[new_i][new_j] = param_stem if random.random() <= chance_stem else P_MAX
                    else:
                        tumor[i][j] -= 1
                        tumor[new_i][new_j] = tumor[i][j]
                else:
                    # Migration case
                    tumor[new_i][new_j] = cell_value
                    tumor[i][j] = 0

                # Update oxygen level after proliferation/migration
                oxygen[i,j] = max(0, oxygen[i,j] - 0.1)
                
                # Update radius
                tumor_rad_new = calc_rad(new_i, new_j, tumor_rad_new)
    
    return tumor, tumor_rad_new, oxygen

def calc_pop(tumor_snap, param_stem):
    """Population calculation"""
    pop_stem = np.sum(tumor == param_stem)
    pop_tot = np.sum(tumor > 0)
    pop_reg = pop_tot - pop_stem
    return pop_tot, pop_stem, pop_reg


INITIAL_SIZE = 50
EXPANSION_BUFFER = 5  # Expand when within 5 cells of edge
current_grid_size = INITIAL_SIZE
original_center = (INITIAL_SIZE//2, INITIAL_SIZE//2)

def create_extension(tumor, oxygen, current_size):
    """Expand both grids symmetrically"""
    global original_center  # Access the mutable center position
    expand_by = EXPANSION_SIZE
    new_size = current_size + 2 * expand_by
    
    # Pad tumor with zeros
    tumor = np.pad(tumor, [(expand_by, expand_by), (expand_by, expand_by)], 
                   mode='constant', constant_values=0)
    
    # Pad oxygen with background value
    oxygen = np.pad(oxygen, [(expand_by, expand_by), (expand_by, expand_by)],
                    mode='constant', constant_values=OXYGEN_SOURCE)
    
    # Update original center coordinates
    original_center[0] += expand_by
    original_center[1] += expand_by
    
    return tumor, oxygen, new_size

def calc_mean(aux_vector):
    """ Return mean and std from an input vector"""
    aux_mean = np.average (aux_vector, axis=0)
    aux_std = np.std(aux_vector, axis=0)
    return aux_mean, aux_std

def calc_index_representative(vect_pop_reg, pop_reg_mean, vect_pop_stem, pop_stem_mean, vect_rad, tumor_rad_mean):
    """Find the index of the most representative simulation run of the average values"""
    ste = (vect_pop_stem[:,-1]-pop_stem_mean[-1])**2/pop_stem_mean[-1]**2 #calc stem relative difference from mean (for every n)
    reg = (vect_pop_reg[:,-1]-pop_reg_mean[-1])**2/pop_reg_mean[-1]**2 #calc reg relative difference from mean (for every n)
    rad = (vect_rad[:,-1]-tumor_rad_mean[-1])**2/tumor_rad_mean[-1]**2 #calc maxrad relative difference from mean (for every n)
    k_rep = np.argmin(0.5*reg + 0.3*ste + 0.2*rad) #identifies the minimal error according to ponderation (50% stc, 30% rtc, 20% dispersion)

    return k_rep #return the index aka the n_run number of the most representative tumor

def see_end():
    """General report / visualization - no input"""    
    plt.style.use('default') #Plot style as default
    gs = gridspec.GridSpec(2,2) #Create a 2x2 grid for plots
    fig1 = plt.figure(figsize=(15,9)) #create a figure
    ax1 = fig1.add_subplot(gs[:,0]) #One big plot on the left
    ax2 = fig1.add_subplot(gs[0,1]) #Second plot on the right
    ax4 = fig1.add_subplot(gs[1,1]) #Third plot on the right
    fig1.suptitle('General Report (averages for '+str(sys_nruns)+' runs)',fontsize=14)    #Tumor spatial view
    cmap = LinearSegmentedColormap.from_list('name', ['black', 'red']) #Cmap for cells distinguishment
    cmap.set_under('white'); cmap.set_over('yellow') #Limits of colors
    tumor_2D = ax1.imshow(tumor_snapshots[t_max], interpolation='nearest', vmin=1, vmax=P_MAX, cmap=cmap, origin='lower') #OPick the last spatial snap for plot
    ax1.set_title('Tumor 2D-View (t={:.1f} days, n={:d})'.format(t_max*dt,tumor_representative_index)) #Title 
    ax1.set_xlabel(r'1 Lattice area = $100\mu m^2$') #Label
    cbar = fig1.colorbar(tumor_2D, ax=ax1, extend='both', extendrect='false', shrink=0.75, ticks=[0.75,0.25*P_MAX,0.5*P_MAX,0.75*P_MAX,param_stem-0.75], orientation='vertical')
    cbar.ax.set_yticklabels(['Empty','Low pot.', 'Avg pot.', 'High pot.','STCs'])  # horizontal colorbar
    #Dynamic population view
    ax2.plot(t_vector*dt, pop_reg_mean, label='RTCs') 
    ax2.fill_between(t_vector*dt, (pop_reg_mean-pop_reg_std), (pop_reg_mean+pop_reg_std), alpha=0.4)
    ax2.plot(t_vector*dt, pop_stem_mean, label='STCs')
    ax2.fill_between(t_vector*dt, (pop_stem_mean-pop_stem_std), (pop_stem_mean+pop_stem_std), alpha=0.4)
    ax2.set_ylim(0.9,)
    ax2.set_yscale('log')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Cell number')
    ax2.set_title('Cell population dynamics')
    ax2.legend()
    #Tumor Radius
    ax4.fill_between(t_vector, (tumor_rad_mean-tumor_rad_std), (tumor_rad_mean+tumor_rad_std), alpha=0.4)
    ax4.plot(t_vector,tumor_rad_mean)
    ax4.set_xlabel('Time steps')
    ax4.set_ylabel(r'Distance from the center ($\times 10 \mu m$)')
    ax4.set_title('Maximum distance from the center')
    plt.show()
    return fig1
def see_partial():
    """General report / visualization - no input"""    
    plt.style.use('default')
    gs = gridspec.GridSpec(2,2)
    fig1 = plt.figure(figsize=(15,9))#figsize=(18,11), dpi=80)
    ax1 = fig1.add_subplot(gs[:,0])
    ax2 = fig1.add_subplot(gs[0,1])
    ax4 = fig1.add_subplot(gs[1,1])
    fig1.suptitle('General Report (n='+str(k)+')',fontsize=14)
    #Tumor spatial view
    cmap = LinearSegmentedColormap.from_list('name', ['black', 'red'])
    cmap.set_under('white'); cmap.set_over('yellow')
    tumor_2D = ax1.imshow(tumor, interpolation='nearest', vmin=1, vmax=P_MAX, cmap=cmap, origin='lower')
    ax1.set_title('Tumor 2D-View (t=%i days)' %(t_max*dt))
    ax1.set_xlabel(r'1 Lattice area = $100\mu m^2$')
    cbar = fig1.colorbar(tumor_2D, ax=ax1, extend='both', extendrect='false', shrink=0.75, ticks=[0.75,0.25*P_MAX,0.5*P_MAX,0.75*P_MAX,param_stem-0.75], orientation='vertical')
    cbar.ax.set_yticklabels(['Empty','Low pot.', 'Avg pot.', 'High pot.','STCs'])  # horizontal colorbar
    #Dynamic population view
    ax2.plot(t_vector*dt, pop_reg, label='RTCs')
    ax2.plot(t_vector*dt, pop_stem, label='STCs')
    ax2.set_ylim(0.9,)
    ax2.set_yscale('log')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Cell number')
    ax2.set_title('Cell population dynamics')
    ax2.legend()
    #Tumor Radius
    ax4.plot(t_vector,tumor_rad)
    ax4.set_xlabel('Time steps')
    ax4.set_ylabel(r'Distance from the center ($\times 10 \mu m$)')
    ax4.set_title('Maximum distance from the center')
    plt.show()
def see_snapshots():    
    """ Exhibits 8 snapshots of tumor evolution - no input"""
    stampstep = t_max//8 #get 8 equally-spaced snaps
    aux_max = len(tumor_snapshots[-1])
    plt.style.use('default')
    fig2, ax = plt.subplots(nrows=2, ncols=4, figsize=(14,8))
    cmap = LinearSegmentedColormap.from_list('name', ['black', 'red'])
    cmap.set_under('white'); cmap.set_over('yellow')
    for k in range(1,9): #axs.flat[k] gets each subplot k in order and equally spaced acacording to nrows, ncols
        aux_tumor = tumor_snapshots[k*stampstep].copy() #copy the tumor evolution of all snapshots
        aux_n = len(aux_tumor); aux_nc = aux_n//2 #get the matrix size of the final tumor
        if aux_n < aux_max: #for all sizes smaller than the biggest tumor matrix
            aux_tumor,aux_n,aux_nc = create_extension(aux_tumor, aux_n, aux_nc, (aux_max-aux_n)//2)  #extend tumor matrices to match the maximum size
        ax.flat[k-1].imshow(aux_tumor, interpolation='nearest', vmin=1, vmax=P_MAX, cmap=cmap, origin='lower') #plot
        ax.flat[k-1].set_title('t=%i days' %(k*stampstep*dt)) #Showing a timestamp in days for each snapshot
    plt.show()
    return fig2
def report_general():
    """ Prints on the screen a small report with crucial details and outputs of the simulation """
    print('========================== SIMPLE REPORT ========================== \n')
    print('N# simulations:',sys_nruns,'   Date:',datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print('\nSystem information (runtime)')
    print('Total:{:.3f}  Average:{:.3f}  Max:{:.3f}  Min:{:.3f}   STD:{:.3f} '.format(sys_nruns*np.average(sys_t_run),np.average(sys_t_run),np.max(sys_t_run),np.min(sys_t_run),np.std(sys_t_run)))
    print('\nModel information:')
    print('Time steps:{:d} Step length:{:.3f} Time spam (days):{:.3f}'.format(t_max,dt,dt*t_max))
    print('CCT (hours):{:d}  Proliferation potential:{:.0f}  Migration potential:{:.0f}'.format(param_cct, P_MAX, param_potm))    
    print('\nProbability information')
    print('Death:{:.2f} Proliferation:{:.2f} Migration:{:.2f} Stem generation:{:.2f}'.format(chance_death,chance_proliferation,chance_migration,chance_stem))
    print('\nGeneral results: (at t =',t_max*dt,')')
    for kk in range(0,sys_nruns):
        print('N:{:d} STC:{:d} RTC:{:d} Max Radius:{:d}'.format(kk, vect_pop_stem[kk,t_max], vect_pop_reg[kk,t_max], vect_rad[kk, t_max]))
    print('\nAverage results: (at t =',t_max*dt,')')
    print('STC:{:.3f} +-{:.3f}   (Max:{:.3f}, Min:{:.3f})'.format(pop_stem_mean[t_max],pop_stem_std[t_max],max(vect_pop_stem[:,t_max]),min(vect_pop_stem[:,t_max])))
    print('RTC:{:.3f} +-{:.3f}   (Max:{:.3f}, Min:{:.3f})'.format(pop_reg_mean[t_max],pop_reg_std[t_max],max(vect_pop_reg[:,t_max]),min(vect_pop_reg[:,t_max])))
    print('Max Radius:{:.3f} +- {:.3f}  (Max:{:.3f}, Min:{:.3f})'.format(tumor_rad_mean[t_max], tumor_rad_std[t_max], max(vect_rad[:,t_max]), min(vect_rad[:,t_max])))
def save_files():
    """Saves plots and report of the simulation. Also saves variables if desired"""
    date_string = datetime.now() #get current date
    date_string = date_string.strftime("_%Y-%m-%d_%H-%M") #convert to string name
    filename = 'n'+str(sys_nruns)+'_tmax'+str(t_max)+date_string #create filename 
    original_stdout = sys.stdout #make a copy of original sys.stdout
    with open('Data/'+filename+'.txt',"w") as file: #open a txt file with filename data
        sys.stdout = file #change the output of the console to this file
        report_general() #run report print function again, but this time to the file
        sys.stdout = original_stdout #return stdout as before
    fig_general.savefig('Data/'+filename+'_overview.pdf', format = 'pdf') #saves general report fig to  a file
    fig_evolution.savefig('Data/'+filename+'_evolution.pdf', format = 'pdf') # saves tumor evolution fig to another file
    fig_chances.savefig('Data/'+filename+'_chances.pdf', format = 'pdf') # saves tumor evolution fig to another file   
    if sys_save == True: #if switch save is on:
        var_parameters = [sys_nruns,t_max, dt,param_cct, P_MAX, param_stem, param_potm,chance_death, chance_proliferation, chance_migration, chance_stem]
        with open('Data/'+filename+'_param.dat', 'wb') as f: #create dat file
            pickle.dump(var_parameters, f)
        with open('Data/'+filename+'_data.dat', 'wb') as f: #create dat file
            pickle.dump([vect_tumor_snapshots, vect_rad], f) #store vect_tumor_snapshots (from which everything can be derived later)
def see_chances():
    """ Returns figure w/ simple plot of time-variable changes - No input"""
    plt.style.use('default')
    fig, ax = plt.subplots()
    ax.plot(t_vector*dt, vect_deat*100, label='Chance of apoptosis')
    ax.plot(t_vector*dt, vect_prol*100, label='Chance of proliferation')
    ax.plot(t_vector*dt, vect_potm*100, label='Chance of migration')
    ax.plot(t_vector*dt, vect_stem*100, label='Chance of STC creation')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Chance in every iteration (%)')
    ax.set_yscale('log')
    ax.legend()
    plt.show()
    return fig

# Add this new constant with other parameters
NUM_BLOOD_VESSELS = 5  # Number of random oxygen sources

def create_initial_oxygen(size):
    """Create initial oxygen grid with randomly placed blood vessels."""
    oxygen = np.ones((size, size)) * OXYGEN_SOURCE
    
    # Create random blood vessel sources
    for _ in range(NUM_BLOOD_VESSELS):
        # Random center coordinates
        center_x = np.random.randint(0, size)
        center_y = np.random.randint(0, size)
        
        # Set maximum oxygen in circular region around the random center
        for i in range(size):
            for j in range(size):
                if np.sqrt((i-center_x)**2 + (j-center_y)**2) <= BLOOD_VESSEL_RADIUS:
                    oxygen[i,j] = 1.0  # Maximum oxygen at blood vessel
                    
    return oxygen

def calculate_oxygen_diffusion(oxygen, tumor):
    """Calculate oxygen diffusion using finite differences."""
    if oxygen.shape != tumor.shape:
        raise ValueError(f"Oxygen and tumor grids must have the same shape. Oxygen: {oxygen.shape}, Tumor: {tumor.shape}")
    
    new_oxygen = np.copy(oxygen)
    D = OXYGEN_DIFFUSION_RATE
    γ = OXYGEN_CONSUMPTION_RATE
    
    # Finite difference calculation
    new_oxygen[1:-1, 1:-1] = oxygen[1:-1, 1:-1] + D * (
        oxygen[2:, 1:-1] + oxygen[:-2, 1:-1] +
        oxygen[1:-1, 2:] + oxygen[1:-1, :-2] -
        4*oxygen[1:-1, 1:-1]
    ) - γ * oxygen[1:-1, 1:-1] * (tumor[1:-1, 1:-1] > 0)
    
    # Boundary conditions (zero flux)
    new_oxygen[0,:] = new_oxygen[1,:]
    new_oxygen[-1,:] = new_oxygen[-2,:]
    new_oxygen[:,0] = new_oxygen[:,1]
    new_oxygen[:,-1] = new_oxygen[:,-2]
    
    return np.clip(new_oxygen, 0, 1)

if __name__ == "__main__":

    INITIAL_SIZE = 50
    current_grid_size = INITIAL_SIZE
    original_center = [INITIAL_SIZE//2, INITIAL_SIZE//2] 
    grid, tumor_center = create_initial_matrix(INITIAL_SIZE)
    oxygen = create_initial_oxygen(INITIAL_SIZE)

    "System run"
    for k in range (0,sys_nruns): #Loop for system runs
        "System initialization"
        tumor, tumor_n, tumor_center = create_tumor(param_stem)
        sys_t_start = timer() # Register simulation start time
        t = 0 #Start initial time step
        t_count = 0 #count for printing progress bar
        t_vector = np.array(range(t_max+1)) # Create time vector
        tumor, tumor_n, tumor_center = create_tumor(param_stem) #Create tumor matrix
        tumor_snapshots = [None]*(t_max+1); tumor_snapshots[0] = tumor.copy() #Creating empty list to receive snapshots of tumor 2D evolution
        tumor_rad = np.empty(t_max+1); tumor_rad[0] = 0 #Create radius vector and initialize vector = 0
        days_elapsed = 0 #initialize days elapsed
        "Time loop"
        for t in range(1,t_max+1): #Loop for every time step until t_max
            chance_death, chance_proliferation, chance_migration, chance_stem = vect_deat[t],vect_prol[t],vect_potm[t],vect_stem[t]  #Probabilities inside the loop (i.e.they may change with t)
            current_radius = tumor_rad[t-1]
            if current_radius > current_grid_size // 2 - EXPANSION_BUFFER:
                tumor, oxygen, current_grid_size = create_extension(tumor, oxygen, current_grid_size)
            tumor_snapshots[t]=tumor.copy() #Copy tumor matrix at this time instant
            oxygen = calculate_oxygen_diffusion(oxygen, tumor)
            tumor, tumor_rad[t], oxygen = main_ca(tumor, tumor_rad[t-1], oxygen)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if start_button.collidepoint(event.pos):
                        running = True
                    elif stop_button.collidepoint(event.pos):
                        running = False
                    elif reset_button.collidepoint(event.pos):
                        reset_grid()

            grid = tumor.copy()
            days_elapsed += delta_t  # Increment simulation days

            # Draw everything
            screen.fill(WHITE)
            draw_grid()
            draw_buttons()
            draw_info()
            pygame.display.flip()
            clock.tick(frame_rate)


            if t%((t_max)//10)==0: # for every 10% of t
                t_count += 10 
                print('\r Progress = [%d %%] \r'%t_count, end="") #Print %progress 10-10 for each iteration 
        "Population dynamics"
        pop_tot, pop_stem, pop_reg = calc_pop(tumor,param_stem) #count population
        vect_pop_stem[k] = pop_stem; vect_pop_reg[k] = pop_reg; vect_rad[k] = tumor_rad #save pop and radius vectors
        vect_tumor_snapshots[k] = tumor_snapshots #save snapshots for this run
        "System calculations"
        sys_t_end = timer() #Get the end time for this sytem run
        sys_t_run.append(sys_t_end - sys_t_start) #Append to the runtime vector the current runtime
        if sys_visu_partial==True: see_partial() #Check if user wants to see partial reports
        print('N={:d}/{:d} at t={:.3f}s (dur={:.3f}s)'.format(k+1, sys_nruns, np.sum(sys_t_run), sys_t_run[k])) #Print the status after each iteration ends
        
    "Average / representative dynamics"
    pop_stem_mean, pop_stem_std = calc_mean(vect_pop_stem) #calc the mean stem time series
    pop_reg_mean, pop_reg_std = calc_mean(vect_pop_reg) #calc the mean reg time series
    tumor_rad_mean, tumor_rad_std = calc_mean(vect_rad) #calc the mean radius time series
    tumor_representative_index = calc_index_representative(vect_pop_reg, pop_reg_mean, vect_pop_stem, pop_stem_mean, vect_rad, tumor_rad_mean)
    tumor_snapshots = vect_tumor_snapshots[tumor_representative_index] #Find and save the snapshots that are closer to the mean 


    "Visualization"
    if sys_visu_end==True:
        fig_general = see_end() #General report of last run 
        fig_evolution = see_snapshots() #Tumor snapshots of last run
        fig_chances = see_chances() #Chances changing with time

    "System saving / storing / printing"
    if sys_print == True: report_general() #If print switch is on: print simple report
    if sys_report == True: save_files() #If save switch is on: save report files

    print('\007') #Sound alert when the simulation is finished.