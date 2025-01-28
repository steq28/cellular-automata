import pygame
import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
CELL_SIZE = 10
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
STC_COLOR = (0, 255, 0)  # Green for Stem Cells
RTC_COLOR = (255, 0, 0)  # Red for Regular Tumor Cells

# Scenario Parameters
P_MAX = 20  # Maximum proliferation potential
P_A = 0  # Apoptosis probability
P_MIGRATION = 10  # Migration potential (10 cell width/day)
delta_t = 1 / 12  # Time step (in days)
CCT = 24  # Cell cycle time (in hours)
P_PROLIF = CCT * delta_t / 24  # Proliferation probability

# Simulation Parameters
running = False
days_elapsed = 0
frame_rate = 10  # Frames per second
population_dynamics = {"total": [], "stem": [], "regular": []}

# Initial Grid Size
INITIAL_SIZE = 11
MAX_SIZE = 400  # Maximum size of the matrix
EXPANSION_SIZE = 5  # Number of rows/columns to add when expanding the matrix

# Initialize Pygame
pygame.init()
screen_width, screen_height = 600, 600
screen = pygame.display.set_mode((screen_width, screen_height + 50))  # Extra space for controls/info
pygame.display.set_caption("Tumor Growth: Dynamic Matrix Expansion")
clock = pygame.time.Clock()

# Initialize Grid
def create_initial_matrix(size):
    """Create an initial tumor matrix with a single clonogenic stem cell."""
    tumor = np.zeros((size, size), dtype=np.int64)
    center = size // 2
    tumor[center, center] = P_MAX + 1  # Clonogenic stem cell
    return tumor, center

grid, tumor_center = create_initial_matrix(INITIAL_SIZE)

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

    # Draw grid lines
    for x in range(0, screen_width, cell_width):
        pygame.draw.line(screen, BLACK, (x, 0), (x, screen_height))
    for y in range(0, screen_height, cell_height):
        pygame.draw.line(screen, BLACK, (0, y), (screen_width, y))

def get_free_neighbors(matrix, i, j):
    """Find free neighbor positions for a cell at (i, j)."""
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    free_spots = []
    for ni, nj in neighbors:
        ni, nj = i + ni, j + nj
        if 0 <= ni < matrix.shape[0] and 0 <= nj < matrix.shape[1] and matrix[ni, nj] == 0:
            free_spots.append((ni, nj))
    return free_spots

def expand_matrix(matrix, size_increase):
    """Expand the tumor matrix by adding rows/columns and re-centering."""
    old_size = matrix.shape[0]
    new_size = old_size + size_increase * 2

    # Check if the new size exceeds the maximum size
    if new_size > MAX_SIZE:
        return matrix, old_size // 2

    # Create new larger matrix
    new_matrix = np.zeros((new_size, new_size), dtype=np.int64)

    # Copy old matrix to the center of the new one
    start = size_increase
    new_matrix[start:start + old_size, start:start + old_size] = matrix

    # Update the new center
    new_center = new_size // 2
    return new_matrix, new_center

def update_grid(matrix):
    """Perform one step of tumor growth dynamics."""
    global grid, tumor_center
    new_matrix = np.zeros_like(matrix, dtype=np.int64)
    rows, cols = matrix.shape
    stem_count = 0
    regular_count = 0

    cells = np.transpose(np.nonzero(matrix))  # Get all active cells
    needs_expansion = False

    for i, j in cells:
        cell = matrix[i, j]

        # Check proximity to borders
        if i == 1 or i == rows - 2 or j == 1 or j == cols - 2:
            needs_expansion = True

        # Clonogenic Stem Cell
        if cell == P_MAX + 1:
            stem_count += 1
            free_spots = get_free_neighbors(matrix, i, j)
            if random.random() < P_PROLIF and free_spots:
                ni, nj = random.choice(free_spots)
                new_matrix[ni, nj] = P_MAX  # Daughter cell with limited proliferation potential
            new_matrix[i, j] = cell  # Stem cell remains unchanged

        # Regular Tumor Cells
        elif 1 <= cell <= P_MAX:
            regular_count += 1

            # Proliferation
            free_spots = get_free_neighbors(matrix, i, j)
            if random.random() < P_PROLIF and free_spots:
                ni, nj = random.choice(free_spots)
                new_matrix[ni, nj] = max(1, cell - 1)  # Daughter cell
                new_matrix[i, j] = max(1, cell - 1)  # Parent cell loses potential
                continue

            # Migration
            if random.random() < P_MIGRATION * delta_t and free_spots:
                ni, nj = random.choice(free_spots)
                new_matrix[ni, nj] = cell
                continue

            # Quiescence
            new_matrix[i, j] = cell

    # Expand matrix if necessary
    if needs_expansion:
        new_matrix, tumor_center = expand_matrix(new_matrix, EXPANSION_SIZE)

    # Update population dynamics
    population_dynamics["stem"].append(stem_count)
    population_dynamics["regular"].append(regular_count)
    population_dynamics["total"].append(stem_count + regular_count)

    return new_matrix

def reset_grid():
    """Reset the grid and population dynamics."""
    global grid, days_elapsed, population_dynamics, tumor_center
    grid, tumor_center = create_initial_matrix(INITIAL_SIZE)
    days_elapsed = 0
    population_dynamics = {"total": [], "stem": [], "regular": []}

# Button setup
start_button = pygame.Rect(10, screen_height + 10, 80, 30)
stop_button = pygame.Rect(100, screen_height + 10, 80, 30)
reset_button = pygame.Rect(190, screen_height + 10, 80, 30)

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

def plot_population_dynamics():
    """Plot the population dynamics after the simulation ends."""
    plt.figure(figsize=(10, 6))
    plt.plot(population_dynamics["total"], label="Total Cells")
    plt.plot(population_dynamics["stem"], label="Stem Cells")
    plt.plot(population_dynamics["regular"], label="Regular Cells")
    plt.xlabel("Time Steps")
    plt.ylabel("Cell Count")
    plt.title("Population Dynamics")
    plt.legend()
    plt.show()

# Initialize the grid
reset_grid()

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            plot_population_dynamics()  # Show the final plot
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if start_button.collidepoint(event.pos):
                running = True
            elif stop_button.collidepoint(event.pos):
                running = False
            elif reset_button.collidepoint(event.pos):
                reset_grid()

    if running:
        grid = update_grid(grid)
        days_elapsed += delta_t  # Increment simulation days

    # Draw everything
    screen.fill(WHITE)
    draw_grid()
    draw_buttons()
    draw_info()
    pygame.display.flip()
    clock.tick(frame_rate)