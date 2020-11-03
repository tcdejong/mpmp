# Small code project for http://think-maths.co.uk/catsanddogs

# Puzzle for submission: How many ways can you completely fill your ten kennels, 
# using only cats and dogs (one animal per kennel), such that no two cats are in adjacent kennels? 

# Functions are ordered alphabetically. 

from itertools import permutations
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool
import seaborn as sns
import tqdm 

CAT = True
DOG = False
OUT_RES_Y = 1080


def brute_force(kennels=8):
    """
    DEPRECATED: Refer to recursive() instead. 
    ---
    Brute-force solution for determining the number of valid cats-and-dogs arrangements.
    Hugely limited because permutations(base_list) is slow. But at least it works.
    """

    arrangements = 0

    for n_cats in range(kennels//2+1):
        n_dogs = kennels - n_cats
        base_list = [CAT for _ in range(0, n_cats)] + [DOG for _ in range(n_dogs)]
        valid_orders = [order for order in set(permutations(base_list)) if is_valid(order)] # Every cat is the same - don't tell the owners, the might get mad.

        arrangements += len(valid_orders)

        print_list(valid_orders, label=f"Valid permutations for {n_cats=}", lim=None, wrapper=cat_dog_wrapper)

    print('\n-----------\n')
    print(f'{arrangements=}')


def cat_dog_wrapper(x):
    return ['C' if val else '_' for val in x]


def convert_to_path(recursive_result: list) -> list:
    """
    Convert a list (in which True represents placing a cat and False a dog)
    to a list of coordinates in a cat-dog grid, where a step along the
    cat axis represents placing a cat and a step along the dog axis
    represents placing a dog. Obviously.
    """
    x_dog = 0
    y_cat = 0

    path = [None for _ in recursive_result]

    for index, animal in enumerate(recursive_result):
        if animal == CAT:
            y_cat += 1
        else:
            x_dog += 1
        
        coord = (x_dog, y_cat)
        path[index] = coord

    return [(0,0), *path]


def is_valid(arrangement):
    for a,b in pairwise(arrangement):
        if a == b == CAT:
            return False
    
    return True


def pairwise(iterable):
    for a, b in zip(iterable, iterable[1:]):
        yield a,b


def plot_all_and_save(arrangements):
    # Cancel if list is empty
    if not arrangements:
        return
    
    # Create directory 'output' if it does not exist
    out_dir = Path('output')
    if not out_dir.exists():
        out_dir.mkdir()

    # Within 'output', create folder for this number of kennels
    kennels = len(arrangements[0])
    out_dir = out_dir / f'{kennels}-kennels'
    if not out_dir.exists():
        out_dir.mkdir()
    
    # Determine number of zeros to pad the counter in plot file names
    n_arrangements = len(arrangements)
    zeros = 0
    while n_arrangements > 10**(zeros):
        zeros += 1

    # Generate function arguments for over-engineered multiprocessing loop
    func_args = ((convert_to_path(arrangement),  out_dir / str(n+1).zfill(zeros)) for n, arrangement in enumerate(arrangements)) # starmap/imap version

    # Plot in parallel
    with Pool() as p:
        list(
            tqdm.tqdm(
                p.imap_unordered(plot_arrangement, func_args),
                total=len(arrangements),
                smoothing=0,
                unit="plot"
            )
        )



def plot_arrangement(args):
    """
    Plot a single cat-dog arrangement on a n*n Cat-Dog grid. 
    args is an iterable containing the arrangement as a list of (x, y) tuples and an output path.
    The output path should include the plot's filename. 
    The extension is not required - matplotlib defaults to png.
    """

    # Unpack args - required for parallel execution
    assert len(args) == 2
    path, out_path = args


    # interpret data
    x, y = zip(*path)
    kennels = len(path) - 1 # account for point (0,0)
    max_dogs = kennels
    max_cats = (kennels + 1) // 2    

    # Generate background point grid
    padding = 0.5
    ax_min = -padding
    ax_max_dog = max_dogs + padding
    ax_max_cat = max_cats + padding

    grid_points = [(x, y) for x in range(0, max_dogs+1) for y in range(0, max_cats+1) if x + y <= kennels and y <= x + 1]
    grid_x, grid_y = zip(*[(x,y) for x,y in grid_points])

    sns.set_context("poster", font_scale=1)
    # sns.set_style("dark")

    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_title("Kennel Arrangement as path on Cat-Dog Grid")
    ax.set_xlabel("Dogs")
    ax.set_ylabel("Cats")
    ax.set_xlim(ax_min, ax_max_dog)
    ax.set_ylim(ax_min, ax_max_cat)
    ax.scatter(grid_x, grid_y, marker='o', c="#555")
    ax.plot(x, y, linestyle='-', marker='o', c="#b00", markersize=24)

    if out_path:
        dpi = int(OUT_RES_Y) / 9
        fig.savefig(out_path, dpi=dpi) 
        plt.clf() # Clear and close to reduce memory usage
        plt.close()


def print_list(iterable, label="Iterable entries", lim=20, blank_lines=0, wrapper=None, wrapper_kwargs={}):
    """
    Print list entries (up to limit 'lim' or None to print all) on seperate rows. 
    Optional arguments allow adding a label above the list emtries, 
    extra spacing, wrapping individual arguments with a function, 
    and providing keyword arguments (in a dict) to the wrapper function.
    """
    if isinstance(iterable, set):
        iterable = list(iterable)

    if lim is None:
        lim = len(iterable) 
    newline = '\n'
    
    print(f"#############################\n# {label}:")

    if wrapper:
        [print(f"    {wrapper(i, **wrapper_kwargs)}{newline * blank_lines}") for i in iterable[:min(len(iterable),lim)]]
    else:
        [print(f"    {i}{newline * blank_lines}") for i in iterable[:min(len(iterable),lim)]]

    if len(iterable) > lim:
        print(f"    ... ({len(iterable) - lim} more rows)")
    
    print('\n')


def recursive(kennels=10, path=[]) -> list:
    """
    Generate all possible distributions of cats and dogs in N kennels,
    Given that no cats can be directly adjacent each other. 
    """
    
    if kennels == 0:
        return []

    # Test if we can place a cat
    cat_possible = len(path) == 0 or path[-1] != CAT

    # If the path is completed, return path down recursion stack
    if len(path) == kennels - 1: 
        dog_path = [*path, DOG]
        if cat_possible:
            cat_path = [*path, CAT]
            return [dog_path, cat_path]
        else:
            return [dog_path]
    
    # If the path is not completed (not returned yet), 
    # explore the possible continuations.
    dog_paths = recursive(kennels, [*path, DOG])  
    if cat_possible:
        cat_paths = recursive(kennels, [*path, CAT])
        return [*dog_paths, *cat_paths]
    else:
        return [*dog_paths]



if __name__ == '__main__':
    n = 10
    arrangements = recursive(n)

    # Plot all arrangements as a path
    plot_all_and_save(arrangements)

    # # Print each arrangement in string representations line-by-line
    # print_list(arrangements, label=f"Valid paths for {n=}", lim=None, wrapper=cat_dog_wrapper)
    # print(f'At {n=}, there are {len(arrangements)} possible arrangements')