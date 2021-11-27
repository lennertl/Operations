import random
import matplotlib.pyplot as plt
import pandas as pd

def generate_node(x_size, y_size, backhaul=False):
    x_coord = random.randint(-x_size, x_size)
    y_coord = random.randint(-y_size, y_size)
    backhaul = backhaul
    node = (x_coord, y_coord, backhaul)
    return node


def generate_grid(size=(200, 200), backhaul_rate=30, number_of_nodes=100):
    nodes = []
    while len(nodes) < backhaul_rate*number_of_nodes/100:
        node = generate_node(size[0]/2, size[1]/2, backhaul=True)
        nodes.append(node)
    while len(nodes) < number_of_nodes:
        node = generate_node(size[0]/2, size[1]/2, backhaul=False)
        nodes.append(node)
    return nodes


def plot_grid(grid):
    for node in grid:
        if node[2]:
            plt.scatter(node[0], node[1], c="red")
        else:
            plt.scatter(node[0], node[1], c="green")
    plt.show()


random.seed(1)
grid = generate_grid()
plot_grid(grid)
grid_frame = pd.DataFrame(grid)
grid_frame.to_csv("grid.csv")
