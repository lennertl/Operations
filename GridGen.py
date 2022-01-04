import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_node(x_size, y_size, backhaul=False):
    """
    generates a node
    :param x_size: one-sided size of the grid in x-direction
    :param y_size: one-sided size of the grid in y-direction
    :param backhaul: truth value if backhaul node or not
    :return: node: tuple containing x-coordinate, y-coordinate, and boolean value (true=backhaul)
    """
    x_coord = random.randint(-x_size, x_size)
    y_coord = random.randint(-y_size, y_size)
    backhaul = backhaul
    node = [x_coord, y_coord, backhaul]
    return node


def generate_grid(size=(200, 200), backhaul_rate=30, number_of_nodes=100):
    """
    generates a grid of nodes
    :param size: tuple containing size
    :param backhaul_rate: percentage of nodes that is backhaul node
    :param number_of_nodes: total number of nodes
    :return: grid: list containing nodes
    """
    nodes = []
    while len(nodes) < (100-backhaul_rate)*number_of_nodes/100:
        node = generate_node(size[0]/2, size[1]/2, backhaul=False)
        nodes.append(node)
    while len(nodes) < number_of_nodes:
        node = generate_node(size[0]/2, size[1]/2, backhaul=True)
        nodes.append(node)
    return nodes


def plot_grid(grid):
    """
    plots grid
    :param grid
    :return: None
    """
    for node in grid:
        if node[2]:
            plt.scatter(node[0], node[1], c="red")
        else:
            plt.scatter(node[0], node[1], c="green")
    plt.show()


def calc_distance(node_1, node_2):
    """
    Calculates distance between nodes
    :param node_1: first node
    :param node_2: second node
    :return: distance between nodes
    """
    return ((node_1[0]-node_2[0])**2+(node_1[1]-node_2[1])**2)**(1/2)


def calc_all_distances(grid):
    """
    generates a matrix with distances
    :param grid: list containing nodes
    :return: numpy array with distances
    """
    distance_matrix = np.zeros((100, 100))
    for count, node in enumerate(grid):
        for count_2 in range(count, len(grid)):
            node_2 = grid[count_2]
            distance_matrix[count, count_2] = distance_matrix[count_2, count] = calc_distance(node, node_2)

    return distance_matrix


random.seed(1)
grid = generate_grid(backhaul_rate=30, number_of_nodes=10)
minimum_distance = 1000
closest_node = 0
for i in range(len(grid)):
    node = grid[i]
    distance_to_zero = node[0]**2 + node[1]**2
    if distance_to_zero < minimum_distance:
        minimum_distance = distance_to_zero
        closest_node = i
temp = grid[0]
if grid[closest_node][2]:
    temp[2] = True
else:
    temp[2] = False
grid[0] = grid[closest_node]
grid[closest_node] = temp
plot_grid(grid)
grid_frame = pd.DataFrame(grid)
grid_frame.to_csv("grid2.csv")
distances = calc_all_distances(grid)
distance_frame = pd.DataFrame(distances)
distance_frame.to_csv("distances.csv")
