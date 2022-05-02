from typing import List, Set, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx
import copy


def LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:
    total_infected = set(patients_0)
    # TODO implement your code here
    for node in graph.nodes:
        graph.nodes[node]['cv'] = 0

    for i in range(iterations):
        new_infected = []
        for node in set(graph.nodes).difference(total_infected):
            weight_infected = 0

            for neighbor in total_infected.intersection(graph.neighbors(node)):
                weight_infected += float(graph.edges[node, neighbor]['weight'])
            if (CONTAGION * weight_infected) >= (1 + graph.nodes[node]['cv']):
                new_infected.append(node)

            if (len(set(graph.neighbors(node)))) > 0:
                graph.nodes[node]['cv'] = len(set(graph.neighbors(node)).intersection(total_infected))\
                                          / len(graph.adj[node])
            else:
                graph.nodes[node]['cv'] = 0

        total_infected = total_infected.union(new_infected)

    return total_infected


def ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]:

    total_infected = set(patients_0)
    total_deceased = set()
    relevant_infected = set()
    # TODO implement your code here
    for node in graph.nodes:
        graph.nodes[node]['cv'] = 0

    for infected in total_infected:
        if np.random.random() <= LETHALITY:
            total_deceased.add(infected)

    relevant_infected = total_infected.difference(total_deceased)
    total_infected = total_infected.difference(total_deceased)

    for i in range(iterations):
        new_infected = set()
        new_deceased = set()

        for node in set(graph.nodes).difference(total_infected.union(total_deceased)):
            for neighbor in relevant_infected.intersection(graph.neighbors(node)):
                probabilty = min(1,
                                 CONTAGION *
                                    float(graph.edges[node, neighbor]['weight']) *
                                    (1 - graph.nodes[node]['cv']))
                if np.random.random() <= probabilty:
                    new_infected.add(node)
                    break

        for infected in new_infected:
            if np.random.random() <= LETHALITY:
                new_deceased.add(infected)

        for node in set(graph.nodes).difference(total_deceased.union(new_deceased)
                                                .union(total_infected).union(new_infected)):
            if(len(set(graph.neighbors(node)))) > 0:
                graph.nodes[node]['cv'] = min(1.0,
                                          (len(set(graph.neighbors(node)).intersection(total_infected)) +
                                            3 * len(set(graph.neighbors(node)).intersection(total_deceased))) /
                                            len(set(graph.neighbors(node))))
            else:
                graph.nodes[node]['cv'] = 0

        total_deceased = total_deceased.union(new_deceased)
        total_infected = (total_infected.union(new_infected)).difference(total_deceased)

        relevant_infected = new_infected.difference(total_deceased)

    return total_infected, total_deceased


def plot_degree_histogram(histogram: Dict):
    # TODO implement your code here
    plt.bar(histogram.keys(), histogram.values(), align='center')
    plt.xlabel("degree")
    plt.ylabel("number of nodes")
    plt.title("Distribution of Nodes degree in graph")
    #plt.show() doesnt run on machine so we put it as error

def calc_degree_histogram(graph: networkx.Graph) -> Dict:
    """
    Example:
    if histogram[1] = 10 -> 10 nodes have only 1 friend
    """
    histogram = {}
    # TODO implement your code here
    for node in graph.nodes:

        if histogram.get(graph.degree[node]):
            histogram[graph.degree[node]] += 1
        else:
            histogram[graph.degree[node]] = 1

    return histogram



def build_graph(filename: str) -> networkx.Graph:
    G = networkx.Graph()
    # TODO implement your code here
    with open(filename, 'r') as file:
        lines = file.read().splitlines()[1:]

        for line in lines:
            edge = line.split(',')
            source, dest = edge[0], edge[1]
            if len(edge) > 2:
                G.add_edge(source, dest, weight=edge[2])
            else:
                G.add_edge(source, dest)

    return G


def clustering_coefficient(graph: networkx.Graph) -> float:
    # TODO implement your code here
    closed_triangles = 0
    open_triangles = 0
    for node in graph.nodes:
        neighbors = list(graph.adj[node])
        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                if graph.has_edge(neighbors[i], neighbors[j]):
                    closed_triangles += 1
                else:
                    open_triangles += 1
    cc = ((closed_triangles) / (closed_triangles + open_triangles))
    return cc


def compute_lethality_effect(graph: networkx.Graph, t: int) -> [Dict, Dict]:
    global LETHALITY
    mean_deaths = {}
    mean_infected = {}

    for l in (.05, .15, .3, .5, .7):
        LETHALITY = l
        list_infected = []
        list_deceased = []
        for iteration in range(30):
            G = copy.deepcopy(graph)
            patients_0 = np.random.choice(list(G.nodes), size=50, replace=False, p=None)
            # TODO implement your code here
            infected, deceased = ICM(G,patients_0,t)
            list_infected.append(len(infected))
            list_deceased.append(len(deceased))
        mean_infected[l] = sum(list_infected)/len(list_infected)
        mean_deaths[l] = sum(list_deceased)/len(list_deceased)

    return mean_deaths, mean_infected


def plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict):
    # TODO implement your code here
    deaths_keys = list(mean_deaths.keys())
    deaths_values = list(mean_deaths.values())
    infected_values = list(mean_infected.values())

    df = pd.DataFrame({'mean_deaths': deaths_values, 'mean_infected': infected_values}, index=deaths_keys)
    ax = df.plot.line()
    #plt.show() doesnt run on machine so we put it as error


def choose_who_to_vaccinate(graph: networkx.Graph) -> List:
    people_to_vaccinate = []
    # TODO implement your code here
    node2weight = {}
    for node in graph.nodes:
        sum_weight = 0
        for neighbor in graph.neighbors(node):
            sum_weight += float(graph[node][neighbor]['weight'])
        node2weight[node] = sum_weight
    sorted_nodes = sorted(node2weight.items(), key=lambda item: item[1], reverse=True)[:50]
    people_to_vaccinate = [node[0] for node in sorted_nodes]

    return people_to_vaccinate



def choose_who_to_vaccinate_example(graph: networkx.Graph) -> List:
    """
    The following heuristic for Part C is simply taking the top 50 friendly people;
     that is, it returns the top 50 nodes in the graph with the highest degree.
    """
    node2degree = dict(graph.degree)
    sorted_nodes = sorted(node2degree.items(), key=lambda item: item[1], reverse=True)[:50]
    people_to_vaccinate = [node[0] for node in sorted_nodes]
    return people_to_vaccinate


"Global Hyper-parameters"
CONTAGION = 0.8
LETHALITY = .2

if __name__ == "__main__":
    filename = ""






