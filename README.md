# Epidemic Simulation on Graphs

This repository contains Python code to simulate the spread of an epidemic on a graph using the Linear Threshold Model (LTM) and Independent Cascade Model (ICM). It also includes functions to analyze the graph's properties, such as degree distribution and clustering coefficient, and to evaluate the impact of different lethality rates on the epidemic's spread.

## Key Functions

### Epidemic Models

- **LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set**:
  - Simulates the spread of an epidemic using the Linear Threshold Model.
  - Returns the set of infected nodes after the given iterations.

- **ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]**:
  - Simulates the spread of an epidemic using the Independent Cascade Model.
  - Returns the sets of infected and deceased nodes after the given iterations.

### Graph Analysis

- **plot_degree_histogram(histogram: Dict)**:
  - Plots the degree distribution histogram of the graph.

- **calc_degree_histogram(graph: networkx.Graph) -> Dict**:
  - Calculates the degree distribution histogram of the graph.
  - Returns a dictionary where keys are degrees and values are the number of nodes with that degree.

- **build_graph(filename: str) -> networkx.Graph**:
  - Builds a graph from a given CSV file.
  - Returns the constructed graph.

- **clustering_coefficient(graph: networkx.Graph) -> float**:
  - Calculates the clustering coefficient of the graph.
  - Returns the clustering coefficient.

### Epidemic Impact Analysis

- **compute_lethality_effect(graph: networkx.Graph, t: int) -> [Dict, Dict]**:
  - Computes the effect of different lethality rates on the epidemic's spread.
  - Returns dictionaries with mean deaths and mean infections for each lethality rate.

- **plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict)**:
  - Plots the effect of different lethality rates on mean deaths and infections.

### Vaccination Strategy

- **choose_who_to_vaccinate(graph: networkx.Graph) -> List**:
  - Chooses the top 50 nodes to vaccinate based on the sum of edge weights.

- **choose_who_to_vaccinate_example(graph: networkx.Graph) -> List**:
  - Chooses the top 50 nodes to vaccinate based on node degree.


