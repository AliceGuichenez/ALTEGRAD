import random
import numpy as np

# = = = = = = = = = = = = = = = 

def random_walk(graph,node,walk_length):
    '''Return the usual random walk'''
    walk = [node]
    for i in range(walk_length):
        neighbors = graph.neighbors(walk[i])
        walk.append(random.choice(list(neighbors)))
    return walk


def compute_probabilities(graph, p, q):
    '''Compute a biased matrix of transition for the graph'''
    probs = {}
    for source_node in graph.nodes():
        probs[source_node] = {}
        for current_node in graph.neighbors(source_node):
            probs_ = []
            for destination in graph.neighbors(current_node):
                if source_node==destination:
                    prob_ = graph[current_node][destination].get('weight',1) * (1/p)
                elif destination in graph.neighbors(source_node):
                        prob_ = graph[current_node][destination].get('weight',1)
                else:
                    prob_ = graph[current_node][destination].get('weight',1) * (1/q)
                probs_.append(prob_)
            probs[source_node][current_node] = probs_/np.sum(probs_)
    return probs

                  
def biased_walk(graph, node, walk_length, probs):
    '''Return a biased random walk with probabilities in probs'''
    walk = [node]
    walk_options = list(graph[node])
    walk.append(random.choice(walk_options))
    for i in range(walk_length-1):
        walk_options = list(graph[walk[-1]])
        probabilities = probs[walk[-2]][walk[-1]]
        walk.append(np.random.choice(walk_options, p=probabilities))
    return walk


def generate_walks(graph, num_walks, walk_length, biased = False, p = None, q = None):
    '''samples num_walks walks of length walk_length+1 from each node of graph'''
    # Set the generator of walk (biased or not)
    if biased:
        probs = compute_probabilities(graph, p, q)
        generator = lambda graph, node, walk_length :  biased_walk(graph, node, walk_length, probs)
    else:
        generator = random_walk
    
    # Generate walk for each node
    graph_nodes = graph.nodes()
    n_nodes = len(graph_nodes)
    walks = []
    for i in range(num_walks):
        nodes = np.random.permutation(graph_nodes)
        for j in range(n_nodes):
            walk = generator(graph, nodes[j], walk_length)
            walks.append(walk)
    return walks