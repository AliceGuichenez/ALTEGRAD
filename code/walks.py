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
    neighbors = list(graph.neighbors(walk[-1]))
    walk.append(random.choice(neighbors))
    for i in range(walk_length-1):
        neighbors = list(graph.neighbors(walk[-1]))
        probabilities = probs[walk[-2]][walk[-1]]
        walk.append(np.random.choice(neighbors, p=probabilities))
    return walk


def generate_walks(graph, num_walks, walk_length, max_doc_size, padding_filler, biased = False, p = None, q = None):
    '''samples num_walks walks of length walk_length+1 from each node of graph'''
    # Set the generator of walk (biased or not)
    if biased:
        probs = compute_probabilities(graph, p, q)
        generator = lambda graph, node, walk_length :  biased_walk(graph, node, walk_length, probs)
    else:
        generator = random_walk
    
    # Generate walk for each node
    graph_nodes = graph.nodes()
    walks = np.empty((max_doc_size, walk_length + 1), dtype = np.int)
    walks.fill(padding_filler)
    
    i = 0
    for _ in range(num_walks):
        nodes = np.random.permutation(graph_nodes)
        for node in nodes:
            walk = generator(graph, node, walk_length)
            walks[i] = np.array(walk, dtype = np.int)
            i += 1
            if max_doc_size <= i:
                break
        if max_doc_size <= i:
            break
    return walks