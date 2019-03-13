import random
import numpy as np
import networkx as nx
from time import time

# = = = = = = = = = = = = = = = 

def random_walk(graph,node,walk_length):
    walk = [node]
    for i in range(walk_length):
        neighbors = graph.neighbors(walk[i])
        walk.append(random.choice(list(neighbors)))
    return walk

# modif 3
# change those values

def compute_probabilities(graph, p, q):
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

# modif 3                    
def biased_walk(graph, node, walk_length, probs):
    walk = [node]
    walk_options = list(graph[node])
    walk.append(random.choice(walk_options))
    for i in range(walk_length-2):
        walk_options = list(graph[walk[-1]])
        probabilities = probs[walk[-2]][walk[-1]]
        walk.append(np.random.choice(walk_options, p=probabilities))
    return walk

# modif 3
def generate_walks(graph, num_walks, walk_length, biased=False, p = None, q = None):
    '''
    samples num_walks walks of length walk_length+1 from each node of graph
    '''
    if not biased:
        graph_nodes = graph.nodes()
        n_nodes = len(graph_nodes)
        walks = []
        for i in range(num_walks):
            nodes = np.random.permutation(graph_nodes)
            for j in range(n_nodes):
                walk = random_walk(graph, nodes[j], walk_length)
                walks.append(walk)
        return walks
    else:
        graph_nodes = graph.nodes()
        n_nodes = len(graph_nodes)
        walks = []
        #print("compute prob..")
        probs = compute_probabilities(graph, p, q)
        #print("compute walks..")
        for i in range(num_walks):
            nodes = np.random.permutation(graph_nodes)
            for j in range(n_nodes):
                walk = biased_walk(graph, nodes[j], walk_length, probs)
                walks.append(walk)
        return walks

# = = = = = = = = = = = = = = =

pad_vec_idx = 1685894 # 0-based index of the last row of the embedding matrix (for zero-padding)

# parameters
num_walks = 5
walk_length = 10
max_doc_size = 70 # maximum number of 'sentences' (walks) in each pseudo-document

# = = = = = = = = = = = = = = =
import GraphData as data



start_time = time() 
q = 1.2
p = np.random.uniform(min(q,1)-0.3, max(q,1)+0.3)
docs = []
idxs, edgelists = data.get_graphs(N_train = 105, test = False)
N = len(idxs)
for i, edgelist in enumerate(edgelists):
    g = nx.read_edgelist(edgelist) # construct graph from edgelist
    doc = generate_walks(g,num_walks,walk_length, biased=False, p =p , q=q) # create the pseudo-document representation of the graph
    docs.append(doc)
    if i % 50 == 0:
        print("Computing graph {}/{}...".format(i+1, N).ljust(10), end = "\r")

print('documents generated')

# truncation-padding at the document level, i.e., adding or removing entire 'sentences'
docs = [d+[[pad_vec_idx]*(walk_length+1)]*(max_doc_size-len(d)) if len(d)<max_doc_size else d[:max_doc_size] for d in docs] 

docs = np.array(docs).astype('int')
print('document array shape:',docs.shape) # (93719, 70, 11)

#data.save_docs(idxs, docs, name = "small_biased")a[0]

print('documents saved')
print('everything done in', round(time() - start_time,2)) # 471.33

# = = = = = = = = = = = = = = =






























