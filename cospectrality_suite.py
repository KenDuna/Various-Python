import numpy as np
import re
from itertools import combinations

### R(x) and N(n) are used for Graph6 encoding.

def R(x):
    padded_rep = x
    if len(x)%6 != 0:
        pad_length = 6-len(x)%6        # Pad with zeros on right to make length a multiple of 6
        padded_rep = x + '0'*pad_length
    
    groups = []
    for i in range(int(len(padded_rep)/6)):
        temp_binary = padded_rep[6*i:6*i+6]
        groups.append(int(temp_binary,2)+63)
    return groups

def N(n):
    assert n < 68719476736
    if n in range(63):
        return [n+63]
    elif n in range(63, 258048):
        x = np.binary_repr(n) 
        if len(x) < 18:
            x = '0'*(18-len(x)) + x
            groups = [x[0:6],x[6:12],x[12:18]]
            groups = list(map(lambda x: int(x,2)+63, groups))
            return [126] + groups
    elif n in range(258048,68719476736):
        x = np.binary_repr(n)
        if len(x) < 36:
            x = '0'*(36-len(x)) + x
            groups = [x[0:6],x[6:12], x[12:18], x[18:24], x[24:30], x[30:36]]
            groups = list(map(lambda x: int(x,2)+63, groups))
            return [126, 126] + groups

        

        
#### Creating the Graph class 


class Graph(object):

    def __init__(self, graph_dict=None, graph6_string=None):
        ''' initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used
        '''
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict
        
        if graph6_string == None:
            graph6_string = self.graph6_string()
        else:
            self.__graph_dict = graph6_to_graph(graph6_string)

    def vertices(self):
        ''' returns the vertices of a graph '''
        return list(self.__graph_dict.keys())

    def edges(self):
        ''' returns the edges of a graph '''
        return self.__generate_edges()
    
    def order(self):
        ''' returns the number of vertices'''
        return len(self.vertices())
    
    def degree(self, vertex):
        return len(self.__graph_dict[vertex])
    
    def adjacency_matrix(self):
        verts = self.vertices()
        n = len(verts)
        A = np.zeros(shape=(n,n))
        for i in range(n):
            for j in range(i+1,n):
                if {verts[i],verts[j]} in self.edges():
                    A[i][j] = 1
                    A[j][i] = 1
        return A
    
    def laplacian(self):
        A = self.adjacency_matrix()
        verts = self.vertices()
        for i in range(len(verts)):
            A[i][i] = self.degree(verts[i])
        return A
    
    def distance(self, start, end):
        verts = self.vertices()
        #assert (start in verts) and (end in verts)
        graph_dict = self.__graph_dict
        dist_dict = {vert: 'inf' for vert in verts}

        dist_dict[start] = 0
        current = start
        visited = []
        unvisited = verts
        while end not in visited:
            visited.append(current)
            unvisited.remove(current)

            for vert in graph_dict[current]:
                if vert in unvisited:
                    if dist_dict[vert] == 'inf':
                        dist_dict[vert] = dist_dict[current] + 1
                    else:
                        dist_dict[vert] = min(dist_dict[current] + 1, dist_dict[vert])

            if unvisited == []:
                return dist_dict[end]

            next_vert_possibilities = [vert for vert in unvisited if dist_dict[vert] != 'inf']
            current = next_vert_possibilities[0]
            for vert in next_vert_possibilities:
                if dist_dict[vert] < dist_dict[current]:
                    current = vert

        return dist_dict[end]
    
    def distance_matrix(self):
        verts = self.vertices()
        n = len(verts)
        A = np.zeros(shape=(n,n))
        for i in range(n):
            for j in range(i+1,n):
                d = self.distance(start=verts[i], end=verts[j])
                A[i][j] = d
                A[j][i] = d
        return A 
    
    def distance_laplacian(self):
        verts = self.vertices()
        n = len(verts)
        DL = -self.distance_matrix()
        for i in range(n):
            DL[i][i] = -sum(DL[i])
        return DL
    
    def graph6_string(self):
        verts = self.vertices()
        n = len(verts)
        x = ''
        for j in range(1,n):
            for i in range(j):
                if {verts[i], verts[j]} in self.edges():
                    x = x + '1'
                else:
                    x = x + '0'
        return ''.join(list(map(chr,N(n) + R(x))))
    
    def add_vertex(self, vertex):
        ''' If the vertex "vertex" is not in 
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        '''
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = []

    def add_edge(self, edge):
        ''' assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        '''
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1].append(vertex2)
        else:
            self.__graph_dict[vertex1] = [vertex2]

    def __generate_edges(self):
        ''' A static method generating the edges of the 
            graph "graph". Edges are represented as sets 
            with one (a loop back to the vertex) or two 
            vertices 
        '''
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                if {neighbour, vertex} not in edges:
                    edges.append({vertex, neighbour})
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

#-------------------------------------------------------------------#
#Evaluates the characteristic polynomial of A at t
#
def evaluate_char_poly(A, t):     
    n = np.shape(A)[0]
    return np.linalg.det(A-t*np.identity(n))

# Convert Graph6 string to an incidence dictionary.

def graph6_to_graph(graph6):
    decimal_rep = list(map(ord, graph6))
    n = decimal_rep[0]-63
    
    del decimal_rep[0]
    decimal_rep = [k-63 for k in decimal_rep]

    bin_rep = list(map(lambda q: np.binary_repr(q, width=6), decimal_rep ))

    x_padded = ''.join(bin_rep)    
    
    graph_dict = {k:[] for k in range(n)}
    counter = 0
    for j in range(1,n):
            for i in range(j):
                if int(x_padded[counter]) == 1:
                    graph_dict[i].append(j)
                    graph_dict[j].append(i)
                counter+=1
    return graph_dict

## graphs = list of graph6 strings to be tested.
## func = matrix function. I.e. adjacency matrix, laplacian matrix, etc... Default = Adjacency matrix
## Make sure all inputs have same number of vertices.
## Current options for func: Graph.adjacency_matrix, Graph.laplacian, Graph.distance_matrix, Graph.distance_laplacian

def cospectral_collection(graphs, func=Graph.adjacency_matrix):
    matrix_dict = {}
    
    for graph in graphs:
        graph_obj = Graph(graph6_string=graph)
        matrix_dict[graph] = func(graph_obj)
    
    n = np.shape(matrix_dict[list(matrix_dict.keys())[0]])[0]
    buckets = {}
    for i in range(n+1):
        old_buckets = buckets
        buckets = {}
        if i==0:
            for graph in matrix_dict.keys():
                val = np.round(evaluate_char_poly(matrix_dict[graph], i))
                if val not in buckets.keys():
                    buckets[val] = [graph]
                else:
                    buckets[val].append(graph)
        else:
            for key in old_buckets.keys():
                for graph in old_buckets[key]:
                    val = np.round(evaluate_char_poly(matrix_dict[graph],i))
                    if val not in buckets.keys():
                        buckets[val] = [graph]
                    else:
                        buckets[val].append(graph)
    
    cospectral_collections = []
    for key in buckets.keys():
        if len(buckets[key]) > 1:
            cospectral_collections.append(buckets[key])
    return cospectral_collections