import numpy as np
from scipy.stats import t as StudentDistr 
from scipy.stats import ttest_ind

class Partition:
    def __init__(self, S1, S2, genes):
        
        self.S1 = S1
        self.S2 = S2
        self.genes = genes
    

def calc_evklid(x1, x2):
    
    return np.linalg.norm(x1 - x2)

def calc_dist(points):

    matrix = -np.ones((points.shape[0], points.shape[0]))
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            dst = calc_evklid(points[i], points[j])
            matrix[i][j] = dst
            matrix[j][i] = dst
            
    return matrix

def calc_t_stat(points, S1, S2, gene):
    a = points[S1, gene]
    b = points[S2, gene]

    
    return ttest_ind(a, b)

def get_cluster(matrix, v):
    used = [0] * len(matrix)
    S = []
    
    def dfs(v):
        used[v] = 1
        S.append(v)
        for i in range(len(matrix)):
            if matrix[v][i] != -1 and not used[i]:
                dfs(i)
                
    dfs(v)
                
    return S, len(S)


def calc_FS(edge, edges_matrix, points):
    
    left, right = edge

    S1, n_obj1 = get_cluster(edges_matrix, left)
    S2, n_obj2 = get_cluster(edges_matrix, right)
    
    sum1 = np.zeros(points.shape[1])
    sum2 = np.zeros(points.shape[1])
    
    for v in S1:
        sum1 += points[v]
        
    for v in S2:
        sum2 += points[v]

    
    mu1 = sum1 / n_obj1
    mu2 = sum2 / n_obj2

    mu_global = (sum1 + sum2) / (n_obj1 + n_obj2)
    
    fs1 = np.sum((np.linalg.norm(points[S1] - mu1, axis=1) ** 2) - (np.linalg.norm(mu1 - mu_global) ** 2))
    fs2 = np.sum((np.linalg.norm(points[S2] - mu2, axis=1) ** 2) - (np.linalg.norm(mu2 - mu_global) ** 2))
    
    
    fs = fs1 + fs2
    
    return S1, S2, fs


def get_best_partition(edges_list, edges_matrix, points):
    min_fs, best_edge = np.inf, (0, 0)
    bestS1, bestS2 = 0, 0
    for edge in edges_list:
        left, right = edge
        dst = edges_matrix[left][right]

        edges_matrix[left][right] = -1
        edges_matrix[right][left] = -1
        S1, S2, fs = calc_FS(edge, edges_matrix, points)
        
        if fs < min_fs:
            min_fs = fs
            best_edge = edge
            bestS1 = S1
            bestS2 = S2
        
        edges_matrix[left][right] = dst
        edges_matrix[right][left] = dst
        
    return bestS1, bestS2, best_edge

def create_mst(matrix):
    N = len(matrix)
    postitive_inf = float('inf')
    
    edges_list = []
    
    selected_nodes = [False for node in range(N)]
    result = [[-1 for column in range(N)] 
                for row in range(N)]
    
    indx = 0
    while (False in selected_nodes):
        minimum = postitive_inf
        start = 0
        end = 0
        for i in range(N):
            if selected_nodes[i]:
                for j in range(N):
                    if (not selected_nodes[j] and matrix[i][j] != -1 and i != j):  
                        if matrix[i][j] < minimum:
                            minimum = matrix[i][j]
                            start, end = i, j
        
        selected_nodes[end] = True
        result[start][end] = minimum
        
        if minimum == postitive_inf:
            result[start][end] = -1
        indx += 1
        
        result[end][start] = result[start][end]

    for i in range(len(result)):
        for j in range(i, len(result)):
            if result[i][j] != -1:
                edges_list.append((i, j))
    
    return edges_list, result


