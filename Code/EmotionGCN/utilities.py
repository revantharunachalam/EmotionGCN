import graph
import scipy.sparse
import networkx as nx

def generate_graph(m, k, corners=False):
    z = graph.grid(m)
    dist, idx = graph.distance_sklearn_metrics(z, k=k, metric='euclidean')
    A = graph.adjacency(dist, idx)
    if corners:

        A = A.toarray()
        A[A < A.max() / 1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz // 2, k * m ** 2 // 2))
    return A


def create_graph(A, m=28, ax=None, spring_layout=False, size_factor=10):

    assert m ** 2 == A.shape[0] == A.shape[1]
    # Create the nx.Graph object
    G = nx.from_scipy_sparse_matrix(A)
    print('Number of nodes: %d; Number of edges: %d' % \
          (G.number_of_nodes(), G.number_of_edges()))
    grid_coords = graph.grid(m)

    if spring_layout:
        # remove nodes without edges
        nodes_without_edges = [n for n, k in G.degree() if k == 0]
        G.remove_nodes_from(nodes_without_edges)
        print('After removing nodes without edges:')
        print('Number of nodes: %d; Number of edges: %d' % \
              (G.number_of_nodes(), G.number_of_edges()))

    z = graph.grid(m)

    # initial positions
    pos = {n: z[n] for n in G.nodes()}

    if spring_layout:
        pos = nx.spring_layout(G,
                               pos=pos,
                               iterations=200)

    ax = nx.draw(G, pos,
                 node_size=[G.degree(n) * size_factor for n in G.nodes()],
                 ax=ax
                 )
    return ax
