import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import networkx as nx

# Create a small random network
N = 50
G = nx.erdos_renyi_graph(N, p=0.1, seed=42)

# Assign random entanglement weights (mutual information) to edges
for (u, v) in G.edges():
    G[u][v]['weight'] = np.random.uniform(0.5, 1.0)

# Ensure connectivity (add minimal spanning tree if needed)
if not nx.is_connected(G):
    # Add edges to connect components
    components = list(nx.connected_components(G))
    for i in range(len(components)-1):
        u = list(components[i])[0]
        v = list(components[i+1])[0]
        G.add_edge(u, v, weight=0.1)  # weak entanglement

# Compute distance matrix: d_ij = -ln(E_ij) for connected nodes,
# and shortest path for unconnected
distance_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            distance_matrix[i,j] = 0
        elif G.has_edge(i, j):
            # Distance from entanglement
            E = G[i][j]['weight']
            distance_matrix[i,j] = -np.log(E + 1e-10)
        else:
            # Use shortest path through network
            try:
                path_length = nx.shortest_path_length(G, i, j, weight='weight')
                distance_matrix[i,j] = path_length
            except:
                distance_matrix[i,j] = 100  # large default

# Ensure non-negative and symmetric
distance_matrix = np.maximum(distance_matrix, 0)
distance_matrix = (distance_matrix + distance_matrix.T) / 2
np.fill_diagonal(distance_matrix, 0)

# Apply multidimensional scaling to embed in 2D and 3D
mds2 = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
coords2 = mds2.fit_transform(distance_matrix)

mds3 = MDS(n_components=3, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
coords3 = mds3.fit_transform(distance_matrix)

# Plot original network and 2D embedding
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original network
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, ax=axes[0], node_size=50, with_labels=False,
        edge_color='gray', alpha=0.7)
axes[0].set_title('Original Quantum Network')

# 2D embedded "spacetime"
axes[1].scatter(coords2[:,0], coords2[:,1], c='red', s=50)
for i in range(N):
    for j in range(i+1, N):
        if distance_matrix[i,j] < 3:  # only plot close pairs
            axes[1].plot([coords2[i,0], coords2[j,0]],
                         [coords2[i,1], coords2[j,1]],
                         'b-', alpha=0.2)
axes[1].set_title('Emergent 2D Geometry')
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].set_aspect('equal')

# 3D embedding (projected to 2D for visualization)
axes[2].scatter(coords3[:,0], coords3[:,1], c='green', s=50)
axes[2].set_title('3D Embedding (xy-projection)')
axes[2].set_xlabel('x'); axes[2].set_ylabel('y')

plt.tight_layout()
plt.show()

# Compute approximate metric from local neighborhood
def local_metric(center_idx, coords, distances, radius=5, min_neighbors=5):
    """Estimate metric g_ab at a point by fitting ds² = g_ab dx^a dx^b."""
    center = coords[center_idx]
    neighbors = []
    for i in range(len(coords)):
        if i == center_idx: continue
        dx = coords[i] - center
        if np.linalg.norm(dx) < radius:
            neighbors.append((i, dx))

    if len(neighbors) < min_neighbors:
        return None

    # Build design matrix: for each neighbor, we have [dx², dy², 2dxdy] (2D case)
    X = []
    y = []
    for i, dx in neighbors:
        dx2 = dx[0]**2
        dy2 = dx[1]**2
        dxdy = 2 * dx[0] * dx[1]
        X.append([dx2, dy2, dxdy])
        y.append(distances[center_idx, i]**2)

    X = np.array(X)
    y = np.array(y)

    # Solve for g_xx, g_yy, g_xy
    g, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return g

# Compute metric at a few points
test_points = [0, N//4, N//2, 3*N//4]
print("\nLocal metric estimates (g_xx, g_yy, g_xy):")
for idx in test_points:
    g = local_metric(idx, coords2, distance_matrix, radius=5)
    if g is not None:
        print(f"Point {idx}: g = ({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})")
    else:
        print(f"Point {idx}: insufficient neighbors")