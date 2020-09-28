import networkx as nx
import itertools
import numpy as np
from networkx.algorithms import bipartite

B = nx.Graph()
queries = ["pc", "camera", "digital camera", "tv", "flower"]
docs = ["hp.com", "bestbuy.com", "teleflora.com", "orchids.com"]
B.add_nodes_from(queries, bipartite=0)
B.add_nodes_from(docs, bipartite=1)
B.add_weighted_edges_from(
    [
        ("pc", "hp.com", 1),
        ("camera", "hp.com", 1),
        ("camera", "bestbuy.com", 1),
        ("digital camera", "hp.com", 1),
        ("digital camera", "bestbuy.com", 1),
        ("tv", "bestbuy.com", 1),
        ("flower", "teleflora.com", 1),
        ("flower", "orchids.com", 1)
    ]
)


def custom_SimRank(G, importance_factor=0.9, max_iterations=100):
    """
    SimRank:
    - 기본적으로 "나를 가리키는(in-degree) object들끼리 비슷할수록, 비슷하다"는 것을 가정함.
    - 이 개념은 recursion을 내포하게 되는데, 가령 "나를 가리키는 object(A)들은 이 object(A)들을 가리키는
    또다른 object(B)들간이 비슷할수록... 으로 무한히 연속될 수 있음.
    - 따라서, matrix를 반복해가면서 어디로 수렴하는지를 파악해야 함.
    - 초기 값은 diagonal matrix가 됨.
    - 또한, "이웃의 이웃의 이웃"이 반복될 수록 importance_factor로 그 영향을 줄여나감.
    - 아래의 코드는 `nx.simrank_similarity(G)`와 코드 실행 결과가 동일함.
    - 또한, `simrank_similarity_numpy(G)`는 matrix간에 연산을 수행하는데,
    행렬간 연산이 더 익숙한 경우 이 쪽이 더 편할 수 있음.
    """
    prevSimRank = None
    # initial SimRank인데, 당연하지만, 그냥 diagnonal matrix
    # 사실, 엄밀히 따지면, 이 두 노드만 1이어야 하므로.
    NewSimRank = {u: {v: 1 if u == v else 0 for v in G} for u in G}
    for _ in range(0, max_iterations):
        # update하기 전에 원래 값을 `prevSimRank`에 저장하고
        prevSimRank = NewSimRank.copy()
        # UPDATE Node SimRank
        for u in NewSimRank:
            for v in NewSimRank[u]:
                if u == v:  # u, v가 같을 경우에는 1.0
                    NewSimRank[u][v] = 1.0
                else:
                    # u, v 가 다를 경우에는 각각의 neighbor들간의 모든 조합으로부터
                    # 기존 w_x_similarity의 평균을 구하여, 업데이트해줌.
                    u_neighbors, v_neighbors = G[u], G[v]
                    neighbors_product = list(itertools.product(u_neighbors, v_neighbors))
                    u_v_SimRank = 0.0
                    if len(neighbors_product) == 0:
                        NewSimRank[u][v] = u_v_SimRank
                    else:
                        for w, x in neighbors_product:
                            #
                            w_x_SimRank = NewSimRank[w][x]
                            u_v_SimRank += w_x_SimRank
                        # u_v_SimRank average
                        u_v_SimRank /= len(neighbors_product)
                        # u_v_SimRank decay
                        u_v_SimRank *= importance_factor
                        NewSimRank[u][v] = u_v_SimRank
    return NewSimRank


########################################################################
# N = 3
# G = nx.scale_free_graph(N, seed=0)
# G = nx.Graph(G)
# assert nx.is_connected(G)==True

# c_SimRank = custom_SimRank(G)
# nx_SimRank = nx.simrank_similarity(G)
# for u in c_SimRank:
#     for v in c_SimRank[u]:
#         assert round(c_SimRank[u][v], 8)==round(nx_SimRank[u][v], 8)
# print(u, v, custom_node_SimRank, nx_node_SimRank)
print("==" * 30)
print("Assertion complte")
print("==" * 30)

print("==" * 30)
for i in range(0, 12):
    print(f"Simrank at iteration time {i:3d}")
    simrank_dict = custom_SimRank(B, importance_factor=0.8, max_iterations=i)
    np_arr = np.array(
        [[simrank_dict[u][v] for v in simrank_dict[u]] for u in simrank_dict]
    )
    if i <= 10:
        continue
    print(np_arr)
    print("--" * 30)
print("==" * 30)
