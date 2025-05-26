import collections
import heapq

# --- Configurações do Grafo ---
# Define o número de nós (vértices) que o grafo terá. Os nós serão numerados de 0 a CONST_N-1.
CONST_N = 5

# --- Definição das Funções de Geração de Grafo ---

def generate_graph_edges():
    """
    Esta lista é usada principalmente para os algoritmos de Árvore Geradora Mínima (MST).
    Formato da aresta: (nó_origem, nó_destino, peso_da_aresta)
    """
    if CONST_N == 5:
        edges_data = [
            (0, 1, 4), (1, 0, 4),  
            (0, 2, 2), (2, 0, 2),  
            (1, 2, 5), (2, 1, 5),  
            (1, 3, 10), (3, 1, 10),
            (2, 3, 3), (3, 2, 3),
            (2, 4, 7), (4, 2, 7),
            (3, 4, 1), (4, 3, 1)
        ]
    elif CONST_N == 6:
        edges_data = [
            (0, 1, 7), (1, 0, 7),
            (0, 2, 9), (2, 0, 9),
            (0, 5, 14), (5, 0, 14),
            (1, 2, 10), (2, 1, 10),
            (1, 3, 15), (3, 1, 15),
            (2, 3, 11), (3, 2, 11),
            (2, 5, 2), (5, 2, 2),
            (3, 4, 6), (4, 3, 6),
            (4, 5, 9), (5, 4, 9)
        ]
    else:
        print(f"AVISO: CONST_N = {CONST_N} não tem um grafo pré-definido. Usando grafo vazio.")
        edges_data = []

    return edges_data

def generate_directed_graph_edges():
    """
    Esta lista é usada para os algoritmos de caminho em grafos direcionados.
    """
    if CONST_N == 5:
        edges_data = [
            (0, 1, 4),  
            (0, 2, 2),  
            (1, 2, 5),  
            (1, 3, 10),
            (2, 4, 7),
            (3, 2, 3),
            (4, 3, 1)
        ]
    elif CONST_N == 6:
        edges_data = [
            (0, 1, 7),
            (0, 2, 9),
            (1, 2, 10),
            (2, 5, 2),
            (3, 1, 15),
            (3, 4, 6),
            (4, 5, 9),
            (5, 0, 14)
        ]
    else:
        print(f"AVISO: CONST_N = {CONST_N} não tem um grafo direcionado pré-definido. Usando grafo vazio.")
        edges_data = []
    return edges_data

def get_unique_undirected_edges(all_edges):
    """
    Pega uma lista de arestas (que pode ter ida e volta) e a transforma em uma lista "limpa".
    Ela remove duplicatas tratando (u, v) e (v, u) como a mesma coisa.
    A forma "canônica" (padrão) é (nó_menor, nó_maior, peso).
    essencial para os algoritmos de MST como Kruskal e Apaga Reverso.
    """
    # Um 'set' é usado para garantir que cada aresta única seja armazenada apenas uma vez.
    unique_edges_set = set()
    # Itera sobre todas as arestas da lista de entrada.
    for u, v, w in all_edges:
        # Ordena os nós (u, v) para que (1, 0) se torne (0, 1), por exemplo.
        # Isso garante que a aresta entre 0 e 1 seja sempre representada da mesma forma.
        edge_tuple = tuple(sorted((u, v))) + (w,)
        # Adiciona a tupla padronizada ao conjunto. Duplicatas são ignoradas automaticamente.
        unique_edges_set.add(edge_tuple)
    # Converte o conjunto de volta para uma lista e a retorna.
    return list(unique_edges_set)

# --- Funções Auxiliares ---
def get_all_nodes(edges):
    """Retorna um conjunto de todos os nós (vértices) que aparecem na lista de arestas."""
    nodes = set()
    # Itera sobre cada aresta (u, v, w) e adiciona u e v ao conjunto de nós.
    # O '_' ignora o peso, pois não precisamos dele aqui.
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    return nodes

def build_undirected_adj_list(edges, use_weights=True):
    """
    Constrói e retorna uma lista de adjacência para um grafo NÃO DIRECIONADO.
    Uma lista de adjacência é um dicionário onde cada chave é um nó, e o valor é uma lista de seus vizinhos.
    """
    # defaultdict(list) cria um dicionário que, se uma chave não existir, cria-a com uma lista vazia como valor.
    adj = collections.defaultdict(list)

    # Itera sobre as arestas para preencher o dicionário.
    for u, v, w in edges:
        # Prepara a informação do vizinho: (vizinho, peso) ou apenas o vizinho se não usar pesos.
        data_u_to_v = (v, w) if use_weights else v
        data_v_to_u = (u, w) if use_weights else u


        # Adiciona a conexão de u para v.
        if data_u_to_v not in adj[u]:
            adj[u].append(data_u_to_v)
        # Como é não direcionado, adiciona também a conexão de v para u.
        if data_v_to_u not in adj[v]:
            adj[v].append(data_v_to_u)

    # Garante que todos os nós de 0 a CONST_N-1 existam no dicionário, mesmo que não tenham arestas (nós isolados).
    all_defined_nodes = set(range(CONST_N))
    for node in all_defined_nodes:
        adj[node].sort() # Ordena a lista de vizinhos para uma exibição consistente.
       
    # Retorna o dicionário final, ordenado pelas chaves (nós) para exibição.
    return collections.OrderedDict(sorted(adj.items()))

def build_directed_adj_list(edges, use_weights=True):
    """Constrói e retorna uma lista de adjacência para um grafo DIRECIONADO."""
    adj = collections.defaultdict(list)
   
    # Itera sobre as arestas.
    for u, v, w in edges:
        # Prepara a informação do vizinho.
        data = (v, w) if use_weights else v
        # Adiciona a conexão APENAS de u para v. Não adiciona a volta (v para u).
        if data not in adj[u]:
            adj[u].append(data)

    # Garante que todos os nós definidos existam no dicionário.
    all_defined_nodes = set(range(CONST_N))
    for node in all_defined_nodes:
        adj[node].sort()
       
    return collections.OrderedDict(sorted(adj.items()))

def reconstruct_path(predecessors, start_node, end_node):
    """
    Refaz o caminho do nó final até o inicial, usando o dicionário 'predecessors'.
    O dicionário 'predecessors' foi criado por BFS ou Dijkstra e armazena "de onde eu vim" para cada nó.
    """
    # Usa um 'deque' porque adicionar no início é mais rápido.
    path = collections.deque()
    current_node = end_node
   
    # Conjunto para detectar ciclos em caminhos inválidos e evitar loops infinitos.
    path_nodes_set = set()


    # Loop que volta no tempo, do destino para a origem.
    while current_node is not None and current_node not in path_nodes_set:
        path.appendleft(current_node) # Adiciona o nó atual no início do caminho.
        path_nodes_set.add(current_node) # Marca como parte do caminho.
        # Se chegamos ao início, o caminho está completo.
        if current_node == start_node:
            return list(path) # Retorna o caminho como uma lista normal.
        # Pega o "pai" do nó atual para continuar o backtracking.
        current_node = predecessors.get(current_node)
   
    # Se o loop terminar sem encontrar o start_node, não há caminho.
    return None

# --- Implementações dos Algoritmos ---

# BFS (Busca em Largura) para caminho mínimo em grafos NÃO PONDERADOS.
def bfs_shortest_path(adj_list, start_node, end_node):
    """Encontra o caminho mais curto em número de arestas."""
    # Validação inicial.
    if start_node not in adj_list or end_node not in adj_list:
        return None, float('inf')
       
    # 'queue' é a fila de nós a serem visitados. Começa com o nó inicial.
    queue = collections.deque([start_node])
    # 'predecessors' armazena o caminho de volta.
    predecessors = {node: None for node in adj_list}
    # 'distances' armazena o número de passos desde o início. Começa como infinito.
    distances = {node: float('inf') for node in adj_list}

    # A distância do nó inicial para ele mesmo é 0.
    if start_node in distances:
        distances[start_node] = 0
    else:
        return None, float('inf')

    # Loop principal: continua enquanto houver nós na fila para visitar.
    while queue:
        # Pega o primeiro nó da fila (o mais antigo).
        current_node = queue.popleft()

        # Se chegamos ao destino, podemos parar.
        if current_node == end_node:
            break

        # Itera sobre todos os vizinhos do nó atual.
        for neighbor in adj_list.get(current_node, []):
            # Se o vizinho ainda não foi visitado (distância infinita).
            if distances[neighbor] == float('inf'):
                # A distância dele é a distância do nó atual + 1.
                distances[neighbor] = distances[current_node] + 1
                # Anota que viemos do 'current_node' para chegar no 'neighbor'.
                predecessors[neighbor] = current_node
                # Adiciona o vizinho no final da fila para ser explorado depois.
                queue.append(neighbor)
   
    # Após o loop, reconstrói e retorna o caminho e a distância.
    final_path = reconstruct_path(predecessors, start_node, end_node)
    if final_path:
        return final_path, distances[end_node]
    else:
        return None, float('inf')

# Dijkstra para caminho mínimo em grafos PONDERADOS.
def dijkstra_shortest_path(adj_list_weighted, start_node, end_node):
    """Encontra o caminho de menor custo (soma dos pesos)."""
    # Validação inicial.
    if start_node not in adj_list_weighted or end_node not in adj_list_weighted:
        return None, float('inf')

    # 'distances' armazena o menor custo encontrado até agora para cada nó.
    distances = {node: float('inf') for node in adj_list_weighted}
    # 'predecessors' armazena o caminho de volta.
    predecessors = {node: None for node in adj_list_weighted}
   
    if start_node not in distances: return None, float('inf')

    # O custo para o nó inicial é 0.
    distances[start_node] = 0
    # 'pq' (priority queue) é a fila de prioridades.
    # Ela armazena (custo_total, nó) e sempre nos dará o par com o menor custo.
    pq = [(0, start_node)]

    # Loop principal: continua enquanto houver "próximos passos" a serem considerados.
    while pq:
        # Pega o passo mais barato da fila de prioridades.
        current_distance, current_node = heapq.heappop(pq)

        # Se já encontramos um caminho mais barato para este nó, ignoramos este passo.
        if current_distance > distances[current_node]:
            continue
       
        # Otimização: se chegamos ao destino, podemos parar.
        if current_node == end_node:
            break

        # Itera sobre os vizinhos do nó atual.
        for neighbor, weight in adj_list_weighted.get(current_node, []):
            # Calcula o custo para chegar a este vizinho através do nó atual.
            distance = current_distance + weight
            # "Relaxamento": Se este novo caminho for mais barato que o conhecido...
            if distance < distances[neighbor]:
                # ...atualizamos a planilha de custos...
                distances[neighbor] = distance
                # ...anotamos o caminho de volta...
                predecessors[neighbor] = current_node
                # ...e adicionamos este novo e melhor caminho à nossa fila de prioridades.
                heapq.heappush(pq, (distance, neighbor))
   
    # Após o loop, reconstrói e retorna o caminho e o custo.
    final_path = reconstruct_path(predecessors, start_node, end_node)
    final_dist = distances.get(end_node, float('inf'))
   
    if final_path:
        return final_path, final_dist
    else:
        return None, final_dist

# Estrutura DSU (Disjoint Set Union) ou Union-Find.
# Ajuda a responder "estes dois nós já estão no mesmo grupo/componente?".
class DSU:
    # Inicializa a estrutura. Cada nó começa como seu próprio "pai", em seu próprio grupo.
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}


    # 'find' encontra o "representante" (ou raiz) do grupo ao qual o nó 'v' pertence.
    def find(self, v):
        # Se o pai do nó é ele mesmo, ele é o representante.
        if self.parent[v] == v:
            return v
        # Otimização (Path Compression): Faz todos os nós no caminho apontarem diretamente para a raiz.
        self.parent[v] = self.find(self.parent[v])
        return self.parent[v]


    # 'union' une os grupos de dois nós, se eles já não estiverem no mesmo grupo.
    def union(self, v1, v2):
        root1 = self.find(v1)
        root2 = self.find(v2)
        # Se os representantes são diferentes, eles estão em grupos diferentes.
        if root1 != root2:
            # Une os grupos fazendo um ser o pai do outro.
            self.parent[root2] = root1
            return True # Retorna True para indicar que uma união foi feita.
        return False # Retorna False se eles já estavam no mesmo grupo.

# Função auxiliar que usa DSU para checar se um grafo é conectado.
def is_graph_connected(edges, nodes):
    """Verifica se todos os nós em 'nodes' estão conectados pelas 'edges'."""
    if not nodes: return True
    if not edges and len(nodes) > 1: return False
   
    node_list = list(nodes)
    dsu = DSU(node_list)
    # Une os nós para cada aresta.
    for u, v, _ in edges:
        dsu.union(u, v)
       
    if not node_list: return True
    # Pega o representante do primeiro nó.
    first_root = dsu.find(node_list[0])
    # Se todos os outros nós tiverem o mesmo representante, o grafo é conectado.
    return all(dsu.find(node) == first_root for node in node_list)

# Algoritmo de Kruskal para encontrar a MST.
def kruskal_mst(unique_undirected_edges, all_graph_nodes):
    """Abordagem "gananciosa": adiciona as arestas mais baratas primeiro, sem formar ciclos."""
    if not all_graph_nodes: return [], 0
   
    # Passo 1: Ordena todas as arestas únicas pelo peso, em ordem crescente.
    sorted_edges = sorted(unique_undirected_edges, key=lambda x: x[2])
   
    # Prepara o DSU para controlar os componentes conectados.
    dsu = DSU(all_graph_nodes)
    mst_edges = []
    mst_weight = 0
   
    # Itera sobre as arestas ordenadas, da mais barata para a mais cara.
    for u, v, weight in sorted_edges:
        # Se a aresta conecta dois componentes diferentes (não forma um ciclo)...
        if dsu.union(u, v):
            # ...adiciona a aresta à nossa árvore...
            mst_edges.append((u, v, weight))
            # ...e soma seu peso ao custo total.
            mst_weight += weight
   
    return mst_edges, mst_weight

# Algoritmo de Prim para encontrar a MST.
def prim_mst(adj_list_weighted, all_graph_nodes):
    """Abordagem "expansionista": começa de um nó e expande a árvore com a aresta mais barata na fronteira."""
    if not all_graph_nodes: return [], 0
   
    # Escolhe um nó qualquer para começar.
    start_node = list(all_graph_nodes)[0]
   
    # 'visited' controla os nós que já estão na nossa árvore.
    visited = {start_node}
    mst_edges = []
    mst_weight = 0
    # 'pq' é a fila de prioridades para guardar as arestas na "fronteira" da nossa árvore.
    pq = []

    # Adiciona as arestas do nó inicial à fila de prioridades.
    for neighbor, weight in adj_list_weighted.get(start_node, []):
        heapq.heappush(pq, (weight, start_node, neighbor))

    # Loop principal: continua enquanto a fronteira não estiver vazia e nem todos os nós foram visitados.
    while pq and len(visited) < len(all_graph_nodes):
        # Pega a aresta mais barata da fronteira.
        weight, u, v = heapq.heappop(pq)
       
        # Se o destino 'v' ainda não foi visitado...
        if v not in visited:
            # ...adiciona 'v' à nossa árvore.
            visited.add(v)
            # Adiciona a aresta e o peso aos nossos resultados.
            mst_edges.append(tuple(sorted((u, v))) + (weight,))
            mst_weight += weight
           
            # Agora, adiciona as novas arestas de 'v' à nossa fronteira.
            for next_neighbor, next_weight in adj_list_weighted.get(v, []):
                if next_neighbor not in visited:
                    heapq.heappush(pq, (next_weight, v, next_neighbor))
   
    return mst_edges, mst_weight

# Algoritmo Apaga Reverso (Reverse-Delete) para encontrar a MST.
def reverse_delete_mst(unique_undirected_edges, all_graph_nodes):
    """Abordagem "pessimista": começa com todas as arestas e remove as mais caras que não quebram a conectividade."""
    if not all_graph_nodes or not unique_undirected_edges: return [], 0

    # Passo 1: Começa com todas as arestas, ordenadas da mais cara para a mais barata.
    mst_final_edges = sorted(unique_undirected_edges, key=lambda x: x[2], reverse=True)
   
    i = 0
    # Itera sobre as arestas para decidir se remove ou não.
    while i < len(mst_final_edges):
        # Cria uma lista temporária sem a aresta atual.
        temp_edges = mst_final_edges[:i] + mst_final_edges[i+1:]
       
        # Se o grafo continuar conectado mesmo sem esta aresta...
        if is_graph_connected(temp_edges, all_graph_nodes):
            # ...a remoção é segura e se torna permanente.
            mst_final_edges = temp_edges
        # Se não, a aresta é essencial. A mantemos e passamos para a próxima.
        else:
            i += 1
           
    # Soma o peso das arestas restantes.
    final_weight = sum(w for _, _, w in mst_final_edges)
    return mst_final_edges, final_weight

def opcao1():
    print("\n--- Opção 1: Lista de Adjacência (Não Direcionado) ---")
    raw_edges = generate_graph_edges()
    if not raw_edges and CONST_N > 0: print("Grafo pré-definido está vazio."); return
    g = build_undirected_adj_list(raw_edges, use_weights=True)
    print("Representação: Nó : [(Vizinho, Peso)]")
    for key in sorted(g.keys()):
        print(f"{key} : {g[key]}")
    print("-" * 40)

def opcao2():
    print("\n--- Opção 2: Lista de Adjacência (Direcionado) ---")
    raw_edges = generate_directed_graph_edges()
    if not raw_edges and CONST_N > 0: print("Grafo pré-definido está vazio."); return
    g = build_directed_adj_list(raw_edges, use_weights=True)
    print("Representação: Nó -> [Vizinho, Peso]")
    for key in sorted(g.keys()):
        print(f"{key} : {g[key]}")
    print("-" * 40)

def get_start_end_nodes(nodes_set):
    """Função auxiliar para pedir ao usuário os nós de origem e destino."""
    if not nodes_set:
        print("Grafo não tem nós.")
        return None, None
    nodes_str = ", ".join(map(str, sorted(list(nodes_set))))

    while True:
        try:
            source = int(input(f"Digite o nó de origem ({nodes_str}): "))
            if source not in nodes_set:
                print("Nó de origem inválido.")
                continue
            break
        except ValueError:
            print("Entrada inválida. Digite um número.")
   
    while True:
        try:
            target = int(input(f"Digite o nó de destino ({nodes_str}): "))
            if target not in nodes_set:
                print("Nó de destino inválido.")
                continue
            break
        except ValueError:
            print("Entrada inválida. Digite um número.")
           
    return source, target

def opcao3():
    print("\n--- Opção 3: Menor Caminho (Direcionado, Não Ponderado - BFS) ---")
    raw_edges = generate_directed_graph_edges()
    if not raw_edges and CONST_N > 0: print("Grafo pré-definido está vazio."); return
    adj = build_directed_adj_list(raw_edges, use_weights=False) # BFS não usa pesos
    nodes_set = get_all_nodes(raw_edges)
    if not nodes_set: print("Grafo não tem nós."); return
    source, target = get_start_end_nodes(nodes_set)
    if source is None: return
    path, length = bfs_shortest_path(adj, source, target)
    if path:
        print(f"Caminho de {source} para {target}: {' -> '.join(map(str,path))}")
        print(f"Comprimento: {length} arestas")
    else:
        print(f"Não há caminho de {source} para {target}.")
    print("-" * 40)

def opcao4():
    print("\n--- Opção 4: Menor Caminho (Direcionado, Ponderado - Dijkstra) ---")
    raw_edges = generate_directed_graph_edges()
    if not raw_edges and CONST_N > 0: print("Grafo pré-definido está vazio."); return
    adj = build_directed_adj_list(raw_edges, use_weights=True)
    nodes_set = get_all_nodes(raw_edges)
    if not nodes_set: print("Grafo não tem nós."); return
    source, target = get_start_end_nodes(nodes_set)
    if source is None: return
    path, weight = dijkstra_shortest_path(adj, source, target)
    if path:
        print(f"Caminho de {source} para {target}: {' -> '.join(map(str,path))}")
        print(f"Custo total: {weight}")
    else:
        print(f"Não há caminho de {source} para {target}.")
    print("-" * 40)

def opcao5():
    print("\n--- Opção 5: Menor Caminho (Não Direcionado, Não Ponderado - BFS) ---")
    raw_edges = generate_graph_edges()
    if not raw_edges and CONST_N > 0: print("Grafo pré-definido está vazio."); return
    adj = build_undirected_adj_list(raw_edges, use_weights=False)
    nodes_set = get_all_nodes(raw_edges)
    if not nodes_set: print("Grafo não tem nós."); return
    source, target = get_start_end_nodes(nodes_set)
    if source is None: return
    path, length = bfs_shortest_path(adj, source, target)
    if path:
        print(f"Caminho de {source} para {target}: {' -> '.join(map(str,path))}")
        print(f"Comprimento: {length} arestas")
    else:
        print(f"Não há caminho de {source} para {target}.")
    print("-" * 40)
   
def opcao6():
    print("\n--- Opção 6: Menor Caminho (Não Direcionado, Ponderado - Dijkstra) ---")
    raw_edges = generate_graph_edges()
    if not raw_edges and CONST_N > 0: print("Grafo pré-definido está vazio."); return
    adj = build_undirected_adj_list(raw_edges, use_weights=True)
    nodes_set = get_all_nodes(raw_edges)
    if not nodes_set: print("Grafo não tem nós."); return
    source, target = get_start_end_nodes(nodes_set)
    if source is None: return
    path, weight = dijkstra_shortest_path(adj, source, target)
    if path:
        print(f"Caminho de {source} para {target}: {' -> '.join(map(str,path))}")
        print(f"Custo total: {weight}")
    else:
        print(f"Não há caminho de {source} para {target}.")
    print("-" * 40)

def opcao7():
    print("\n--- Opção 7: Árvore Geradora Mínima - Kruskal ---")
    raw_edges = generate_graph_edges()
    if not raw_edges and CONST_N > 0: print("Grafo pré-definido está vazio."); return
    unique_edges = get_unique_undirected_edges(raw_edges)
    nodes_set = get_all_nodes(raw_edges)
    if not unique_edges and len(nodes_set) > 1 : print("Grafo não tem arestas."); return
    mst, weight = kruskal_mst(unique_edges, nodes_set)
    if mst:
        print("Arestas da MST/Floresta (Kruskal):")
        for u, v, w_edge in mst:
            print(f"  ({u} - {v}, peso: {w_edge})")
        print(f"Peso total: {weight}")
    elif len(nodes_set) <=1:
        print("MST de um grafo com 0 ou 1 nó tem peso 0 e nenhuma aresta.")
    print("-" * 40)

def opcao8():
    print("\n--- Opção 8: Árvore Geradora Mínima - Prim ---")
    raw_edges = generate_graph_edges()
    if not raw_edges and CONST_N > 0: print("Grafo pré-definido está vazio."); return
    adj = build_undirected_adj_list(raw_edges, use_weights=True)
    nodes_set = get_all_nodes(raw_edges)
    if not nodes_set: print("Grafo não tem nós."); return
    mst, weight = prim_mst(adj, nodes_set)
    if mst:
        print("Arestas da MST/Floresta (Prim):")
        for u, v, w_edge in mst:
            print(f"  ({u} - {v}, peso: {w_edge})")
        print(f"Peso total: {weight}")
    elif len(nodes_set) <=1:
        print("MST de um grafo com 0 ou 1 nó tem peso 0 e nenhuma aresta.")
    print("-" * 40)

def opcao9():
    print("\n--- Opção 9: Árvore Geradora Mínima - Apaga Reverso ---")
    raw_edges = generate_graph_edges()
    if not raw_edges and CONST_N > 0: print("Grafo pré-definido está vazio."); return
    unique_edges = get_unique_undirected_edges(raw_edges)
    nodes_set = get_all_nodes(raw_edges)
    if not unique_edges and len(nodes_set) > 1 : print("Grafo não tem arestas."); return
    mst, weight = reverse_delete_mst(unique_edges, nodes_set)
    if mst:
        print("Arestas da MST/Floresta (Apaga Reverso):")
        for u, v, w_edge in mst:
            print(f"  ({u} - {v}, peso: {w_edge})")
        print(f"Peso total: {weight}")
    elif len(nodes_set) <=1:
        print("MST de um grafo com 0 ou 1 nó tem peso 0 e nenhuma aresta.")
    print("-" * 40)

def opcao10():
    print("\n--- Opção 10: Ordenação de Arestas por Peso ---")
    raw_edges = generate_graph_edges()
    if not raw_edges and CONST_N > 0: print("Grafo pré-definido está vazio."); return
    unique_edges = get_unique_undirected_edges(raw_edges)
    if not unique_edges: print("Grafo não possui arestas para ordenar."); print("-" * 40); return

    # Ordena por peso (crescente), e usa os nós como critério de desempate.
    sorted_asc = sorted(unique_edges, key=lambda x: (x[2], x[0], x[1]))
    print("\nArestas (não direcionadas únicas) ordenadas por peso (Crescente):")
    for u, v, w in sorted_asc:
        print(f"  ({u} - {v}, peso: {w})")

    # Ordena por peso (decrescente).
    sorted_desc = sorted(unique_edges, key=lambda x: (x[2], x[0], x[1]), reverse=True)
    print("\nArestas (não direcionadas únicas) ordenadas por peso (Decrescente):")
    for u, v, w in sorted_desc:
        print(f"  ({u} - {v}, peso: {w})")
    print("-" * 40)

# --- Estrutura do Menu Principal ---
opcoes = {
    1: opcao1, 2: opcao2, 3: opcao3, 4: opcao4, 5: opcao5,
    6: opcao6, 7: opcao7, 8: opcao8, 9: opcao9, 10: opcao10
}

menu_texto = [
    "1. Lista de adjacência de grafos não direcionados",
    "2. Lista de adjacência de grafos direcionados",
    "3. Menor caminho de grafos direcionados (Não Ponderado)",
    "4. Menor caminho de grafos direcionados ponderados (Dijkstra)",
    "5. Menor caminho de grafos não direcionados (Não Ponderado)",
    "6. Menor caminho de grafos não direcionados ponderados (Dijkstra)",
    "7. Árvore geradora mínima - Kruskal",
    "8. Árvore geradora mínima - Prim",
    "9. Árvore geradora mínima - Apaga Reverso",
    "10. Ordenação de arestas por peso (Crescente e decrescente)"
]

if __name__ == '__main__':
    print("Bem-vindo a interface de aprendizado sobre Estrutura de dados!")
    print("---------------------------------------------------------------")
    opc = 0
    while opc != -1:
        print("\nDigite o número do algoritmo que deseja visualizar ou -1 para sair:")
        for item in menu_texto:
            print(item)
       
        try:
            opc_input = input("Digite aqui: ")
            if not opc_input.strip():
                print("Nenhuma opção digitada. Tente novamente.")
                continue
            opc = int(opc_input)

            if opc in opcoes:
                opcoes[opc]()
            elif opc == -1:
                print("Saindo do programa...")
            else:
                print("Opção inválida. Digite um número de 1 a 10 ou -1 para sair.")
        except ValueError:
            print("Erro: Digite apenas números inteiros!")
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")