import numpy as np
import torch
from torch_geometric.data import Data
from src.utils.functions.parse import tokenizer
from src.utils import log as logger
from gensim.models.keyedvectors import Word2VecKeyedVectors


class NodesEmbedding:
    def __init__(
        self,
        max_nodes: int,
        w2v_keyed_vectors: Word2VecKeyedVectors,
        vector_size,
        is_tokenize_func=False,
    ):
        self.is_tokenize_func = is_tokenize_func
        if not is_tokenize_func:
            self.w2v_keyed_vectors = w2v_keyed_vectors
        else:
            self.tokenize = w2v_keyed_vectors
        self.kv_size = vector_size
        self.max_nodes = max_nodes

        assert self.max_nodes >= 0

        # Buffer for embeddings with padding
        self.target = torch.zeros(self.max_nodes, self.kv_size + 1).float()

    def __call__(self, nodes):
        embedded_nodes = self.embed_nodes(nodes)
        nodes_tensor = torch.from_numpy(embedded_nodes).float()

        self.target[: nodes_tensor.size(0), :] = nodes_tensor

        return self.target

    def embed_nodes(self, nodes):
        embeddings = []

        if self.is_tokenize_func:
            tokens = []
            groups = []
            for i, (n_id, node) in enumerate(nodes.items()):
                # Get node's code
                node_code = node.get_code()
                # Tokenize the code
                tokenized_code = tokenizer(node_code, True)
                if not tokenized_code:
                    # print(f"Dropped node {node}: tokenized code is empty.")
                    msg = f"Empty TOKENIZED from node CODE {node_code}"
                    logger.log_warning('embeddings', msg)
                    continue
                tokens.extend(tokenized_code)
                groups.extend([i] * len(tokenized_code))

            vectorized_code = self.get_vectors(tokens, node)
            assert len(vectorized_code) == len(tokens) and len(vectorized_code) == len(
                groups
            ), f"{len(vectorized_code)} {len(tokens)} {len(groups)}"
            cached_vectorized_code = []
            l = 0
            while l < len(groups):
                for r in range(l, len(groups)):
                    if groups[l] != groups[r]:
                        r -= 1
                        break
                cached_vectorized_code.append(vectorized_code[l : r + 1])
                l = r + 1
        p = 0
        for n_id, node in nodes.items():
            # Get node's code
            node_code = node.get_code()
            # Tokenize the code
            try:
                tokenized_code = tokenizer(node_code, True)
            except Exception as e:
                print(node_code)
                print(e)
                continue
            if not tokenized_code:
                # print(f"Dropped node {node}: tokenized code is empty.")
                msg = f"Empty TOKENIZED from node CODE {node_code}"
                logger.log_warning('embeddings', msg)
                continue
            if not self.is_tokenize_func:
                # Get each token's learned embedding vector
                vectorized_code = np.array(self.get_vectors(tokenized_code, node))
                # The node's source embedding is the average of it's embedded tokens
            else:
                # vectorized_code = self.get_vectors(tokenized_code, node)
                vectorized_code = cached_vectorized_code[p]
                p += 1
            source_embedding = np.mean(vectorized_code, 0)
            # The node representation is the concatenation of label and source embeddings
            embedding = np.concatenate(
                (np.array([node.type]), source_embedding), axis=0
            )
            embeddings.append(embedding)
        # print(node.label, node.properties.properties.get("METHOD_FULL_NAME"))

        return np.array(embeddings)

    # fromTokenToVectors
    def get_vectors(self, tokenized_code, node):
        if self.is_tokenize_func:
            result = self.tokenize(tokenized_code)
            return result
        vectors = []

        for token in tokenized_code:
            if token in self.w2v_keyed_vectors.key_to_index:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                # print(node.label, token, node.get_code(), tokenized_code)
                vectors.append(np.zeros(self.kv_size))
                if node.label not in [
                    "Identifier",
                    "Literal",
                    "MethodParameterIn",
                    "MethodParameterOut",
                ]:
                    msg = f"No vector for TOKEN {token} in {node.get_code()}."
                    logger.log_warning('embeddings', msg)

        return vectors


class GraphsEmbedding:
    def __init__(self, edge_type):
        self.edge_type = edge_type

    def __call__(self, nodes):
        connections = self.nodes_connectivity(nodes)

        return torch.tensor(connections).long()

    # nodesToGraphConnectivity
    def nodes_connectivity(self, nodes):
        # nodes are ordered by line and column
        coo = [[], []]

        for node_idx, (node_id, node) in enumerate(nodes.items()):
            if node_idx != node.order:
                raise Exception("Something wrong with the order")

            for e_id, edge in node.edges.items():
                if edge.type != self.edge_type:
                    continue

                if edge.node_in in nodes and edge.node_in != node_id:
                    coo[0].append(nodes[edge.node_in].order)
                    coo[1].append(node_idx)

                if edge.node_out in nodes and edge.node_out != node_id:
                    coo[0].append(node_idx)
                    coo[1].append(nodes[edge.node_out].order)

        return coo


def nodes_to_input(
    nodes,
    target,
    max_nodes,
    keyed_vectors,
    edge_type,
    vector_size,
    is_tokenize_func=False,
):
    nodes_embedding = NodesEmbedding(
        max_nodes, keyed_vectors, vector_size, is_tokenize_func=is_tokenize_func
    )
    graphs_embedding = GraphsEmbedding(edge_type)
    label = torch.tensor([target]).float()

    return Data(x=nodes_embedding(nodes), edge_index=graphs_embedding(nodes), y=label)
