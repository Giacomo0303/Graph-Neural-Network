from torch_geometric.datasets import Reddit
from torch_geometric.utils import degree, subgraph, to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from types import SimpleNamespace
from numpy.random import randint
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import torch.nn.functional as F



class RedditDataset():
    def __init__(self, path):
        self.path = path
        self.dataset = Reddit(root="Reddit")[0]
        self.stats = self.get_stats()

    def get_stats(self):
        num_nodes = self.dataset.num_nodes
        num_edges = self.dataset.num_edges
        num_features = self.dataset.num_features

        train_size = int(self.dataset.train_mask.sum())
        val_size = int(self.dataset.val_mask.sum())
        test_size = int(self.dataset.test_mask.sum())

        num_classes_node = self.dataset.y.max() + 1

        # Calcola la cardinalità di tutte le classi
        classes_cardinality = torch.bincount(self.dataset.y)

        # Calcolo del grado medio/minimo/massimo
        node_degrees = degree(self.dataset.edge_index[0], num_nodes=num_nodes)

        print("Numero di nodi: ", num_nodes)
        print("Numero di archi: ", num_edges)
        print("Dimensionalità delle features: ", num_features)
        print(
            "Di queste, le prime 300 l'embedding di Glove del titolo, le seconde 300 l'embedding di Glove medio di tutti i commenti")
        print("La feature 601 è lo score di reddit e la 602 è il numero di commenti")
        print("--------------------------------------------------------------------")
        print("Analisi della cardinalità delle classi e bilanciamento del dataset")
        for c, count in enumerate(classes_cardinality):
            print(f"classe {c:2d} : {count:6d} -- {100 * count / num_nodes:.2f}%")
        print("--------------------------------------------------------------------")
        print("Grado dei nodi")
        print(f"Medio: {node_degrees.mean().item():.2f}")
        print(f"Massimo: {node_degrees.max().item()}")
        print(f"Minimo: {node_degrees.min().item()}")

        return SimpleNamespace(
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_features=num_features,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            classes=classes_cardinality
        )

    def visualize(self, max_nodes=200):
        node_index = randint(0, self.stats.num_nodes)
        row, col = self.dataset.edge_index

        # Selezioniamo i nodi di destinazione dove la sorgente è il nostro target_node
        neighbours = col[row == node_index].unique()

        # Tronchiamo i vicini se sono troppi
        if len(neighbours) > (max_nodes - 1):
            neighbours = neighbours[:max_nodes - 1]

        # Uniamo il nodo target con i vicini
        local_neighbours = torch.cat([torch.tensor([node_index], device=neighbours.device), neighbours])

        # Estrazione del sotto-grafo
        # relabel_nodes=True rinomina i nodi estratti da 0 a N-1
        edge_index_sub, _ = subgraph(local_neighbours, self.dataset.edge_index, relabel_nodes=True)

        # Creiamo un oggetto Data per la conversione con Networkx
        sub_graph = to_networkx(Data(edge_index=edge_index_sub, num_nodes=local_neighbours.size(0)), to_undirected=True)

        # Rimuovi nodi isolati
        sub_graph.remove_nodes_from(list(nx.isolates(sub_graph)))

        print("VALUTAZIONE DELLA DENSITÀ LOCALE")
        print(f"Nodo Target Centrale analizzato: {node_index}")
        print(f"Cardinalità del sotto-grafo: {sub_graph.number_of_nodes()} nodi")
        print(f"Numero di archi locali interni: {sub_graph.number_of_edges()}")

        plt.figure(figsize=(8, 8))  # Formato quadrato, perfetto per la distribuzione spring

        # k=0.3 distanzia maggiormente i nodi tra loro rispetto allo standard
        pos = nx.spring_layout(sub_graph, k=0.3, iterations=50)
        gradi_locali = dict(sub_graph.degree())

        # Disegno dei Nodi: Dimensione fissa + Contorno di separazione
        nx.draw_networkx_nodes(
            sub_graph, pos,
            node_size=80,  # Tutti i nodi hanno rigorosamente la stessa dimensione
            node_color=list(gradi_locali.values()),  # Il colore mappa il grado per mostrare la struttura
            cmap=plt.cm.viridis,  # Palette ad alto contrasto e molto pulita
            alpha=0.9,  # Leggera trasparenza per intravedere gli archi sotto
            edgecolors='black',  # BORDO NERO: Fondamentale per distinguere nodi adiacenti
            linewidths=0.6  # Spessore del bordo del nodo
        )

        # Disegno degli Archi: Sottili e discreti per non appesantire la vista
        nx.draw_networkx_edges(
            sub_graph, pos,
            alpha=0.18,  # Molto chiari per far risaltare i nodi
            edge_color="gray",
            width=0.8  # Spessore ridotto anti-caos
        )

        plt.title(f"Sottografo del nodo {node_index}", fontsize=12, pad=10)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # Parametri di campionamento dei vicini derivati dalla Sezione 4 del paper GraphSAGE.
    def get_node_loaders(self, batch_size=512, num_neighbors=[25, 10]):
        train_loader = NeighborLoader(
            self.dataset,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=self.dataset.train_mask,  # Considera solo i nodi di train tramite maschera
            shuffle=True, num_workers = 3
        )

        val_loader = NeighborLoader(
            self.dataset,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=self.dataset.val_mask,
            shuffle=False
        )

        test_loader = NeighborLoader(
            self.dataset,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=self.dataset.test_mask,
            shuffle=False
        )

        return train_loader, val_loader, test_loader


class GCNmodel(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, dropout = 0.5):
        super().__init__()

        # La prima conv guarda solo ai primi vicini, la seconda anche ai vicini dei vicini
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

        self.mlp1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.mlp2 = torch.nn.Linear(hidden_size // 2, out_channels)

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)


        x = self.mlp1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.mlp2(x)
        return x
