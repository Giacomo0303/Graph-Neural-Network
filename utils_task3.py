from torch_geometric.datasets import Reddit
from torch_geometric.loader import ClusterData
from sklearn.model_selection import train_test_split
import torch
import numpy as np


class RedditSubGraphDataset:
    def __init__(self, path, num_graphs, task_type):
        self.path = path
        self.num_graphs = num_graphs
        self.task_type = task_type

        full_graph = Reddit(root=self.path)[0]

        self.train_graphs, self.val_graphs, self.test_graphs = self.partition_graph(full_graph)

    def partition_graph(self, full_graph):
        cluster_data = ClusterData(
            full_graph, num_parts=self.num_graphs, recursive=False
        )

        if self.task_type == "a":
            graphs = self.calcola_label_task_a(cluster_data)
        elif self.task_type == "b":
            graphs = self.calcola_label_task_b(cluster_data)
        else:
            raise ValueError("Tipo di task non valido. Scegliere 'a' o 'b'.")
        
        #split stratificato per mantenere la distribuzione delle classi nei set di train, val e test 

        labels = [subgraph.y.item() for subgraph in graphs]

        train_graphs, temp_graphs, _, y_temp = train_test_split(
            graphs, labels, test_size=0.20, stratify=labels, random_state=42
        )

        val_graphs, test_graphs = train_test_split(
            temp_graphs, test_size=0.50, stratify=y_temp, random_state=42
        )

        return train_graphs, val_graphs, test_graphs

    def calcola_label_task_a(self, cluster_data):
        densita_clusters = []

        # Calcolo della densità interna di ogni singolo sotto-grafo
        for i in range(len(cluster_data)):
            subgraph = cluster_data[i]
            v = subgraph.num_nodes
            e = subgraph.num_edges

            if v > 1:
                # Formula della densità topologica per grafi
                d = e / (v * (v - 1))
            else:
                d = 0.0

            densita_clusters.append(d)

        # Calcolo della densità media globale
        media_globale = np.mean(densita_clusters)
        print(f"Densità interna media dei sotto-grafi: {media_globale:.6f}\n")

        # Assegnazione delle label
        grafi_finali = []
        conteggio_classi = {0: 0, 1: 0}

        for i in range(len(cluster_data)):
            subgraph = cluster_data[i]
            densita_corrente = densita_clusters[i]

            # Target = 1 se sopra la media, 0 se sotto la media
            if densita_corrente > media_globale:
                label = 1
            else:
                label = 0

            # Aggiorniamo il dizionario di controllo dello sbilanciamento
            conteggio_classi[label] += 1

            # Trasformiamo y da vettore di nodi a scalare di grafo
            subgraph.y = torch.tensor(label, dtype=torch.long)

            grafi_finali.append(subgraph)

        print(f"BILANCIAMENTO DELLE CLASSI")
        print(f"Classe 0 (Sotto la media): {conteggio_classi[0]} sotto-grafi")
        print(f"Classe 1 (Sopra la media): {conteggio_classi[1]} sotto-grafi")

        return grafi_finali

    def calcola_label_task_b(self, cluster_data):
        grafi_finali = []

        for i in range(len(cluster_data)):
            subgraph = cluster_data[i]  # Estrae il sotto-grafo i-esimo isolato

            # Calcolo del target sulla base della community di maggioranza
            values, _ = torch.bincount(subgraph.y).max(dim=0)

            subgraph.y = values  # Ora y è un singolo scalare per l'intero sotto-grafo

            grafi_finali.append(subgraph)

        return grafi_finali
