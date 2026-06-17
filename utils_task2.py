
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader

class RedditLinkDataset:
    def __init__(self, path, reduction_factor=0.01): 
        """
        manager di dati per il Task 2 con filtro di campionamento preventivo.
        reduction_factor=0.05 significa che terremo solo il 5% degli archi totali.
        """
        self.path = path
        
        print("Caricamento del macro-grafo originale di Reddit...")
        full_graph = Reddit(root=self.path)[0] 
        print(f"Grafo originale: {full_graph.num_nodes} nodi, {full_graph.num_edges} archi")
     
        # Selezioniamo casualmente una frazione (5%) degli indici degli archi totali
        num_edges_to_keep = int(full_graph.num_edges * reduction_factor)
        perm = torch.randperm(full_graph.num_edges)[:num_edges_to_keep]
        
        # Ricostruiamo un oggetto Data compatto mantenendo intatte le feature (x) e i target (y)
        self.dataset = Data(
            x=full_graph.x,
            y=full_graph.y,
            edge_index=full_graph.edge_index[:, perm]
        )
        print(f"--> Grafo ridotto al {reduction_factor*100}%: {self.dataset.num_nodes} nodi, {self.dataset.num_edges} archi")
        
        # Avviamo lo split degli archi sullo scheletro ridotto
        self.train_set, self.val_set, self.test_set = self.split_edges()

    def split_edges(self, val_size=0.1, test_size=0.2):
        # Splitting degli archi in train, validation e test tramite utility PyG [cite: 25]
        link_split = RandomLinkSplit(
            num_val=val_size,
            num_test=test_size,
            is_undirected=True,
            add_negative_train_samples=False, # Generati dinamicamente nel loader di train
            neg_sampling_ratio=1.0,           # 1 campione negativo per ogni positivo [cite: 26]
        )

        train_links, val_links, test_links = link_split(self.dataset)

        print(f"\n--- STATISTICHE STRUTTURALI DEL GRAFO RIDOTTO ---")
        print(f"Archi positivi (Supervisione) in Train: {train_links.edge_label_index.size(1)}")
        print(f"Coppie totali (Pos+Neg) in Val: {val_links.edge_label_index.size(1)}")
        print(f"Coppie totali (Pos+Neg) in Test: {test_links.edge_label_index.size(1)}")

        return train_links, val_links, test_links
    
    def get_link_loaders(self, batch_size=4096, num_neighbors=[15, 10]):
        # Avendo alleggerito il grafo originario, un batch_size di 4096 è perfetto 
        # per far correre i cicli mantenendo la CPU totalmente reattiva.
        
        train_loader = LinkNeighborLoader(
            self.train_set,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            edge_label_index=self.train_set.edge_label_index,
            neg_sampling_ratio=1.0, 
            shuffle=True,
            num_workers=0, # Obbligatorio su Windows per evitare crash di multiprocessing
        )

        val_loader = LinkNeighborLoader(
            self.val_set,
            num_neighbors=num_neighbors,
            batch_size=batch_size, 
            edge_label_index=self.val_set.edge_label_index,
            edge_label=self.val_set.edge_label,
            neg_sampling_ratio=0.0, 
            shuffle=False,
            num_workers=0,
        )

        test_loader = LinkNeighborLoader(
            self.test_set,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            edge_label_index=self.test_set.edge_label_index,
            edge_label=self.test_set.edge_label,
            neg_sampling_ratio=0.0,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, val_loader, test_loader