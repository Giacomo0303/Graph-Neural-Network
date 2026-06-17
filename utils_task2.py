from torch_geometric.datasets import Reddit
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader

class RedditLinkDataset:
    def __init__(self, path):
        self.path = path
        self.dataset = Reddit(root=self.path)[0]
        self.train_set, self.val_set, self.test_set = self.split_edges()

    def split_edges(self, val_size=0.1, test_size=0.2):
        
        #split degli archi in train, validation e test.
        link_split = RandomLinkSplit(
            num_val=val_size,
            num_test=test_size,
            is_undirected=True,
            add_negative_train_samples=True, # verranno aggiunti dopo nel loader di train, val e set li hanno
            neg_sampling_ratio=1.0, # 1 campione negativo per ogni campione positivo
        )

        train_links, val_links, test_links = link_split(self.dataset)

        print(f"Archi totali: {self.dataset.edge_index.size(1)//2}")
        print(f"Archi positivi in Train: {train_links.edge_label_index.size(1)}")
        print(f"Coppie di archi pos/neg in Val: {val_links.edge_label_index.size(1)}")
        print(f"Coppie di archi pos/neg in Test: {test_links.edge_label_index.size(1)}")

        return train_links, val_links, test_links
    

    # usiamo lo stesso vicinato del task sul node prediction
    def get_link_loaders(self, batch_size=512, num_neighbors=[25, 10]):
        train_loader = LinkNeighborLoader(
            self.train_set,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            edge_label_index=self.train_set.edge_label_index,
            edge_label=self.train_set.edge_label,
            neg_sampling_ratio=0.0, # Genera i negativi dinamicamente a ogni batch di train!  # Considera solo gli archi di train
            shuffle=True,
            num_workers=6,
        )

        val_loader = LinkNeighborLoader(
            self.val_set,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            edge_label_index=self.val_set.edge_label_index,
            edge_label=self.val_set.edge_label,
            neg_sampling_ratio=0.0,
            shuffle=False,
        )

        test_loader = LinkNeighborLoader(
            self.test_set,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            edge_label_index=self.test_set.edge_label_index,
            edge_label=self.test_set.edge_label,
            neg_sampling_ratio=0.0,
            shuffle=False,
        )

        return train_loader, val_loader, test_loader