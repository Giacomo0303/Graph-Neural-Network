import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


class RedditLinkDataset:
    def __init__(self, path, reduction_factor=0.01):
        self.path = path

        print("Caricamento del macro-grafo originale di Reddit...")
        full_graph = Reddit(root=self.path)[0]
        print(
            f"Grafo originale: {full_graph.num_nodes} nodi, {full_graph.num_edges} archi"
        )
        del full_graph

        # il grafo originale contiene troppi archi che pesano sul costo computazionale del training
        # sottocampioniamo prendendo un sotto-grafo

        # carichiamo il sottografo generato con subsample_graph()
        self.dataset = torch.load("subgraph_task2.pt", weights_only=False)

        print("\n=== SOTTOGRAFO PER TASK 2 ===")
        print(f"Nodi nel sottografo: {self.dataset.num_nodes}")
        print(f"Archi nel sottografo: {self.dataset.num_edges}")
        print("==============================================")

        # Avviamo lo split degli archi sul grafo ridotto
        self.train_set, self.val_set, self.test_set = self.split_edges()

    def subsample_graph(self, full_graph, num_sub_graphs=20):
        
        # viene usato un algoritmo METIS per suddividere il grafo
        cluster_data = ClusterData(
            full_graph, num_parts=num_sub_graphs, recursive=False
        )

        # Estraiamo il primo subgraph
        subgraph = cluster_data[0]
        return subgraph

    def split_edges(self, val_size=0.1, test_size=0.2):
        # Splitting degli archi in train, validation e test
        link_split = RandomLinkSplit(
            num_val=val_size,
            num_test=test_size,
            is_undirected=True, #Reddit è un grafo non orientato
            add_negative_train_samples=False,  # Generati dinamicamente nel loader di train per regolarizzare
            neg_sampling_ratio=1.0,  # 1 campione negativo per ogni positivo in modo da bilanciare il dataset
        )

        train_links, val_links, test_links = link_split(self.dataset)

        print(f"\n--- STATISTICHE STRUTTURALI DEL GRAFO RIDOTTO ---")
        print(
            f"Archi positivi (Supervisione) in Train: {train_links.edge_label_index.size(1)}"
        )
        print(f"Coppie totali (Pos+Neg) in Val: {val_links.edge_label_index.size(1)}")
        print(f"Coppie totali (Pos+Neg) in Test: {test_links.edge_label_index.size(1)}")

        return train_links, val_links, test_links

    def get_link_loaders(self, batch_size=4096, num_neighbors=[15, 10]):

        # Creazione dei LinkNeighborLoader per train, validation e test set
        train_loader = LinkNeighborLoader(
            self.train_set,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            edge_label_index=self.train_set.edge_label_index,
            neg_sampling_ratio=1.0, #aggiungiamo i negativi tramite il loader
            shuffle=True,
            num_workers=0,
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


def train_epoch(
    epoch_idx, model, train_loader, optimizer, loss_fn, device, scaler=None
):
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Epoca {epoch_idx:02d} [Train Link]", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            # Encoder usa batch.edge_index, decoder predice su batch.edge_label_index
            out = model(batch.x, batch.edge_index, batch.edge_label_index)
            loss = loss_fn(out, batch.edge_label)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        num_graphs_edges = batch.edge_label.size(0)
        total_loss += loss.item() * num_graphs_edges
        total_samples += num_graphs_edges

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / total_samples


def evaluate(model, val_loader, loss_fn, device, test=False):
    model.eval()
    all_preds = []
    all_targets = []

    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(val_loader, desc="Valutazione Link...", leave=False):
        batch = batch.to(device)

        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.edge_label_index)
            loss = loss_fn(out, batch.edge_label)

            # Trasformiamo i logit in probabilità pure [0, 1]
            probs = torch.sigmoid(out)

        num_samples = batch.edge_label.size(0)
        total_loss += loss.item() * num_samples
        total_samples += num_samples

        all_preds.append(probs.cpu())
        all_targets.append(batch.edge_label.cpu())

    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()

    avg_val_loss = total_loss / total_samples
    y_pred_binary = (y_pred >= 0.5).astype(int)

    # Calcolo delle metriche richieste
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)

    cm = confusion_matrix(y_true, y_pred_binary) if test else None

    return {
        "val_loss": avg_val_loss,
        "roc_auc": auc,
        "average_precision": ap,
        "balanced_accuracy": balanced_acc,
        "f1_score": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred if test else None,
        "y_true": y_true if test else None,
    }


def train_loop(
    num_epochs,
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    best_model_path,
    scaler=None,
    patience=5,
):
    print("\n--- AVVIO LOOP DI ADDESTRAMENTO LINK PREDICTION ---")
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    for epoch in range(1, num_epochs + 1):
        loss_t = train_epoch(
            epoch, model, train_loader, optimizer, loss_fn, device, scaler
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        print(
            f"Epoca: {epoch:02d}/{num_epochs:02d} |"
            f"Loss Train: {loss_t:.4f} | "
            f"Loss Val: {val_metrics['val_loss']:.4f} | "
            f"Val AUC-ROC: {val_metrics['roc_auc']:.4f} |"
            f"Val AP: {val_metrics['average_precision']:.4f} |"
            f"Val Balanced Acc: {val_metrics['balanced_accuracy']:.4f} |"
            f"Val F1: {val_metrics['f1_score']:.4f}"
        )

        train_losses.append(loss_t)
        val_losses.append(val_metrics["val_loss"])

        if val_metrics["val_loss"] < best_val_loss:
            patience_counter = 0
            best_val_loss = val_metrics["val_loss"]
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Modello salvato con Loss Val Migliore: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(
                f"--> Nessun miglioramento. Contatore di pazienza: {patience_counter}/{patience}"
            )
            if patience_counter >= patience:
                print("Early stopping attivato. Interruzione dell'addestramento.")
                break

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


class GCNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.25):
        super().__init__()
        # ENCODER (GCN a 2 strati)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # PROIETTORE (MLP lineare per raffinare gli embedding dei nodi)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.dropout = dropout

    def encode(self, x, edge_index):
        # Genera le rappresentazioni latenti di base per tutti i nodi
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # Prende gli embedding, li proietta separatamente e calcola la somiglianza
        nodes_src = edge_label_index[0]
        nodes_dst = edge_label_index[1]

        # Proiezione nello spazio comune
        first_emb = self.projector(z[nodes_src])
        second_emb = self.projector(z[nodes_dst])

        # Prodotto scalare tra i vettori dei nodi sorgente e destinazione per ottenere un punteggio di similarità
        # In questo modo la rete impara a creare embedding che massimizzano la somiglianza per coppie di nodi con un arco esistente
        # e minimizzano la somiglianza per coppie senza arco

        logit = torch.sum(first_emb * second_emb, dim=-1)

        return logit

    def forward(self, x, edge_index, edge_label_index):
        # L'encoder estrae l'embedding dei nodi
        z = self.encode(x, edge_index)

        # Il decoder confronta le coppie specifiche richieste dal batch
        out = self.decode(z, edge_label_index)

        return out


class SAGEConvLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.25):
        super().__init__()
        # ENCODER (SAGEConv a 2 strati per il messaggio topologico)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # PROIETTORE (MLP lineare per raffinare gli embedding dei nodi)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.dropout = dropout

    def encode(self, x, edge_index):
        # Genera le rappresentazioni latenti per tutti i nodi
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # Prende gli embedding, li proietta separatamente e calcola la somiglianza
        nodes_src = edge_label_index[0]
        nodes_dst = edge_label_index[1]

        # Proiezione in uno spazio vettoriale comune
        first_emb = self.projector(z[nodes_src])
        second_emb = self.projector(z[nodes_dst])

        # Prodotto scalare tra i vettori embedding dei nodi sorgente e destinazione per ottenere un punteggio di similarità
        # In questo modo la rete impara a creare embedding che massimizzano la somiglianza per coppie di nodi con un arco esistente
        # e minimizzano la somiglianza per coppie senza arco (Contrastive learning)

        logit = torch.sum(first_emb * second_emb, dim=-1)

        return logit

    def forward(self, x, edge_index, edge_label_index):
        # L'encoder estrae l'embedding dei nodi
        z = self.encode(x, edge_index)

        # Il decoder confronta le coppie specifiche richieste dal batch
        out = self.decode(z, edge_label_index)

        return out


class GATLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.25):
        super().__init__()
        # ENCODER (GATConv a 2 strati per il messaggio topologico)
        self.conv1 = GATv2Conv(in_channels, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels)

        # PROIETTORE (MLP lineare per raffinare gli embedding dei nodi)
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.dropout = dropout

    def encode(self, x, edge_index):
        # Genera le rappresentazioni latenti di base per tutti i nodi
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # Prende gli embedding, li proietta separatamente e calcola la somiglianza
        nodes_src = edge_label_index[0]
        nodes_dst = edge_label_index[1]

        # Proiezione nello spazio comune
        first_emb = self.projector(z[nodes_src])
        second_emb = self.projector(z[nodes_dst])

        # Prodotto scalare tra i vettori embedding dei nodi sorgente e destinazione per ottenere un punteggio di similarità
        # In questo modo la rete impara a creare embedding che massimizzano la somiglianza per coppie di nodi con un arco esistente
        # e minimizzano la somiglianza per coppie senza arco

        logit = torch.sum(first_emb * second_emb, dim=-1)

        return logit

    def forward(self, x, edge_index, edge_label_index):
        # L'encoder estrae l'embedding dei nodi
        z = self.encode(x, edge_index)

        # Il decoder confronta le coppie specifiche richieste dal batch
        out = self.decode(z, edge_label_index)

        return out
