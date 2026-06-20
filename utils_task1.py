from torch_geometric.datasets import Reddit
from torch_geometric.utils import degree, subgraph, to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from types import SimpleNamespace
from numpy.random import randint
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    classification_report,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


class RedditDataset:
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
        # Il conteggio va diviso per due in quanto il grafo è non orientato e la matrice di adiacenza è simmetrica
        print("Numero di archi: ", num_edges//2)
        print("Dimensionalità delle features: ", num_features)
        print(
            "Di queste, le prime 300 l'embedding di Glove del titolo del post, le seconde 300 l'embedding di Glove medio di tutti i commenti"
        )
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
            classes=classes_cardinality,
        )

    def visualize(self, max_nodes=200):
        # selezioniamo un nodo in maniera randomica
        node_index = randint(0, self.stats.num_nodes)

        row, col = self.dataset.edge_index

        # Selezioniamo i nodi di destinazione dove la sorgente è il nostro target_node
        neighbours = col[row == node_index].unique()

        # Tronchiamo i vicini se sono troppi
        if len(neighbours) > (max_nodes - 1):
            neighbours = neighbours[: max_nodes - 1]

        # Uniamo il nodo target con i vicini
        local_neighbours = torch.cat(
            [torch.tensor([node_index], device=neighbours.device), neighbours]
        )

        # Estrazione del sotto-grafo
        # relabel_nodes=True rinomina i nodi estratti da 0 a N-1
        edge_index_sub, _ = subgraph(
            local_neighbours, self.dataset.edge_index, relabel_nodes=True
        )

        # Creiamo un oggetto Data per la conversione con Networkx
        sub_graph = to_networkx(
            Data(edge_index=edge_index_sub, num_nodes=local_neighbours.size(0)),
            to_undirected=True,
        )

        # Rimuovi nodi isolati
        sub_graph.remove_nodes_from(list(nx.isolates(sub_graph)))

        print("VALUTAZIONE DELLA DENSITÀ LOCALE")
        print(f"Nodo Target Centrale analizzato: {node_index}")
        print(f"Cardinalità del sotto-grafo: {sub_graph.number_of_nodes()} nodi")
        print(f"Numero di archi locali interni: {sub_graph.number_of_edges()}")

        plt.figure(
            figsize=(8, 8)
        )  # Formato quadrato, perfetto per la distribuzione spring

        # k=0.3 distanzia maggiormente i nodi tra loro rispetto allo standard
        pos = nx.spring_layout(sub_graph, k=0.3, iterations=50)
        gradi_locali = dict(sub_graph.degree())

        # Disegno dei Nodi: Dimensione fissa + Contorno di separazione
        nx.draw_networkx_nodes(
            sub_graph,
            pos,
            node_size=80,  # Tutti i nodi hanno rigorosamente la stessa dimensione
            node_color=list(
                gradi_locali.values()
            ),  # Il colore mappa il grado per mostrare la struttura
            cmap=plt.cm.viridis,  # Palette ad alto contrasto e molto pulita
            alpha=0.9,  # Leggera trasparenza per intravedere gli archi sotto
            edgecolors="black",  # BORDO NERO: Fondamentale per distinguere nodi adiacenti
            linewidths=0.6,  # Spessore del bordo del nodo
        )

        # Disegno degli Archi: Sottili e discreti per non appesantire la vista
        nx.draw_networkx_edges(
            sub_graph,
            pos,
            alpha=0.18,  # Molto chiari per far risaltare i nodi
            edge_color="gray",
            width=0.8,  # Spessore ridotto anti-caos
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
            shuffle=True,
            num_workers=3,
        )

        val_loader = NeighborLoader(
            self.dataset,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=self.dataset.val_mask,
            shuffle=False,
        )

        test_loader = NeighborLoader(
            self.dataset,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=self.dataset.test_mask,
            shuffle=False,
        )

        return train_loader, val_loader, test_loader


def train_epoch(
    epoch_idx, model, train_loader, optimizer, loss_fn, device, stats, scaler=None
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoca {epoch_idx:02d} [Train AMP]", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(batch.x, batch.edge_index)
            # calcolo della loss solo sui nodi effettivi del batch senza considerare i vicini che protrebbero essere di validation o test, evita data leakage
            loss = loss_fn(out[: batch.batch_size], batch.y[: batch.batch_size])

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch.batch_size
        pbar.set_postfix({"Loss_batch": f"{loss.item():.4f}"})

    return total_loss / stats.train_size


def evaluate(model, val_loader, loss_fn, device, test=False):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(val_loader, desc="Valutazione...", leave=False):
        batch = batch.to(device)
        with torch.no_grad():
            out = model(batch.x, batch.edge_index)
            loss = loss_fn(out[: batch.batch_size], batch.y[: batch.batch_size])
        total_loss += loss.item() * batch.batch_size
        total_samples += batch.batch_size

        preds = torch.nn.LogSoftmax(dim=-1)(out[: batch.batch_size]).argmax(dim=-1)
        targets = batch.y[: batch.batch_size]

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        probs = F.softmax(out[: batch.batch_size], dim=-1)
        all_probs.append(probs.cpu())

    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy()
    y_pred_prob = torch.cat(all_probs, dim=0).numpy()

    f1_weight = f1_score(y_true, y_pred, average="weighted")
    precision_weighted = precision_score(y_true, y_pred, average="weighted")
    recall_weighted = recall_score(y_true, y_pred, average="weighted")
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    avg_loss = total_loss / total_samples

    cm = confusion_matrix(y_true, y_pred) if test else None

    return SimpleNamespace(
        f1_weighted=f1_weight,
        precision_weighted=precision_weighted,
        recall_weighted=recall_weighted,
        balanced_acc=balanced_acc,
        avg_loss=avg_loss,
        confusion_matrix=cm,
        y_pred=y_pred if test else None,
        y_true=y_true if test else None,
        y_pred_prob=y_pred_prob if test else None,
    )


def train_loop(
    num_epochs,
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    stats,
    patience=5,
    best_model_path="best_model.pth",
    scaler=None,
):
    print("\n--- AVVIO LOOP DI ADDESTRAMENTO ---")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, num_epochs + 1):
        loss_corrente = train_epoch(
            epoch, model, train_loader, optimizer, loss_fn, device, stats, scaler
        )
        eval_stats = evaluate(model, val_loader, loss_fn, device)

        print(
            f"Epoca: {epoch:02d}/{num_epochs:02d} | "
            f"Loss Train: {loss_corrente:.2f} | "
            f"Val Loss: {eval_stats.avg_loss:.2f} | "
            f"Val F1 weighted: {eval_stats.f1_weighted:.2f} | "
            f"Val Precision: {eval_stats.precision_weighted:.2f} | "
            f"Val Recall: {eval_stats.recall_weighted:.2f} |"
            f"Val Balanced Acc: {eval_stats.balanced_acc:.2f}"
        )

        train_losses.append(loss_corrente)
        val_losses.append(eval_stats.avg_loss)

        if eval_stats.avg_loss < best_val_loss:
            best_val_loss = eval_stats.avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Nuovo best model salvato (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\nEarly stopping attivato all'epoca {epoch}.")
                break
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }


def plot_history(history, title):
    train_loss, val_loss = history["train_losses"], history["val_losses"]
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        color="#ff7f0e",
        linewidth=2,
        linestyle="--",
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epoche", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.show()


class GCNmodel(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, dropout=0.3):
        super().__init__()

        # La prima conv guarda solo ai primi vicini, la seconda anche ai vicini dei vicini
        self.conv1 = GCNConv(in_channels, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)

        self.linear1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.linear2 = torch.nn.Linear(hidden_size // 2, out_channels)

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear2(x)
        return x


class SAGEConvModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, dropout=0.3):
        super().__init__()

        # Con project = True, il modello riproietta le feature aggreagate dal nodo e dal vicinato, in un nuovo spazio
        # introducendo non linearità e potere espressivo
        self.conv1 = SAGEConv(in_channels, hidden_size, project=True)
        self.conv2 = SAGEConv(hidden_size, hidden_size, project=True)

        self.linear1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.linear2 = torch.nn.Linear(hidden_size // 2, out_channels)

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear2(x)
        return x


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, dropout=0.3):
        super().__init__()

        # Usiamo GATv2Conv per una maggiore flessibilità e capacità rispetto al GAT standard
        # inoltre da doc ufficiale essa risolve il problema legato all'attenzione statica.
        self.conv1 = GATv2Conv(in_channels, hidden_size)
        self.conv2 = GATv2Conv(hidden_size, hidden_size)

        self.linear1 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.linear2 = torch.nn.Linear(hidden_size // 2, out_channels)

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear2(x)
        return x
