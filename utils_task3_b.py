from collections import Counter
from torch_geometric.datasets import Reddit
from torch_geometric.loader import ClusterData
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool,global_max_pool
from sklearn.metrics import  balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

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

        labels_grezze = [subgraph.y.item() for subgraph in graphs]
    
        # 2. Conto quante volte compare ogni community
        conteggio_classi = Counter(labels_grezze)
        
        # 3. FILTRAGGIO CRUCIALE: Teniamo solo i grafi la cui classe compare almeno 2 volte
        graphs_filtrati = []
        labels_filtrate = []
        
        for subgraph, label in zip(graphs, labels_grezze):
            if conteggio_classi[label] >= 10: # <-- Soglia di predicibilità
                graphs_filtrati.append(subgraph)
                labels_filtrate.append(label)
                
        print(f"Rimossi {len(graphs) - len(graphs_filtrati)} grafi impredicibili.")
        print(f"Grafi totali rimasti per il Task B: {len(graphs_filtrati)}")

        # 4. SPLIT STRATIFICATO BLINDATO (Ora funziona al 100% perché non ci sono membri singoli)
        train_graphs, temp_graphs, _, y_temp = train_test_split(
            graphs_filtrati, 
            labels_filtrate, 
            test_size=0.20, 
            stratify=labels_filtrate, # <-- Ritorna attivo e sicuro!
            random_state=42
        )

        val_graphs, test_graphs = train_test_split(
            temp_graphs, 
            test_size=0.50, 
            stratify=y_temp, 
            random_state=42
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

def train_epoch(model, loader, optimizer, loss_fn, device,scaler=None):
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(loader,desc= "Training..."):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * batch.num_graphs
        
    # Calcoliamo la loss media reale per grafo
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    model.eval()
    val_loss=0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader,desc= "Evaluating..."):
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index, batch.batch)

            loss = loss_fn(out, batch.y)
            
            preds = torch.nn.LogSoftmax(dim=-1)(out).argmax(dim=-1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

            val_loss += loss.item() * batch.num_graphs
            
    # Concateniamo i vettori di tutti i batch
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    acc = balanced_accuracy_score(all_targets, all_preds)
    
    # 'weighted' calcola l'F1 per ogni classe e ne fa la media pesata sul numero di campioni,
    f1_bilanciato = f1_score(all_targets, all_preds, average='weighted')
    
    return {
        "balanced_accuracy": acc,
        "f1_score": f1_bilanciato,
        "val_loss": val_loss / len(loader.dataset)
    }


def train_loop(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs,best_model_path, scaler=None,patience=5):
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device,scaler)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_metrics['val_loss'])
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_metrics['val_loss']:.4f} - Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f} - F1 Score: {val_metrics['f1_score']:.4f}")


        # Salvataggio del modello migliore basato sulla loss di validazione
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_model_state = model.state_dict()
            torch.save(best_model_state, best_model_path)
            print(f"Nuovo miglior modello salvato con Val Loss: {best_val_loss:.4f}")
            patience_counter = 0 
        else:
            patience_counter += 1
            print(f"Nessun miglioramento. Contatore di pazienza: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping attivato.")
                break
        print("-----------------------------------------------")
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
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





class HierarchicalGCNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        # BLOCCO 1
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # Tiene solo il 50% dei nodi (ratio=0.5) basandosi su un punteggio di rilevanza appreso
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5) 
        
        # BLOCCO 2
        self.conv2 = GI(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)
        
        # Classificatore finale
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Il pooling gerarchico taglia i nodi meno importanti.
        # Restituisce il nuovo x, il nuovo edge_index ristretto e il batch aggiornato
        #x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
     
        #x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        
        # Solo ora applichiamo il Global Mean Pool per condesare le informazioni di tutti i nodi rimasti in un singolo vettore per grafo
        x = global_mean_pool(x, batch)
        
        # Classificazione finale dell'intero grafo
        out = self.lin(x)
        return out
    
class RobustGraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Strati convolutivi per estrarre le feature locali
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Guardrail contro l'over-smoothing: una proiezione lineare diretta dei nodi
        self.lin_nodes = torch.nn.Linear(in_channels, hidden_channels)
        
        # Il classificatore finale lavorerà sulla concatenazione di Mean e Max Pooling
        # Quindi l'input dell'MLP sarà hidden_channels * 2
        self.mlp1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.mlp2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. Via Convolutiva
        x_conv = F.relu(self.conv1(x, edge_index))
        x_conv = F.dropout(x_conv, p=0.3, training=self.training)
        x_conv = F.relu(self.conv2(x_conv, edge_index))
        
        # 2. Via Lineare Diretta (Preserva l'identità delle feature prima che la GCN le smoothing-izzi)
        x_lin = F.relu(self.lin_nodes(x))
        
        # Fondiamo i due contributi dei nodi
        x_combined = x_conv + x_lin
        
        # 3. POOLING IBRIDO: Concateniamo la media e i picchi massimi
        mean_p = global_mean_pool(x_combined, batch)
        max_p = global_max_pool(x_combined, batch)
        x_pool = torch.cat([mean_p, max_p], dim=-1) # Spazio latente completo del grafo
        
        # 4. Classificatore profondo per stabilizzare le 40 classi
        out = F.relu(self.mlp1(x_pool))
        out = F.dropout(out, p=0.3, training=self.training)
        return self.mlp2(out)