from collections import Counter
from torch_geometric.datasets import Reddit
from torch_geometric.loader import ClusterData
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool,SAGEConv
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

        raw_labels = [subgraph.y.item() for subgraph in graphs]
    
        #conteggio quante volte compare ogni label
        class_count = Counter(raw_labels)
        
        if self.task_type == "a":
            #per il task a non ci sono problemi di stratify perchè le 2 classi hanno tanti campioni, quindi non rimuoviamo nulla
            graphs_filtrati = graphs
            labels_filtrate = raw_labels
        elif self.task_type == "b":
            #teniamo solo i grafi la cui classe compare almeno 10 volte per evitare errori nello stratify
            graphs_filtrati = []
            labels_filtrate = []
            
            for subgraph, label in zip(graphs, raw_labels):
                if class_count[label] >= 10:
                    graphs_filtrati.append(subgraph)
                    labels_filtrate.append(label)
                
            print(f"Rimossi {len(graphs) - len(graphs_filtrati)} grafi impredicibili.")
            print(f"Grafi totali rimasti per il Task B: {len(graphs_filtrati)}")

        #split stratificato in train, val e test (80% train, 10% val, 10% test)
        train_graphs, temp_graphs, _, y_temp = train_test_split(
            graphs_filtrati, 
            labels_filtrate, 
            test_size=0.20, 
            stratify=labels_filtrate,
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

        #calcolo della densità interna di ogni singolo sotto-grafo
        for i in range(len(cluster_data)):
            subgraph = cluster_data[i]
            v = subgraph.num_nodes
            e = subgraph.num_edges

            if v > 1:
                #formula della densità topologica per grafi, omettiamo il fattore 2 perché in PyG gli edge sono diretti 
                # e quindi ogni edge è contato due volte
                d = e / (v * (v - 1))
            else:
                d = 0.0

            densita_clusters.append(d)

        #calcolo della densità media globale
        media_globale = np.mean(densita_clusters)
        print(f"Densità interna media dei sotto-grafi: {media_globale:.6f}\n")

        #assegnazione delle label
        grafi_finali = []
        conteggio_classi = {0: 0, 1: 0}

        for i in range(len(cluster_data)):
            subgraph = cluster_data[i]
            densita_corrente = densita_clusters[i]

            # target = 1 se sopra la media, 0 se sotto la media
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
        
        #iteriamo su ogni sotto-grafo
        for i in range(len(cluster_data)):
            subgraph = cluster_data[i] 

            # Calcolo del target sulla base della community di maggioranza, bincount conta le frequenze e max prende il massimo
            values, _ = torch.bincount(subgraph.y).max(dim=0)

            subgraph.y = values  # ora y è un singolo scalare per l'intero sotto-grafo

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
            loss = loss_fn(out, batch.y.float().unsqueeze(1))
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        # Accumuliamo la loss pesata per il numero di grafi nel batch
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

            loss = loss_fn(out, batch.y.float().unsqueeze(1))
            
            preds = torch.sigmoid(out)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())

            val_loss += loss.item() * batch.num_graphs
            
    # Concateniamo i vettori di tutti i batch
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    y_pred_binary = (all_preds >= 0.5).astype(int)
    acc = balanced_accuracy_score(all_targets, y_pred_binary)
    
    # 'weighted' calcola l'F1 per ogni classe e ne fa la media pesata sul numero di campioni
    f1_bilanciato = f1_score(all_targets, y_pred_binary, average='weighted')
    
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



class HierarchicalGCNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        
        # BLOCCO 1
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # tiene solo il 50% dei nodi (ratio=0.5) basandosi su un punteggio di rilevanza appreso
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5) 
        
        # BLOCCO 2
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)
        
        # BLOCCO 3 
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Classificatore finale
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        #il pooling gerarchico taglia i nodi meno importanti sulla base dei punteggi di rilevanza appresi.
        #restituisce il nuovo x, il nuovo edge_index ristretto e il batch aggiornato
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #solo ora applichiamo il Global Mean Pool per condesare le informazioni di tutti i nodi rimasti 
        # in un singolo vettore per grafo
        x = global_mean_pool(x, batch)
        
        # Classificazione finale dell'intero grafo
        out = self.lin(x)
        return out
    

class HierarchicalSAGEConvClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        
        # BLOCCO 1
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        # tiene solo il 50% dei nodi (ratio=0.5) basandosi su un punteggio di rilevanza appreso
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5) 
        
        # BLOCCO 2
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)
        
        # BLOCCO 3 
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        
        # Classificatore finale
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        #il pooling gerarchico taglia i nodi meno importanti sulla base dei punteggi di rilevanza appresi.
        #restituisce il nuovo x, il nuovo edge_index ristretto e il batch aggiornato
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #solo ora applichiamo il Global Mean Pool per condesare le informazioni di tutti i nodi rimasti 
        # in un singolo vettore per grafo
        x = global_mean_pool(x, batch)
        
        # Classificazione finale dell'intero grafo
        out = self.lin(x)
        return out


class HierarchicalGATClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        
        # BLOCCO 1
        self.conv1 = GATConv(in_channels, hidden_channels)
        # tiene solo il 50% dei nodi (ratio=0.5) basandosi su un punteggio di rilevanza appreso
        self.pool1 = TopKPooling(hidden_channels, ratio=0.5) 
        
        # BLOCCO 2
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.pool2 = TopKPooling(hidden_channels, ratio=0.5)
        
        # BLOCCO 3 
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        
        # Classificatore finale
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        #il pooling gerarchico taglia i nodi meno importanti sulla base dei punteggi di rilevanza appresi.
        #restituisce il nuovo x, il nuovo edge_index ristretto e il batch aggiornato
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #solo ora applichiamo il Global Mean Pool per condesare le informazioni di tutti i nodi rimasti 
        # in un singolo vettore per grafo
        x = global_mean_pool(x, batch)
        
        # Classificazione finale dell'intero grafo
        out = self.lin(x)
        return out
    