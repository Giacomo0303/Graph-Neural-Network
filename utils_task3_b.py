from collections import Counter
from torch_geometric.datasets import Reddit
from torch_geometric.loader import ClusterData
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool,GCNConv,SAGEConv,GINConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from sklearn.metrics import  balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm


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



class GCNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        # BLOCCO 1
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # BLOCCO 2
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Classificatore finale
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
    
        # Solo ora applichiamo il Global Mean Pool per condesare le informazioni di tutti i nodi rimasti in un singolo vettore per grafo
        x = global_mean_pool(x, batch)
        
        # Classificazione finale dell'intero grafo
        x = self.lin(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
    

class SAGEClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        # BLOCCO 1
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        
        # BLOCCO 2
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Classificatore finale
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
    
        # Solo ora applichiamo il Global Mean Pool per condesare le informazioni di tutti i nodi rimasti in un singolo vettore per grafo
        x = global_mean_pool(x, batch)
        
        # Classificazione finale dell'intero grafo
        x = self.lin(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
    
    
class GATv2Classifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        # BLOCCO 1
        self.conv1 = GATv2Conv(in_channels, hidden_channels)
        
        # BLOCCO 2
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
        
        # Classificatore finale
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
    
        # Solo ora applichiamo il Global Mean Pool per condesare le informazioni di tutti i nodi rimasti 
        # in un singolo vettore per grafo
        x = global_mean_pool(x, batch)
        
        # Classificazione finale dell'intero grafo
        x = self.lin(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x
    
    

   
class HierarchicalGINClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        
        self.dropout = dropout

        # BLOCCO 1
        # GIN richiede un layer linear all'interno, nel paper originale viene usato un MLP a 2 layer con BatchNorm e ReLU
        mlp1 = Sequential(
            Linear(in_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(mlp1)
        self.bn1 = BatchNorm1d(hidden_channels)
        #pooling gerarchico
        #self.pool1 = TopKPooling(hidden_channels, ratio=0.5) 
        
        # BLOCCO 2
        mlp2 = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(mlp2)
        self.bn2 = BatchNorm1d(hidden_channels)
        #self.pool2 = TopKPooling(hidden_channels, ratio=0.5)
        
        # BLOCCO 3
        mlp3 = Sequential(
            Linear(hidden_channels, hidden_channels),
            BatchNorm1d(hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.conv3 = GINConv(mlp3)
        self.bn3 = BatchNorm1d(hidden_channels)
        
        #classificatore finale 
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # forward blocco 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # pooling 1
        #x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        # forward blocco 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # pooling 2
        #x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)
        
        # forward blocco 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # pooling globale
        x = global_mean_pool(x, batch)
        
        # classificazione finale 
        out = self.lin(x)
        return out