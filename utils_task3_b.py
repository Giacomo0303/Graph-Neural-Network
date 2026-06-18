import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool,global_max_pool
from sklearn.metrics import  balanced_accuracy_score, f1_score
from tqdm import tqdm


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