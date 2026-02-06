import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from dataclasses import dataclass
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from models import CNNEncoder, DNNEncoder, LinearClassifier
from data import load_imbalanced_dataset, collate_tuples, ContrastiveTupleDataset

# -----------------------------------------------------
# CONSTANTS 
# -----------------------------------------------------
EPOCHS = 300
CLF_EPOCHS = 200
DATASET_TO_INDIM = { 'mnist': 784, 'fashion_mnist': 784, 'cifar10': 3072 }
DATASET_TO_SHAPE = { 'mnist': (1, 28, 28), 'fashion_mnist': (1, 28, 28), 'cifar10': (3, 32, 32) }
DATASET_MAP = { 'mnist': datasets.MNIST, 'fashion_mnist': datasets.FashionMNIST, 'cifar10': datasets.CIFAR10 }

# Distrust random initialization
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------
# Configuration
# -----------------------------------------------------
@dataclass
class ContrastiveConfig:
    n_samples: int = 10000  
    n_classes: int = 10
    k_negatives: int = 5
    rho_max: float = 0.45
    temperature: float = 0.5
    batch_size: int = 64 
    m_incomplete: int = 5000 
    test_size: int = 10000 
    dataset: str = 'mnist'
    model: str = 'cnn'

# -----------------------------------------------------
# Batched contrastive loss
# -----------------------------------------------------
def batched_contrastive_loss(anchors, positives, negatives, temperature):
    batch_size = anchors.shape[0]
    pos_sim = (anchors * positives).sum(dim=1) / temperature
    neg_sims = torch.bmm(negatives, anchors.unsqueeze(2)).squeeze(2) / temperature
    v = pos_sim.unsqueeze(1) - neg_sims
    losses = torch.log1p(torch.exp(-v).sum(dim=1))
    return losses

# -----------------------------------------------------
# Single tuple loss
# -----------------------------------------------------
def contrastive_loss(anchor, positive, negatives, temperature):
    pos_sim = (anchor * positive).sum() / temperature
    neg_sims = (negatives @ anchor) / temperature
    v = pos_sim - neg_sims
    loss = torch.log1p(torch.exp(-v).sum())
    return loss

# -----------------------------------------------------
# Evaluate on specific classes
# -----------------------------------------------------
def evaluate_on_classes(test_dataset, encoder, config, device, target_classes):
    encoder.eval()
    X = test_dataset.X
    
    losses_per_class = {c: [] for c in target_classes}
    n_samples_per_class = 50
    
    with torch.no_grad():
        for c in target_classes:
            for _ in range(n_samples_per_class):
                t = test_dataset.sample_tuple(c)
                if t is None:
                    continue
                anchor_idx, pos_idx, neg_indices = t
                
                z_anchor = encoder(X[anchor_idx:anchor_idx+1])[0]
                z_pos = encoder(X[pos_idx:pos_idx+1])[0]
                z_negs = encoder(X[neg_indices])
                
                loss = contrastive_loss(z_anchor, z_pos, z_negs, config.temperature)
                losses_per_class[c].append(float(loss))
    
    avg_losses = {}
    for c in target_classes:
        if len(losses_per_class[c]) > 0:
            avg_losses[c] = np.mean(losses_per_class[c])
        else:
            avg_losses[c] = float('nan')
    
    encoder.train()
    return avg_losses

# -----------------------------------------------------
# Training loop
# -----------------------------------------------------
def train_contrastive_model(X_train, labels_train, X_test, labels_test, config, n_epochs=100, use_weighting=True, avoid_collision=False):
    set_seed(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, labels_train = X_train.to(device), labels_train.to(device)
    X_test, labels_test = X_test.to(device), labels_test.to(device)
    
    # Create encoder based on config
    in_channels = DATASET_TO_SHAPE[config.dataset][0]
    in_dims = DATASET_TO_INDIM[config.dataset]
    encoder = CNNEncoder(in_channels=in_channels, hidden_dim=128, output_dim=64).to(device)
    if config.model == 'dnn':
        encoder = DNNEncoder(in_dims, hidden_dim=128, output_dim=64).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, amsgrad=True)
    print(f"\nUsing CNN encoder with {in_channels} input channels")
    
    loss_history = []
    test_loss_history = []
    
    # Identify 5 rarest classes
    class_counts = np.bincount(labels_train.cpu().numpy())
    rarest_classes = np.argsort(class_counts)[:5].tolist()
    
    method_name = "WEIGHTED" if use_weighting else "UNWEIGHTED"
    print(f"\n{'='*60}")
    print(f"Training with {method_name} incomplete U-statistics")
    print(f"N={len(X_train)} labeled samples")
    print(f"M={config.m_incomplete} tuples per epoch")
    print(f"k={config.k_negatives} negatives per tuple")
    print(f"Batch size={config.batch_size}")
    print(f"Rarest classes: {rarest_classes} with counts {class_counts[rarest_classes]}")
    print(f"{'='*60}")

    dataset = ContrastiveTupleDataset(
        X_train, labels_train, 
        config.k_negatives, 
        config.m_incomplete,
        use_weighting=use_weighting,
        avoid_collision=avoid_collision
    )

    test_dataset = ContrastiveTupleDataset(
        X_test, labels_test, 
        config.k_negatives, 
        config.m_incomplete,
        use_weighting=False,
        avoid_collision=True
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_tuples,
        num_workers=0
    )
    
    best_model, best_loss = None, np.inf
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for anchors, positives, negatives, weights in dataloader:
            optimizer.zero_grad()
            
            weights = weights.to(device)
            z_anchors = encoder(anchors)
            z_positives = encoder(positives)
            
            batch_size, k, C, H, W = negatives.shape
            z_negatives = encoder(negatives.view(batch_size * k, C, H, W))
            z_negatives = z_negatives.view(batch_size, k, -1)
            
            losses = batched_contrastive_loss(
                z_anchors, z_positives, z_negatives, 
                config.temperature
            )
            
            weighted_losses = losses * weights
            batch_loss = weighted_losses.sum() / weights.sum()
            
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
        
        # Compute train loss
        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
            
        # Update model based on train loss 
        if avg_epoch_loss < best_loss:
            best_model = encoder.state_dict()
            best_loss = avg_epoch_loss
            print(f' - Update model at epoch {epoch}, new best = {best_loss:.5f}')
        
        if (epoch + 1) % 20 == 0:
            # Compute rare class loss
            test_losses = evaluate_on_classes(test_dataset, encoder, config, device, rarest_classes)
            avg_test_loss = np.mean([v for v in test_losses.values() if not np.isnan(v)])
            test_loss_history.append(avg_test_loss)
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_epoch_loss:.4f} | Test Loss (rare): {avg_test_loss:.4f}")
    
    # Load best model
    encoder.load_state_dict(best_model)
    final_test_losses = evaluate_on_classes(test_dataset, encoder, config, device, rarest_classes)
    return encoder, loss_history, test_loss_history, final_test_losses, rarest_classes

def train_linear_classifier(encoder, X_train, labels_train, X_test, labels_test, config, device, n_epochs=100):
    print("\n" + "="*60)
    print("TRAINING LINEAR CLASSIFIER")
    print("="*60)
    
    encoder.eval()
    
    # Extract representations
    with torch.no_grad():
        batch_size = 256
        train_reps = []
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size].to(device)
            reps = encoder(batch)
            train_reps.append(reps.cpu())
        train_reps = torch.cat(train_reps, dim=0).to(device)
        
        test_reps = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            reps = encoder(batch)
            test_reps.append(reps.cpu())
        test_reps = torch.cat(test_reps, dim=0).to(device)
    
    embedding_dim = train_reps.shape[1]
    classifier = LinearClassifier(embedding_dim, config.n_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, amsgrad=True)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataloader
    train_dataset = torch.utils.data.TensorDataset(train_reps, labels_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    print(f"Training for {n_epochs} epochs...")
    print(f"Train representations: {train_reps.shape}")
    print(f"Test representations: {test_reps.shape}")
    
    for epoch in range(n_epochs):
        classifier.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_reps, batch_labels in train_loader:
            optimizer.zero_grad()
            batch_labels = batch_labels.to(device)
            
            logits = classifier(batch_reps)
            loss = criterion(logits, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        train_acc = 100. * correct / total
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
    
    return classifier

def evaluate_classifier_rare_classes(classifier, encoder, X_test, labels_test, rarest_classes, config, device):
    print("\n" + "="*60)
    print("EVALUATING CLASSIFIER ON RARE CLASSES")
    print("="*60)
    
    encoder.eval()
    classifier.eval()
    
    # Extract test representations
    with torch.no_grad():
        batch_size = 256
        test_reps = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            reps = encoder(batch)
            test_reps.append(reps.cpu())
        test_reps = torch.cat(test_reps, dim=0).to(device)
        
        # Get predictions
        all_logits = []
        for i in range(0, len(test_reps), batch_size):
            batch = test_reps[i:i+batch_size]
            logits = classifier(batch)
            all_logits.append(logits.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        predictions = all_logits.argmax(dim=1).cpu().numpy()
    labels_np = labels_test.cpu().numpy()
    
    # Calculate overall metrics
    overall_acc = (predictions == labels_np).mean() * 100
    
    # Compute precision and recall manually for each class
    f1, recall, precision, accuracy, support = {}, {}, {}, {}, {}
    
    for c in range(config.n_classes):
        tp = np.sum((predictions == c) & (labels_np == c))
        fp = np.sum((predictions == c) & (labels_np != c))
        fn = np.sum((predictions != c) & (labels_np == c))
        support[c] = np.sum(labels_np == c)
        
        # Precision: TP / (TP + FP)
        if tp + fp > 0:
            precision[c] = tp / (tp + fp)
        else:
            precision[c] = 0.0
        
        # Recall: TP / (TP + FN)
        if tp + fn > 0:
            recall[c] = tp / (tp + fn)
        else:
            recall[c] = 0.0
        
        # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
        if precision[c] + recall[c] > 0:
            f1[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])
        else:
            f1[c] = 0.0
    
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")
    print("\n" + "-"*60)
    print("RARE CLASS METRICS")
    print("-"*60)
    print(f"{'Class':<8} {'Support':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    
    rare_metrics = {}
    for c in rarest_classes:
        rare_metrics[c] = {
            'precision': precision[c],
            'recall': recall[c],
            'f1': f1[c],
            'support': support[c]
        }
        print(f"{c:<8} {support[c]:<10} {precision[c]:<12.4f} {recall[c]:<12.4f} {f1[c]:<12.4f}")
    print("-"*60)
    
    # Calculate average metrics for rare classes
    avg_precision = np.mean([rare_metrics[c]['precision'] for c in rarest_classes])
    avg_recall = np.mean([rare_metrics[c]['recall'] for c in rarest_classes])
    avg_f1 = np.mean([rare_metrics[c]['f1'] for c in rarest_classes])
    results = {
        'overall_acc': overall_acc,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1
    }
    
    print(f"\nAverage across rare classes:")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  F1-Score:  {avg_f1:.4f}")
    print("="*60)
    
    return results

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main(config):
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Report device + dataset
    print("="*60)
    print(f"{config.dataset.upper()} Dataset")
    print(f"Device: {device}")
    print("="*60)
    
    # Load dataset 
    print("\n" + "-"*60)
    print(f"LOADING {config.dataset.upper()} DATASET")
    print("-"*60)
    X_train_img, labels_train, X_test_img, labels_test, class_sizes = load_imbalanced_dataset(config)

    # Train WEIGHTED
    print("\n" + "="*60)
    print("EXPERIMENT 1: WEIGHTED U-STATISTICS")
    print("="*60)
    results_weighted = train_contrastive_model(
        X_train_img, labels_train, X_test_img, labels_test,
        config, n_epochs=EPOCHS, use_weighting=True, avoid_collision=False
    )
    encoder_weighted, _, _, final_loss_weighted, rarest_classes = results_weighted

    # Train classifier weighted
    print("\n--- Training classifier on WEIGHTED encoder ---")
    classifier_weighted = train_linear_classifier(
        encoder_weighted, X_train_img, labels_train,
        X_test_img, labels_test, config, device, n_epochs=CLF_EPOCHS
    )
    clf_result_weighted = evaluate_classifier_rare_classes(
        classifier_weighted, encoder_weighted, X_test_img, labels_test,
        rarest_classes, config, device
    )
    
    # Train UNWEIGHTED
    print("\n" + "="*60)
    print("EXPERIMENT 2: UNWEIGHTED U-STATISTICS")
    print("="*60)
    results_unweighted = train_contrastive_model(
        X_train_img, labels_train, X_test_img, labels_test,
        config, n_epochs=EPOCHS, use_weighting=False, avoid_collision=True
    )
    encoder_unweighted, _, _, final_loss_unweighted, rarest_classes = results_unweighted

    # Train classifier unweighted
    print("\n--- Training classifier on UNWEIGHTED encoder ---")
    classifier_unweighted = train_linear_classifier(
        encoder_unweighted, X_train_img, labels_train,
        X_test_img, labels_test, config, device, n_epochs=CLF_EPOCHS
    )
    clf_result_unweighted = evaluate_classifier_rare_classes(
        classifier_unweighted, encoder_unweighted, X_test_img, labels_test,
        rarest_classes, config, device
    )

    # Record output 
    output_filename = f"results/clf_result_{config.dataset}_k{config.k_negatives}_rhomax{config.rho_max}_{config.model}.json"
    with open(output_filename, "w") as f:
        json.dump({
            'weighted': clf_result_weighted, 
            'unweighted': clf_result_unweighted,
            'final_contrastive_loss_weighted': final_loss_weighted,
            'final_contrastive_loss_unweighted': final_loss_unweighted
        }, f)
    print(f"\nResults saved to: {output_filename}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=False, default='cifar10', choices=['mnist', 'fashion_mnist', 'cifar10'], help='Real Dataset')
    parser.add_argument('--model', type=str, required=False, default='cnn', choices=['cnn', 'dnn'], help='Model architecture type')
    parser.add_argument('--rho_max', type=float, required=False, default=0.5, help='Probability of dominant class')
    parser.add_argument('--k', type=int, required=False, default=5, help='Number of negative samples')
    parser.add_argument('--M', type=int, required=False, default=20000, help='Number of sub-sampled tuples')
    parser.add_argument('--N', type=int, required=False, default=10000, help='Number of labeled instances')
    args = vars(parser.parse_args())

    # Run main
    config = ContrastiveConfig(
        model=args['model'],
        k_negatives=args['k'], 
        dataset=args['dataset'], 
        n_samples=args['N'],
        m_incomplete=args['M'], 
        rho_max=args['rho_max']
    )
    main(config)
