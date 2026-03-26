import json
import random
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from argparse import ArgumentParser

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset

from models import CNNEncoder, DNNEncoder, ResnetEncoder, LinearClassifier, NonlinearClassifier
from data import load_imbalanced_dataset, load_balanced_dataset
from losses import SupConLoss

# -----------------------------------------------------
# CONSTANTS
# -----------------------------------------------------
EPOCHS = 200
CLF_EPOCHS = 1000
N_CLASSES = { 'mnist': 10, 'fashion_mnist': 10, 'cifar10': 10, 'cifar100': 100 }
DATASET_TO_INDIM = { 'mnist': 784, 'fashion_mnist': 784, 'cifar10': 3072, 'cifar100': 3072 }
DATASET_TO_SHAPE = { 'mnist': (1, 28, 28), 'fashion_mnist': (1, 28, 28), 'cifar10': (3, 32, 32), 'cifar100': (3, 32, 32) }
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
    temperature: float = 0.07
    batch_size: int = 64
    m_incomplete: int = 5000
    test_size: int = 10000
    patience: int = 20
    dataset: str = 'mnist'
    model: str = 'cnn'

# -----------------------------------------------------
# Evaluate classifier on rare classes
# -----------------------------------------------------
def evaluate_classifier_rare_classes(classifier, encoder, X_test, labels_test, rarest_classes, config, device):
    print("\n" + "="*60)
    print("EVALUATING CLASSIFIER ON RARE CLASSES")
    print("="*60)

    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        batch_size = 256
        test_reps = []
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size].to(device)
            reps = encoder(batch)
            test_reps.append(reps.cpu())
        test_reps = torch.cat(test_reps, dim=0).to(device)

        all_logits = []
        for i in range(0, len(test_reps), batch_size):
            batch = test_reps[i:i+batch_size]
            logits = classifier(batch)
            all_logits.append(logits.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        predictions = all_logits.argmax(dim=1).cpu().numpy()

    labels_np = labels_test.cpu().numpy()
    overall_acc = (predictions == labels_np).mean() * 100

    f1, recall, precision, accuracy, support = {}, {}, {}, {}, {}
    for c in range(config.n_classes):
        tp = np.sum((predictions == c) & (labels_np == c))
        fp = np.sum((predictions == c) & (labels_np != c))
        fn = np.sum((predictions != c) & (labels_np == c))
        support[c] = np.sum(labels_np == c)
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * precision[c] * recall[c]) / (precision[c] + recall[c]) \
                if (precision[c] + recall[c]) > 0 else 0.0

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

    avg_precision = np.mean([rare_metrics[c]['precision'] for c in rarest_classes])
    avg_recall    = np.mean([rare_metrics[c]['recall']    for c in rarest_classes])
    avg_f1        = np.mean([rare_metrics[c]['f1']        for c in rarest_classes])
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
# Training loop  (SupConLoss, plain image/label loader)
# -----------------------------------------------------
def train_contrastive_model(X_train, labels_train, X_val, labels_val, X_test, labels_test,
                            config, n_epochs=100):
    set_seed(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, labels_train = X_train.to(device), labels_train.to(device)
    X_test,  labels_test  = X_test.to(device),  labels_test.to(device)
    X_val,   labels_val   = X_val.to(device),   labels_val.to(device)

    # ---- Encoder ----
    in_channels = DATASET_TO_SHAPE[config.dataset][0]
    in_dims     = DATASET_TO_INDIM[config.dataset]
    if config.model == 'resnet':
        encoder = ResnetEncoder(in_channels=in_channels, hidden_dim=256, output_dim=128).to(device)
    elif config.model == 'dnn':
        encoder = DNNEncoder(in_dims, hidden_dim=256, output_dim=128).to(device)
    else:
        encoder = CNNEncoder(in_channels=in_channels, hidden_dim=256, output_dim=128).to(device)
    print(f"\nUsing {config.model.upper()} encoder with {in_channels} input channels")

    # ---- Loss & optimiser ----
    criterion = SupConLoss(temperature=config.temperature, base_temperature=config.temperature)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3, amsgrad=True)

    # ---- Plain dataloader: (image, label) batches ----
    train_dataset = TensorDataset(X_train, labels_train)
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,   # SupConLoss needs ≥2 samples per class in a batch
    )

    # ---- Identify 5 rarest classes ----
    class_counts  = np.bincount(labels_train.cpu().numpy())
    rarest_classes = np.argsort(class_counts)[:5].tolist()

    print(f"\n{'='*60}")
    print(f"Training with SupConLoss")
    print(f"N={len(X_train)} labeled samples")
    print(f"Batch size={config.batch_size}")
    print(f"Rarest classes: {rarest_classes} with counts {class_counts[rarest_classes]}")
    print(f"{'='*60}")

    loss_history = []
    best_acc, best_model, epoch_no_improve = 0.0, encoder.state_dict(), 0

    for epoch in range(n_epochs):
        encoder.train()
        epoch_loss, num_batches = 0.0, 0

        for images, batch_labels in dataloader:
            images       = images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            # Encode → (bsz, d), then unsqueeze to (bsz, 1, d) for SupConLoss
            z = encoder(images)                        # (bsz, d)
            z = z.unsqueeze(1)                         # (bsz, 1, d)

            loss = criterion(z, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)

        # ---- Validation: KNN on train reps, predict val ----
        encoder.eval()
        with torch.no_grad():
            batch_size = 256
            train_reps = torch.cat(
                [encoder(X_train[i:i+batch_size]) for i in range(0, len(X_train), batch_size)]
            ).cpu()
            val_reps = torch.cat(
                [encoder(X_val[i:i+batch_size]) for i in range(0, len(X_val), batch_size)]
            ).cpu()

        knn_model = KNeighborsClassifier(n_neighbors=config.k_negatives).fit(
            train_reps.numpy(), labels_train.cpu().numpy()
        )
        pred_val = knn_model.predict(val_reps.numpy())
        acc_knn  = accuracy_score(labels_val.cpu().numpy(), pred_val)

        if acc_knn >= best_acc:
            epoch_no_improve = 0
            best_model, best_acc = encoder.state_dict(), acc_knn
            print(f' - Update model at epoch {epoch+1}, '
                  f'best KNN acc = {acc_knn:.5f}, train loss = {avg_epoch_loss:.5f}')
        else:
            epoch_no_improve += 1

        if epoch_no_improve >= config.patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_epoch_loss:.4f} | KNN Val Acc: {acc_knn:.4f}")

    encoder.load_state_dict(best_model)
    return encoder, loss_history, rarest_classes


# -----------------------------------------------------
# Linear / nonlinear classifier on frozen encoder
# -----------------------------------------------------
def train_linear_classifier(encoder, X_train, labels_train, X_val, labels_val,
                             config, device, n_epochs=100):
    print("\n" + "="*60)
    print("TRAINING LINEAR CLASSIFIER")
    print("="*60)

    encoder.eval()

    with torch.no_grad():
        batch_size = 256
        train_reps = torch.cat(
            [encoder(X_train[i:i+batch_size].to(device)) for i in range(0, len(X_train), batch_size)]
        ).to(device)
        val_reps = torch.cat(
            [encoder(X_val[i:i+batch_size].to(device)) for i in range(0, len(X_val), batch_size)]
        ).to(device)

    embedding_dim = train_reps.shape[1]
    if config.dataset == 'cifar100':
        classifier = NonlinearClassifier(embedding_dim, config.n_classes).to(device)
    else:
        classifier = LinearClassifier(embedding_dim, config.n_classes).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4, amsgrad=True)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(train_reps, labels_train.to(device)),
        batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_reps, labels_val.to(device)),
        batch_size=config.batch_size, shuffle=False
    )

    print(f"Training for up to {n_epochs} epochs...")
    print(f"Train representations: {train_reps.shape}")
    print(f"Validation representations: {val_reps.shape}")

    best_acc, epoch_no_improve = 0.0, 0
    CLF_PATIENCE = 50

    for epoch in range(n_epochs):
        classifier.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for batch_reps, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = classifier(batch_reps)
            loss   = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = logits.max(1)
            total   += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        train_acc = 100. * correct / total

        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_reps, batch_labels in val_loader:
                logits = classifier(batch_reps)
                _, predicted = logits.max(1)
                total   += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
        val_acc = 100. * correct / total

        if val_acc >= best_acc:
            best_acc = val_acc
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1

        if epoch_no_improve >= CLF_PATIENCE:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss/len(train_loader):.4f} "
                  f"| Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    return classifier


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*60)
    print(f"{config.dataset.upper()} Dataset")
    print(f"Device: {device}")
    print("="*60)

    print("\n" + "-"*60)
    print(f"LOADING {config.dataset.upper()} DATASET")
    print("-"*60)
    if config.dataset != 'cifar100':
        X_train_img, labels_train, X_test_img, labels_test, X_val_img, labels_val, class_sizes = \
            load_imbalanced_dataset(config)
    else:
        X_train_img, labels_train, X_test_img, labels_test, X_val_img, labels_val, class_sizes = \
            load_balanced_dataset(config)

    # ---- Train contrastive encoder with SupConLoss ----
    encoder, loss_history, rarest_classes = train_contrastive_model(
        X_train_img, labels_train, X_val_img, labels_val, X_test_img, labels_test,
        config, n_epochs=EPOCHS
    )

    # ---- Train downstream classifier ----
    print("\n--- Training linear classifier on encoder representations ---")
    classifier = train_linear_classifier(
        encoder, X_train_img, labels_train,
        X_val_img, labels_val, config, device, n_epochs=CLF_EPOCHS
    )

    # ---- Evaluate ----
    clf_result = evaluate_classifier_rare_classes(
        classifier, encoder, X_test_img, labels_test,
        rarest_classes, config, device
    )

    # ---- Save results ----
    output_filename = (
        f"results/clf_result_supcon_{config.dataset}"
        f"_rhomax{config.rho_max}_{config.model}.json"
    )
    with open(output_filename, "w") as f:
        json.dump({'supcon': clf_result}, f)
    print(f"\nResults saved to: {output_filename}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset',  type=str,   default='cifar10',
                        choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'])
    parser.add_argument('--patience', type=int,   default=30)
    parser.add_argument('--model',    type=str,   default='cnn',
                        choices=['cnn', 'dnn', 'resnet'])
    parser.add_argument('--rho_max',  type=float, default=0.5)
    parser.add_argument('--k',        type=int,   default=5)
    parser.add_argument('--M',        type=int,   default=20000)
    parser.add_argument('--N',        type=int,   default=20000)
    args = vars(parser.parse_args())

    config = ContrastiveConfig(
        model=args['model'],
        k_negatives=args['k'],
        dataset=args['dataset'],
        n_samples=args['N'],
        m_incomplete=args['M'],
        rho_max=args['rho_max'],
        patience=args['patience'],
        n_classes=N_CLASSES[args['dataset']]
    )
    main(config)
