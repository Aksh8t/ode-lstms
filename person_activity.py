# Copyright 2021 The ODE-LSTM Authors. All Rights Reserved.
import os
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from irregular_sampled_datasets import PersonData
from torch_node_cell import ODELSTM  # Ensure this matches your implementation

# Suppress TensorFlow warnings (if needed)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class PersonActivityModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = ODELSTM(
            in_features=input_size,
            hidden_size=hidden_size,
            out_feature=hidden_size,
            solver_type='fixed_rk4'
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, t):
        out, _ = self.rnn(x, t)
        return self.fc(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="odelstm")
    parser.add_argument("--size", default=64, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    args = parser.parse_args()

    # Load dataset
    data = PersonData()
    train_dataset = TensorDataset(
        torch.FloatTensor(data.train_x),
        torch.FloatTensor(data.train_t),
        torch.LongTensor(data.train_y)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(data.test_x),
        torch.FloatTensor(data.test_t),
        torch.LongTensor(data.test_y)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Model setup
    model = PersonActivityModel(
        input_size=data.feature_size,
        hidden_size=args.size,
        num_classes=data.num_classes
    )
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        for x, t, y in train_loader:
            outputs = model(x, t)
            loss = criterion(outputs.view(-1, data.num_classes), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, t, y in test_loader:
                outputs = model(x, t)
                _, predicted = torch.max(outputs.data, 2)
                total += y.nelement()
                correct += (predicted == y).sum().item()
        
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc

    # Save results
    base_path = "results/person_activity"
    os.makedirs(base_path, exist_ok=True)
    with open(f"{base_path}/{args.model}_{args.size}.csv", "a") as f:
        f.write(f"{best_acc:.6f}\n")

if __name__ == "__main__":
    main()