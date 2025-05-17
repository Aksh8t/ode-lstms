import os
import torch
import argparse
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from irregular_sampled_datasets import XORData
from torch_node_cell import ODELSTM  # Assume custom cells implemented

class XORModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = ODELSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            return_sequences=False
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, t, mask):
        # Apply mask to input
        x = x * mask.unsqueeze(-1).float()
        outputs, _ = self.rnn(x, t)
        return self.fc(outputs[:, -1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="odelstm")
    parser.add_argument("--size", default=64, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()

    # Load dataset
    data = XORData(time_major=False, event_based=not args.dense, pad_size=32)
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(data.train_events),
        torch.FloatTensor(data.train_elapsed),
        torch.BoolTensor(data.train_mask),
        torch.FloatTensor(data.train_y)
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(data.test_events),
        torch.FloatTensor(data.test_elapsed),
        torch.BoolTensor(data.test_mask),
        torch.FloatTensor(data.test_y)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    model = XORModel(input_size=1, hidden_size=args.size)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        for x, t, mask, y in train_loader:
            outputs = model(x, t, mask)
            loss = criterion(outputs.squeeze(), y.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, t, mask, y in test_loader:
                outputs = model(x, t, mask)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted.squeeze() == y).sum().item()
                total += y.size(0)
        
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc

    # Save results
    base_path = "results/xor_dense" if args.dense else "results/xor_event"
    os.makedirs(base_path, exist_ok=True)
    with open(f"{base_path}/{args.model}_{args.size}.csv", "a") as f:
        f.write(f"{best_acc:.6f}\n")

if __name__ == "__main__":
    main()