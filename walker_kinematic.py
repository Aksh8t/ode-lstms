import os
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from irregular_sampled_datasets import Walker2dImitationData
from torch_node_cell import ODELSTM  # Assume custom cells are implemented

class WalkerKinematicModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = ODELSTM(
            in_features=input_size,
            hidden_size=hidden_size,
            out_feature=hidden_size,
            solver_type='fixed_rk4'
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, t):
        outputs, _ = self.rnn(x, t)
        return self.fc(outputs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="odelstm")
    parser.add_argument("--size", default=64, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    args = parser.parse_args()

    # Load dataset
    data = Walker2dImitationData(seq_len=64)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(data.train_x),
        torch.FloatTensor(data.train_times),
        torch.FloatTensor(data.train_y)
    )
    
    valid_dataset = TensorDataset(
        torch.FloatTensor(data.valid_x),
        torch.FloatTensor(data.valid_times),
        torch.FloatTensor(data.valid_y)
    )
    
    test_dataset = TensorDataset(
        torch.FloatTensor(data.test_x),
        torch.FloatTensor(data.test_times),
        torch.FloatTensor(data.test_y)
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize model
    model = WalkerKinematicModel(
        input_size=data.input_size,
        hidden_size=args.size
    )
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    for epoch in range(args.epochs):
        # Training
        model.train()
        for x, t, y in train_loader:
            pred = model(x, t)
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, t, y in valid_loader:
                pred = model(x, t)
                val_loss += criterion(pred, y).item()
        
        val_loss /= len(valid_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    # Testing
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, t, y in test_loader:
            pred = model(x, t)
            test_loss += criterion(pred, y).item()
    
    test_loss /= len(test_loader)
    print(f"Best test loss: {test_loss:.3f}")

    # Save results
    base_path = "results/walker"
    os.makedirs(base_path, exist_ok=True)
    with open(f"{base_path}/{args.model}_{args.size}.csv", "a") as f:
        f.write(f"{test_loss:.6f}\n")

if __name__ == "__main__":
    main()