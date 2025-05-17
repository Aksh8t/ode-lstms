import os
import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from irregular_sampled_datasets import PersonData, ETSMnistData, XORData
from torch_node_cell import ODELSTM, IrregularSequenceLearner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="person")
    parser.add_argument("--solver", default="dopri5")
    parser.add_argument("--size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--gpus", default=0, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    args = parser.parse_args()

    def load_dataset():
        if args.dataset == "person":
            # Person Activity Dataset
            dataset = PersonData()
            train_x = torch.Tensor(dataset.train_x)
            train_y = torch.LongTensor(dataset.train_y)
            train_t = torch.Tensor(dataset.train_t)
            test_x = torch.Tensor(dataset.test_x)
            test_y = torch.LongTensor(dataset.test_y)
            test_t = torch.Tensor(dataset.test_t)
            
            train_dataset = TensorDataset(train_x, train_t, train_y)
            test_dataset = TensorDataset(test_x, test_t, test_y)
            
            in_features = train_x.size(-1)
            num_classes = int(torch.max(train_y).item()) + 1  # Fixed parenthesis
            return_sequences = True
            
        elif args.dataset == "et_mnist":
            # Event-based MNIST Dataset
            dataset = ETSMnistData(time_major=False)
            train_x = torch.Tensor(dataset.train_events)
            train_y = torch.LongTensor(dataset.train_y)
            train_t = torch.Tensor(dataset.train_elapsed)
            test_x = torch.Tensor(dataset.test_events)
            test_y = torch.LongTensor(dataset.test_y)
            test_t = torch.Tensor(dataset.test_elapsed)
            
            in_features = train_x.size(-1)
            num_classes = int(torch.max(train_y).item()) + 1  # Fixed parenthesis
            return_sequences = False
            
        elif args.dataset == "xor":
            # XOR Dataset
            dataset = XORData(time_major=False, event_based=True, pad_size=32)
            train_x = torch.Tensor(dataset.train_events)
            train_y = torch.LongTensor(dataset.train_y)
            train_t = torch.Tensor(dataset.train_elapsed)
            test_x = torch.Tensor(dataset.test_events)
            test_y = torch.LongTensor(dataset.test_y)
            test_t = torch.Tensor(dataset.test_elapsed)
            
            in_features = train_x.size(-1)
            num_classes = int(torch.max(train_y).item()) + 1  # Fixed parenthesis
            return_sequences = False
            
        else:
            raise ValueError(f"Unknown dataset '{args.dataset}'")

        # Create DataLoaders for all datasets
        trainloader = DataLoader(
            TensorDataset(train_x, train_t, train_y),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        testloader = DataLoader(
            TensorDataset(test_x, test_t, test_y),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        return trainloader, testloader, in_features, num_classes, return_sequences

    trainloader, testloader, in_features, num_classes, return_sequences = load_dataset()

    model = ODELSTM(
        in_features=in_features,
        hidden_size=args.size,
        out_feature=num_classes,
        solver_type=args.solver,
        return_sequences=return_sequences
    )
    learn = IrregularSequenceLearner(model, lr=args.lr)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="odelstm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    trainer = pl.Trainer(
    accelerator="gpu" if args.gpus > 0 else "cpu",
    devices=args.gpus if args.gpus > 0 else 1,
    max_epochs=args.epochs,
    callbacks=[checkpoint_callback],
    gradient_clip_val=1,
    enable_progress_bar=True,
    logger=True,
    log_every_n_steps=20
)


    trainer.fit(learn, trainloader)
    results = trainer.test(learn, testloader)

    base_path = f"results/{args.dataset}"
    os.makedirs(base_path, exist_ok=True)
    with open(f"{base_path}/pt_ode_lstm_{args.size}.csv", "a") as f:
        f.write(f"{results[0]['test_acc']:.6f}\n")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()

