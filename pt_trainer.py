import os
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from irregular_sampled_datasets import PersonData
from torch_node_cell import ODELSTM, IrregularSequenceLearner
from torchmetrics.functional import accuracy

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='person')
    parser.add_argument('--solver', type=str, default='dopri5')
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gpus', type=int, default=0)
    args = parser.parse_args()

    # Set up dataset
    if args.dataset == 'person':
        dataset = PersonData(seq_len=32)
        train_x = torch.from_numpy(dataset.train_x).float()
        train_t = torch.from_numpy(dataset.train_t).float()
        train_y = torch.from_numpy(dataset.train_y).long()
        
        train_dataset = torch.utils.data.TensorDataset(train_x, train_t, train_y)
        trainloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,  # Required for Windows stability
            persistent_workers=False
        )

    # Model setup
    model = ODELSTM(
        in_features=train_x.shape[-1],
        hidden_size=args.size,
        out_feature=len(dataset.class_map),
        solver_type=args.solver
    )
    learn = IrregularSequenceLearner(model)

    # Trainer configuration
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='odelstm-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1 if args.gpus > 0 else 'auto',
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        logger=True,
        log_every_n_steps=20
    )

    # Start training
    trainer.fit(learn, trainloader)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()