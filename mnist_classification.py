import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


class LitMNIST(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4):
        super().__init__()

        # Set our init args as class attributes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        channels, width, height = (1, 28, 28)

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        return x

    def evaluate(self, batch, mode=None):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if mode:
            self.log(f"{mode}_loss", loss, prog_bar=True)
            self.log(f"{mode}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, mode='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pytorch Lightning Practice with MNIST')
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_data = MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = MNIST(args.data_dir, train=False, download=True, transform=transform)

    train_data, val_data = train_test_split(train_data, random_state=args.seed, test_size=args.val_ratio, shuffle=True)

    train_loader = DataLoader(train_data, batch_size=args.batch)
    val_loader = DataLoader(val_data, batch_size=args.batch)
    test_loader = DataLoader(test_data, batch_size=args.batch)

    # train
    model = LitMNIST(learning_rate=args.lr)
    trainer = pl.Trainer(
        accelerator='cpu',
        max_epochs=args.epoch,
        progress_bar_refresh_rate=100,
    )
    trainer.fit(model, train_loader, val_loader)

    # test
    trainer.test(dataloaders=test_loader, ckpt_path='best')
