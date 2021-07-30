from argparse import Namespace, ArgumentParser
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
from pytorch_lightning import LightningModule
import torchmetrics


def create_model() -> nn.Module:
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("LitResnet")
        parser.add_argument("--lr", type=float, default=0.05)
        return parent_parser

    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.model = create_model()

        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, stage=None):
        x, y = batch
        logits = F.log_softmax(self.model(x), dim=1)
        loss = F.nll_loss(logits, y)
        self.log("Loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("Loss/val", loss)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)

    def validation_epoch_end(self, outputs):
        self.log("Accuracy/val", self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("Test/loss", loss)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)

    def test_epoch_end(self, outputs):
        self.log("Test/Accuracy", self.test_acc.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4
        )
        steps_per_epoch = 45000 // self.hparams.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
