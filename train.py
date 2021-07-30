from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import loggers as pl_loggers

from datamodules import bases as datamodule_base
from models.bases import LitResnet


seed_everything(42)

parser = ArgumentParser()
parser = datamodule_base.add_argparse_args(parser)
parser = LitResnet.add_argparse_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

model = LitResnet(args)
datamodule = datamodule_base.get_datamodule(args)
wandb_logger = pl_loggers.WandbLogger()
trainer = Trainer.from_argparse_args(args, logger=wandb_logger)

trainer.fit(model, datamodule)
trainer.test(model, datamodule=datamodule)
