import logging
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_examples import cli_lightning_logo
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.cli import LightningCLI
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ):
        if epoch == self.milestones[0]:
            # unfreeze 5 last layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )

        elif epoch == self.milestones[1]:
            # unfreeze remaing layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5],
                optimizer=optimizer,
                train_bn=self.train_bn,
            )


class FTRDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, Path] = "data",
        num_workers: int = 0,
        batch_size: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self._num_workers = num_workers
        self._batch_size = batch_size

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def val_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def test_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def setup(self, stage=None):
        ftr_dataset = ImageFolder(self.data_dir)

        num_train = int(0.75 * len(ftr_dataset))
        num_valid = int(0.15 * len(ftr_dataset))
        num_test = len(ftr_dataset) - num_train - num_valid

        self.train, self.val, self.test = random_split(
            ftr_dataset, [num_train, num_valid, num_test]
        )
        self.train.dataset.transform = self.train_transform
        self.val.dataset.transform = self.val_transform
        self.test.dataset.transform = self.test_transform

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )


class FTRModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet50",  # Name (as in ``torchvision.models``) of the feature extractor
        train_bn: bool = False,  # Whether the BatchNorm layers should be trainable
        milestones: tuple = (5, 10),  # List of two epochs milestones
        batch_size: int = 32,
        lr: float = 1e-3,  # Initial learning rate
        lr_scheduler_gamma: float = 1e-1,  # Factor by which the learning rate is reduced at each milestone
        num_workers: int = 6,
        **kwargs,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        self.__build_model()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.save_hyperparameters()

    def __build_model(self):
        # Feature Extractor
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)

        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)

        # Classifier:
        _fc_layers = [
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.Linear(32, 1),
        ]
        self.fc = nn.Sequential(*_fc_layers)

        # 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits

    def forward(self, x):
        """Forward pass. Returns logits."""
        # Feature extraction
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        # Classifier (returns logits):
        x = self.fc(x)
        return x

    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        # y_scores = torch.sigmoid(y_logits) # original
        y_scores = F.log_softmax(y_logits)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss
        train_loss = self.loss(y_logits, y_true)

        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_scores, y_true.int()), prog_bar=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)

        # 2. Compute loss
        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)

        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_scores, y_true.int()), prog_bar=True)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma
        )
        return [optimizer], [scheduler]


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_class_arguments(MilestonesFinetuning, "finetuning")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("finetuning.milestones", "model.milestones")
        parser.link_arguments("finetuning.train_bn", "model.train_bn")
        parser.set_defaults(
            {
                "trainer.max_epochs": 15,
                "trainer.weights_summary": None,
                "trainer.progress_bar_refresh_rate": 1,
                "trainer.num_sanity_val_steps": 0,
            }
        )

    def instantiate_trainer(self):
        finetuning_callback = MilestonesFinetuning(**self.config_init["finetuning"])
        self.trainer_defaults["callbacks"] = [finetuning_callback]
        super().instantiate_trainer()


def cli_main():
    MyLightningCLI(
        TransferLearningModel, CatDogImageDataModule, seed_everything_default=1234
    )


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
