import json
import logging
import os
import uuid

import click
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from knowledge_tracing.datasets.XES3G5M.xes3g5m import (
    XES3G5MDataModule,
    XES3G5MDataModuleConfig,
)
from knowledge_tracing.models.DKT.dkt import DKTLightningModule


@click.command()
@click.option("--use-question-embeddings", type=bool, default=False)
@click.option("--use-previous-responses", type=bool, default=False)
@click.option("--input-dim", type=int, default=768)
@click.option("--hidden-dim", type=int, default=100)
@click.option("--lr", type=float, default=1e-2)
@click.option("--batch-size", type=int, default=64)
@click.option("--max-seq-length", type=int, default=200)
@click.option("--val-fold", type=int, default=4)
@click.option("--max-epochs", type=int, default=10)
@click.option("--output-dir", type=str, default="checkpoints")
@click.option("--log-dir", type=str, default="logs")
@click.option("--seed", type=int, default=42)
@click.option("--devices", type=list[int], default=[0])
def main(
    use_question_embeddings: bool,
    use_previous_responses: bool,
    input_dim: int,
    hidden_dim: int,
    lr: float,
    batch_size: int,
    max_seq_length: int,
    val_fold: int,
    max_epochs: int,
    output_dir: str,
    log_dir: str,
    seed: int,
    devices: list[int],
):
    run_id = uuid.uuid4()

    os.makedirs(f"{log_dir}/train/{run_id}", exist_ok=True)
    log_file = f"{log_dir}/train/{run_id}/train.log"
    logging.basicConfig(level=logging.INFO, filename=log_file)

    logger = logging.getLogger(__name__)
    logger.info(f"Run ID: {run_id}")
    logger.info("Training with the following parameters:")
    logger.info(
        f"Training with {use_question_embeddings=}, {use_previous_responses=}, {hidden_dim=}, {lr=}, {batch_size=}, {max_seq_length=}, {val_fold=}, {max_epochs=}, {output_dir=}"
    )

    config = {
        "use_question_embeddings": use_question_embeddings,
        "use_previous_responses": use_previous_responses,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "batch_size": batch_size,
        "max_seq_length": max_seq_length,
        "val_fold": val_fold,
        "output_dir": output_dir,
        "max_epochs": max_epochs,
    }
    with open(f"{log_dir}/train/{run_id}/config.json", "w") as f:
        json.dump(config, f)

    logger.info("Data module configuration:")
    logger.info(config)
    logger.info("Setting up data module...")

    dm = XES3G5MDataModule(XES3G5MDataModuleConfig())
    dm.prepare_data()
    dm.setup()

    logger.info("Data module setup complete.")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    logger.info("Setting up model...")

    model = DKTLightningModule(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        use_question_embeddings=use_question_embeddings,
        use_previous_responses=use_previous_responses,
    )

    logger.info(model)
    logger.info("Training...")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=devices,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename=f"{output_dir}/best",
                save_last=True,
            )
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    test_results = trainer.test(model, test_loader)
    for result in test_results:
        for key, value in result.items():
            logger.info(f"{key}: {value}")
    with open(f"{log_dir}/train/{run_id}/test_results.json", "w") as f:
        json.dump(test_results, f)
    logger.info("Testing complete.")


if __name__ == "__main__":
    main()
