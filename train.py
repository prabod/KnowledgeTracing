import json
import logging
import os
import random
import uuid

import click
import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from knowledge_tracing.datasets.XES3G5M.xes3g5m import (
    XES3G5MDataModule,
    XES3G5MDataModuleConfig,
)
from knowledge_tracing.models.DKT.dkt import DKTLightningModule
from knowledge_tracing.models.DKT.dkt_with_interaction import (
    DKTLightningModuleWithInteraction,
)


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
@click.option("--repeat", type=int, default=10)
@click.option("--with-interaction", type=bool, default=False)
@click.option("--embed-size", type=int, default=100)
@click.option("--number-of-concepts", type=int, default=865)
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
    repeat: int,
    with_interaction: bool,
    embed_size: int,
    number_of_concepts: int,
):
    run_ids = [uuid.uuid4() for _ in range(repeat)]
    logger = logging.getLogger(__name__)

    test_results_all_runs = []

    for i in range(repeat):
        seed_everything(seed + i)
        run_id = run_ids[i]
        os.makedirs(f"{log_dir}/train/{run_id}", exist_ok=True)
        log_file = f"{log_dir}/train/{run_id}/train.log"
        logging.basicConfig(level=logging.INFO, filename=log_file)

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

        if with_interaction:
            model = DKTLightningModuleWithInteraction(
                number_of_concepts=number_of_concepts,
                embed_size=embed_size,
                hidden_dim=hidden_dim,
                lr=lr,
            )
        else:
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
            deterministic=True,
            logger=TensorBoardLogger(save_dir=f"{log_dir}/train/{run_id}"),
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
        test_results = test_results[0]
        test_results["run_id"] = str(run_id)
        test_results["seed"] = seed + i
        for key, value in test_results.items():
            logger.info(f"{key}: {value}")
        with open(f"{log_dir}/train/{run_id}/test_results.json", "w") as f:
            json.dump(test_results, f)
        logger.info("Testing complete.")
        test_results_all_runs.append(test_results)

    # calculate the mean and std of the test results
    test_results_all = {}
    for key in test_results_all_runs[0].keys():
        if key not in ["run_id", "seed", "config"]:
            test_results_mean = np.mean(
                [result[key] for result in test_results_all_runs]
            )
            test_results_std = np.std([result[key] for result in test_results_all_runs])
            test_results_all[key] = f"{test_results_mean:.4f}+-{test_results_std:.4f}"
    test_results_all["run_ids"] = [result["run_id"] for result in test_results_all_runs]
    test_results_all["seeds"] = [result["seed"] for result in test_results_all_runs]
    test_results_all["config"] = config
    with open(
        f"{log_dir}/train/test_results_{repeat=}_{lr=}_{hidden_dim=}_{input_dim=}_{use_question_embeddings=}_{use_previous_responses=}_{with_interaction=}.json",
        "w",
    ) as f:
        json.dump(test_results_all, f)


if __name__ == "__main__":
    main()
