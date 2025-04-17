import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import BinaryAUROC


class DKTModelWithInteraction(nn.Module):
    def __init__(self, number_of_concepts, embed_size, hidden_dim=100):
        """
        Initialize the DKT model.

        Args:
            number_of_concepts (int): The number of questions in the dataset.
            embed_size (int): The size of the embeddings for the questions and concepts.
            hidden_dim (int): The size of the hidden dimension of the LSTM.
        """
        super(DKTModelWithInteraction, self).__init__()
        self.number_of_concepts = number_of_concepts
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.interaction_embedding = nn.Embedding(
            number_of_concepts * 2 + 1, self.embed_size, padding_idx=0
        )
        self.lstm = nn.LSTM(self.embed_size, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, self.number_of_concepts)

    def forward(
        self,
        questions,
        responses,
    ):
        """
        Forward pass of the DKT model.

        Args:
            questions (torch.Tensor): The input tensor of shape [batch_size, seq_len].
            responses (torch.Tensor): The input tensor of shape [batch_size, seq_len].
        """
        # Create interaction embeddings
        questions = questions
        interaction_indices = questions + self.number_of_concepts * responses
        interaction_embeddings = self.interaction_embedding(interaction_indices)
        lstm_out, _ = self.lstm(interaction_embeddings)
        out = self.output_layer(lstm_out)
        pred = torch.sigmoid(out)
        return pred


class DKTLightningModuleWithInteraction(pl.LightningModule):
    def __init__(
        self,
        number_of_concepts,
        embed_size,
        hidden_dim=100,
        lr=1e-3,
    ):
        super(DKTLightningModuleWithInteraction, self).__init__()
        self.save_hyperparameters()
        self.model = DKTModelWithInteraction(number_of_concepts, embed_size, hidden_dim)
        self.criterion = nn.BCELoss(reduction="none")
        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()
        self.lr = lr
        self.number_of_concepts = number_of_concepts
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim

    def forward(self, questions, responses):
        return self.model(questions, responses)

    def training_step(self, batch, batch_idx):
        q_sequences, r_sequences, q_sequences_rshifted, r_sequences_rshifted, mask = (
            self.prepare_batch(batch)
        )
        y_pred = self(q_sequences, r_sequences)
        y = r_sequences_rshifted
        y_pred = (
            y_pred
            * one_hot(
                q_sequences_rshifted.long(),
                num_classes=self.number_of_concepts,
            )
        ).sum(dim=-1)
        loss = self.compute_loss(y_pred, y, mask)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        q_sequences, r_sequences, q_sequences_rshifted, r_sequences_rshifted, mask = (
            self.prepare_batch(batch)
        )
        y_pred = self(q_sequences, r_sequences)
        y = r_sequences_rshifted
        y_pred = (
            y_pred
            * one_hot(
                q_sequences_rshifted.long(),
                num_classes=self.number_of_concepts,
            )
        ).sum(dim=-1)
        loss = self.compute_loss(y_pred, y, mask)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        preds_flat = y_pred[mask.bool()]
        labels_flat = y[mask.bool()]

        if labels_flat.numel() > 0:
            self.val_auc.update(preds_flat, labels_flat.int())
        return loss

    def test_step(self, batch, batch_idx):
        q_sequences, r_sequences, q_sequences_rshifted, r_sequences_rshifted, mask = (
            self.prepare_batch(batch)
        )
        y_pred = self(q_sequences, r_sequences)
        y = r_sequences_rshifted
        y_pred = (
            y_pred
            * one_hot(
                q_sequences_rshifted.long(),
                num_classes=self.number_of_concepts,
            )
        ).sum(dim=-1)
        loss = self.compute_loss(y_pred, y, mask)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        preds_flat = y_pred[mask.bool()]
        labels_flat = y[mask.bool()]
        if labels_flat.numel() > 0:
            self.test_auc.update(preds_flat, labels_flat.int())
        return loss

    def compute_loss(self, y_pred, y, mask):
        loss = self.criterion(y_pred.float(), y.float())
        mask = (mask > 0).bool()
        masked_loss = (loss * mask).sum() / mask.sum()
        return masked_loss

    def prepare_batch(self, batch):
        concepts = batch["concepts"]
        responses = batch["responses"]
        mask = (batch["selectmasks"] > 0).bool()

        # for each sequence, remove the last question and response before padding
        # because the last question and response are not used for training
        q_sequences = [concepts[i][: mask[i].sum() - 1] for i in range(len(concepts))]
        r_sequences = [responses[i][: mask[i].sum() - 1] for i in range(len(responses))]

        q_sequences_rshifted = [
            concepts[i][1 : mask[i].sum()] for i in range(len(concepts))
        ]
        r_sequences_rshifted = [
            responses[i][1 : mask[i].sum()] for i in range(len(responses))
        ]

        q_sequences = pad_sequence(q_sequences, batch_first=True, padding_value=-1)
        r_sequences = pad_sequence(r_sequences, batch_first=True, padding_value=0)
        q_sequences_rshifted = pad_sequence(
            q_sequences_rshifted, batch_first=True, padding_value=-1
        )
        r_sequences_rshifted = pad_sequence(
            r_sequences_rshifted, batch_first=True, padding_value=0
        )
        new_mask = (q_sequences != -1) * (r_sequences != 0)

        q_sequences = q_sequences * new_mask
        r_sequences = r_sequences * new_mask
        q_sequences_rshifted = q_sequences_rshifted * new_mask
        r_sequences_rshifted = r_sequences_rshifted * new_mask
        return (
            q_sequences,
            r_sequences,
            q_sequences_rshifted,
            r_sequences_rshifted,
            new_mask,
        )

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        self.log("val/auc", auc, prog_bar=True)
        self.val_auc.reset()

    def on_test_epoch_end(self):
        auc = self.test_auc.compute()
        self.log("test/auc", auc, prog_bar=True)
        self.test_auc.reset()

    def configure_optimizers(self):
        """
        Configure the optimizer, learning rate scheduler, and gradient clipping.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
