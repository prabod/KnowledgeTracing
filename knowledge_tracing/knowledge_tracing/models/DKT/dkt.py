import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAUROC


class DKTModel(nn.Module):
    def __init__(self, embed_size, hidden_dim=100):
        """
        Initialize the DKT model.

        Args:
            embed_size (int): The size of the embeddings for the questions and concepts.
            hidden_dim (int): The size of the hidden dimension of the LSTM.
        """
        super(DKTModel, self).__init__()
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(self.embed_size, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass of the DKT model.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_len, input_dim].
        """
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_dim]
        out = self.output_layer(lstm_out)  # [batch_size, seq_len, num_questions]
        pred = torch.sigmoid(out)  # probabilities for each question
        return pred


class DKTLightningModule(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 100,
        lr: float = 1e-3,
        use_question_embeddings: bool = True,
        use_previous_responses: bool = False,
    ):
        """
        Initialize the DKT Lightning module.

        Args:
            input_dim (int): The size of the input dimension.
            hidden_dim (int): The size of the hidden dimension of the LSTM.
            lr (float): The learning rate.
            use_question_embeddings (bool): Whether to use question embeddings.
        """
        super(DKTLightningModule, self).__init__()
        self.save_hyperparameters()
        self.use_question_embeddings = use_question_embeddings
        self.use_previous_responses = use_previous_responses
        self.lr = lr
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if self.use_previous_responses:
            self.input_dim += 1
        if self.use_question_embeddings:
            self.input_dim += input_dim
        self.model = DKTModel(self.input_dim, self.hidden_dim)
        self.criterion = nn.BCELoss(reduction="none")
        self.val_auc = BinaryAUROC()

    def forward(self, x):
        """
        Forward pass of the DKT model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step of the DKT model.
        """
        x, y, mask = self.prepare_batch(batch)
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y, mask)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def compute_loss(self, y_pred, y, mask):
        """
        Compute the loss of the DKT model.
        """
        loss = self.criterion(y_pred, y)
        mask = (mask > 0).bool()
        masked_loss = (loss * mask).sum() / mask.sum()
        self.log("train_loss", masked_loss)
        return masked_loss

    def prepare_batch(self, batch):
        """
        Prepare the batch for the DKT model.
        """
        question_embeddings = batch["question_embeddings"]
        concept_embeddings = batch["concept_embeddings"]
        selectmasks = (batch["selectmasks"].unsqueeze(-1) > 0).float()
        responses = batch["responses"].unsqueeze(-1).float()
        if self.use_question_embeddings:
            x = torch.cat([question_embeddings, concept_embeddings], dim=-1)
        else:
            x = concept_embeddings
        if self.use_previous_responses:
            previous_responses = responses.detach().clone()
            # t0 shouldn't have a previous response. It can be ignored by setting it to -1.
            # shift the responses by one time step
            previous_responses = torch.cat(
                [
                    torch.ones_like(previous_responses[:, :1]) * -1,
                    previous_responses[:, :-1],
                ],
                dim=1,
            )
            x = torch.cat([x, previous_responses], dim=-1)
        return x, responses, selectmasks

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the DKT model.
        """
        x, y, mask = self.prepare_batch(batch)
        y_pred = self(x)
        loss = self.compute_loss(y_pred, y, mask)
        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        preds_flat = y_pred[mask.bool()]
        labels_flat = y[mask.bool()]

        if labels_flat.numel() > 0:
            self.val_auc.update(preds_flat, labels_flat.int())

        return loss

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

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        self.log("val/auc", auc, prog_bar=True)
        self.val_auc.reset()
