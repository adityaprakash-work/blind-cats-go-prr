import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from ignite.engine import Engine, Events
from ignite.handlers import (
    ModelCheckpoint,
    EarlyStopping,
    TensorboardLogger,
    ProgressBar,
)


class Trainer1:
    def __init__(
        self,
        eeg_enc,
        img_enc,
        lat_dec,
        trn_loader,
        val_loader=None,
        freeze_eeg_enc=False,
        freeze_img_enc=False,
        freeze_lat_dec=False,
        num_scales=3,
        device="cpu",
    ):
        self.eeg_enc = eeg_enc
        self.img_enc = img_enc
        self.lat_dec = lat_dec
        self.trn_loader = trn_loader
        self.val_loader = val_loader
        self.freeze_eeg_enc = freeze_eeg_enc
        self.freeze_img_enc = freeze_img_enc
        self.freeze_lat_dec = freeze_lat_dec
        self.num_scales = num_scales
        self.device = device
        self.lambda_wt = nn.Parameter(torch.tensor(1.0))
        self.optimizer = Adam(
            list(eeg_enc.parameters())
            + list(img_enc.parameters())
            + list(lat_dec.parameters())
            + list(self.lambda_wt),
        )

    def _train_step(self, engine, batch):
        for param in self.eeg_enc.parameters():
            param.requires_grad = not self.freeze_eeg_enc
        for param in self.img_enc.parameters():
            param.requires_grad = not self.freeze_img_enc
        for param in self.lat_dec.parameters():
            param.requires_grad = not self.freeze_lat_dec
        self.eeg_enc.train()
        self.img_enc.train()
        self.lat_dec.train()
        self.optimizer.zero_grad()
        eeg, img = batch
        eeg = eeg.to(self.device)
        img = img.to(self.device)
        eeg_emb = self.eeg_enc(eeg)
        img_emb = self.img_enc(img)
        p_img = self.lat_dec(eeg_emb)
        cos_sim_loss = 1 - F.cosine_similarity(eeg_emb, img_emb).mean()
        _rec_scl_loss = {}
        for i in range(self.num_scales):
            sf = 0.5**i
            if i == 0:
                _rec_scl_loss[f"rec_scl_loss{sf}"] = F.mse_loss(p_img, img)

            else:
                term = F.mse_loss(
                    F.interpolate(p_img, scale_factor=sf, mode="bilinear"),
                    F.interpolate(img, scale_factor=sf, mode="bilinear"),
                )
                _rec_scl_loss[f"rec_scl_loss{sf}"] = term / sf
        rec_scl_loss = sum(_rec_scl_loss.values()) / len(_rec_scl_loss)
        # NOTE: Need to multiply by empirical values to normalize both losses.
        lamb = torch.clamp(self.lambda_wt, 0.1, 0.9)
        loss = lamb * cos_sim_loss + (1 - lamb) * rec_scl_loss
        loss.backward()
        self.optimizer.step()
        return {
            "cos_sim_loss": cos_sim_loss.item(),
            "rec_scl_loss": rec_scl_loss.item(),
            "loss": loss.item(),
            **_rec_scl_loss,
        }

    def _eval_step(self, engine, batch):
        self.eeg_enc.eval()
        self.img_enc.eval()
        self.lat_dec.eval()
        with torch.no_grad():
            eeg, img = batch
            eeg = eeg.to(self.device)
            img = img.to(self.device)
            eeg_emb = self.eeg_enc(eeg)
            img_emb = self.img_enc(img)
            p_img = self.lat_dec(eeg_emb)
            cos_sim_loss = 1 - F.cosine_similarity(eeg_emb, img_emb).mean()
            _rec_scl_loss = {}
            for i in range(self.num_scales):
                sf = 0.5**i
                if i == 0:
                    _rec_scl_loss[f"rec_scl_loss{sf}"] = F.mse_loss(p_img, img)

                else:
                    term = F.mse_loss(
                        F.interpolate(p_img, scale_factor=sf, mode="bilinear"),
                        F.interpolate(img, scale_factor=sf, mode="bilinear"),
                    )
                    _rec_scl_loss[f"rec_scl_loss{sf}"] = term / sf
            rec_scl_loss = sum(_rec_scl_loss.values()) / len(_rec_scl_loss)
            lamb = torch.clamp(self.lambda_wt, 0.1, 0.9)
            loss = lamb * cos_sim_loss + (1 - lamb) * rec_scl_loss
            return {
                "cos_sim_loss": cos_sim_loss.item(),
                "rec_scl_loss": rec_scl_loss.item(),
                "loss": loss.item(),
                **_rec_scl_loss,
            }

    def train(self, max_epochs=100, patience=10, k=28, log_dir="logs"):
        trainer = Engine(self._train_step)
        evaluator = Engine(self._eval_step)
        pbar = ProgressBar()
        pbar.attach(trainer)
        pbar.attach(evaluator)
        tb_logger = TensorboardLogger(log_dir)
        tb_logger.attach(trainer)
        tb_logger.attach(evaluator)

        @trainer.on(Events.ITERATION_COMPLETED(every=k))
        def log_images(engine):
            eeg, img = next(iter(self.val_loader))
            eeg = eeg.to(self.device)
            img = img.to(self.device)
            eeg_emb = self.eeg_enc(eeg)
            _ = self.img_enc(img)
            p_img = self.lat_dec(eeg_emb)
            img_grid = torch.cat((img, p_img), dim=3)
            tb_logger.writer.add_image(
                "Reconstructions",
                img_grid[0],
                engine.state.iteration,
                dataformats="NCHW",
            )

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_lambda_wt(engine):
            tb_logger.writer.add_scalar(
                "Lambda", self.lambda_wt.item(), engine.state.iteration
            )

        checkpoint_handler = ModelCheckpoint(
            log_dir, "twin", save_interval=1, n_saved=10, require_empty=False
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            checkpoint_handler,
            {
                "eeg_enc": self.eeg_enc,
                "img_enc": self.img_enc,
                "lat_dec": self.lat_dec,
            },
        )
        early_stopping_handler = EarlyStopping(
            patience=patience,
            score_function=lambda engine: -engine.state.metrics["loss"],
        )
        evaluator.add_event_handler(
            Events.COMPLETED,
            early_stopping_handler,
            trainer,
        )
        trainer.run(self.trn_loader, max_epochs=max_epochs)
        return {
            "eeg_enc": self.eeg_enc,
            "img_enc": self.img_enc,
            "lat_dec": self.lat_dec,
        }
