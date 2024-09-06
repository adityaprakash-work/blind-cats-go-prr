import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from ignite.engine import Engine, Events
from ignite.handlers import (
    ModelCheckpoint,
    TensorboardLogger,
    ProgressBar,
)
from ignite.contrib.handlers.tensorboard_logger import OutputHandler


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
            + [self.lambda_wt],
        )

    def _log_images(self, engine, tb_logger, set_name):
        eeg, img = next(iter(self.val_loader))
        eeg = eeg.to(self.device)
        img = img.to(self.device)
        eeg_emb = self.eeg_enc(eeg)
        _ = self.img_enc(img)
        p_img = self.lat_dec(eeg_emb)
        img_grid = torch.cat((img, p_img), dim=3)
        tb_logger.writer.add_image(
            f"{set_name} Set Reconstruction",
            img_grid[:3],
            engine.state.iteration,
            dataformats="NCHW",
        )

    @staticmethod
    def _log_avg_eval_output(engine, engine2, tb_logger):
        for key, value in engine.state.output.items():
            tb_logger.writer.add_scalar(
                f"Validation/{key}",
                value,
                engine2.state.epoch,
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
            nf = (img.shape[-1] * img.shape[-2]) * sf**2
            if i == 0:
                _rec_scl_loss[f"rec_scl_loss{sf}"] = F.mse_loss(p_img, img) / nf
            else:
                term = F.mse_loss(
                    F.interpolate(p_img, scale_factor=sf, mode="bilinear"),
                    F.interpolate(img, scale_factor=sf, mode="bilinear"),
                )
                _rec_scl_loss[f"rec_scl_loss{sf}"] = term / nf
        # NOTE: The coefficients can be in some other patterns also right now
        # weights of all scales are equal.
        rec_scl_loss = sum(_rec_scl_loss.values()) / len(_rec_scl_loss)
        # NOTE: Both losses are combined with a weighted sum
        lamb = torch.clamp(self.lambda_wt, 0.1, 0.9)
        loss = lamb * cos_sim_loss + (1 - lamb) * rec_scl_loss
        loss.backward()
        self.optimizer.step()
        return {
            "cos_sim_loss": cos_sim_loss.item(),
            "rec_scl_loss": rec_scl_loss.item(),
            "loss": loss.item(),
            **{key: value.item() for key, value in _rec_scl_loss.items()},
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
                nf = (img.shape[-1] * img.shape[-2]) * sf**2
                if i == 0:
                    _rec_scl_loss[f"rec_scl_loss{sf}"] = F.mse_loss(p_img, img) / nf
                else:
                    term = F.mse_loss(
                        F.interpolate(p_img, scale_factor=sf, mode="bilinear"),
                        F.interpolate(img, scale_factor=sf, mode="bilinear"),
                    )
                    _rec_scl_loss[f"rec_scl_loss{sf}"] = term / nf
            rec_scl_loss = sum(_rec_scl_loss.values()) / len(_rec_scl_loss)
            lamb = torch.clamp(self.lambda_wt, 0.1, 0.9)
            loss = lamb * cos_sim_loss + (1 - lamb) * rec_scl_loss
            return {
                "cos_sim_loss": cos_sim_loss.item(),
                "rec_scl_loss": rec_scl_loss.item(),
                "loss": loss.item(),
                **{key: value.item() for key, value in _rec_scl_loss.items()},
            }

    def fire(
        self,
        max_epochs=100,
        log_rec_interval=2,
        log_dir="logs",
    ):
        trainer = Engine(self._train_step)
        evaluator = Engine(self._eval_step)
        tpbar = ProgressBar()
        epbar = ProgressBar()
        tpbar.attach(trainer)
        epbar.attach(evaluator)
        tb_logger = TensorboardLogger(log_dir)
        tb_logger.attach(
            trainer,
            log_handler=OutputHandler(
                tag="training",
                output_transform=lambda x: x,
            ),
            event_name=Events.ITERATION_COMPLETED,
        )

        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=log_rec_interval),
            self._log_images,
            tb_logger,
            "Training",
        )

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            self._log_images,
            tb_logger,
            "Validation",
        )

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_lambda_wt(engine):
            tb_logger.writer.add_scalar(
                "Lambda", self.lambda_wt.item(), engine.state.iteration
            )

        checkpoint_handler = ModelCheckpoint(
            log_dir,
            n_saved=10,
            require_empty=False,
            score_function=lambda engine: -engine.state.output["loss"],
        )

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            checkpoint_handler,
            {
                "eeg_enc": self.eeg_enc,
                "img_enc": self.img_enc,
                "lat_dec": self.lat_dec,
                "optimizer": self.optimizer,
            },
        )

        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            self._log_avg_eval_output,
            trainer,
            tb_logger,
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_evaluator(engine):
            evaluator.run(self.val_loader, max_epochs=1)

        trainer.run(self.trn_loader, max_epochs=max_epochs)
        return {
            "eeg_enc": self.eeg_enc,
            "img_enc": self.img_enc,
            "lat_dec": self.lat_dec,
        }
