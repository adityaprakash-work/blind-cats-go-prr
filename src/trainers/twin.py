import math
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
        freeze_eeg_enc=True,
        freeze_img_enc=False,
        freeze_lat_dec=False,
        num_scales=3,
        device="cpu",
    ):
        self.eeg_enc = eeg_enc.to(device)
        self.img_enc = img_enc.to(device)
        self.lat_dec = lat_dec.to(device)
        self.eeg_enc.float()
        self.img_enc.float()
        self.lat_dec.float()
        self.trn_loader = trn_loader
        self.val_loader = val_loader
        self.freeze_eeg_enc = freeze_eeg_enc
        self.freeze_img_enc = freeze_img_enc
        self.freeze_lat_dec = freeze_lat_dec
        self.num_scales = num_scales
        self.device = device
        self.ope = Adam(list(eeg_enc.parameters()) + list(lat_dec.parameters()))
        self.opi = Adam(list(img_enc.parameters()) + list(lat_dec.parameters()))
        self.fbsi = None

    def _log_images(self, engine, tb_logger, set_name):
        loader = self.trn_loader if set_name == "Training" else self.val_loader
        eeg, ein, img = next(iter(loader))
        eeg = eeg.to(self.device).float()
        ein = ein.to(self.device).int()
        img = img.to(self.device).float()
        eeg_emb = self.eeg_enc(eeg, ein)
        img_emb = self.img_enc(img)
        p_img_e = self.lat_dec(eeg_emb)
        p_img_i = self.lat_dec(img_emb)
        img_grid = torch.cat((img, p_img_i, p_img_e), dim=3)
        img_grid = img_grid.clamp(0, 1).detach().cpu()
        tb_logger.writer.add_image(
            f"{set_name} Set Reconstruction",
            img_grid[:3],
            engine.state.iteration,
            dataformats="NCHW",
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
        op = self.opi if self.freeze_eeg_enc is True else self.ope
        op.zero_grad()
        eeg, ein, img = batch
        eeg = eeg.to(self.device).float()
        ein = ein.to(self.device).int()
        img = img.to(self.device).float()
        eeg_emb = self.eeg_enc(eeg, ein)
        img_emb = self.img_enc(img)
        if self.freeze_eeg_enc is True:
            p_img = self.lat_dec(img_emb)
        else:
            p_img = self.lat_dec(eeg_emb)
        cos_sim_loss = 1 - F.cosine_similarity(eeg_emb, img_emb).mean()
        _rec_scl_loss = {}
        for i in range(self.num_scales):
            sf = 0.5**i
            if i == 0:
                _rec_scl_loss[f"rec_scl_loss{sf}"] = F.mse_loss(p_img, img)
            else:
                _rec_scl_loss[f"rec_scl_loss{sf}"] = F.mse_loss(
                    F.interpolate(p_img, scale_factor=sf, mode="bilinear"),
                    F.interpolate(img, scale_factor=sf, mode="bilinear"),
                )
        # NOTE: The coefficients can be in some other patterns also right now
        # weights of all scales are equal.
        ns = self.num_scales
        weights = [(i + 1) / sum(range(1, ns + 1)) for i in range(ns)]
        rec_scl_loss = sum(
            wt * closs for wt, closs in zip(weights, _rec_scl_loss.values())
        )
        # NOTE: Both losses are combined with a weighted sum
        lamb = self.lambda_wt(engine.state.iteration)
        loss = lamb * cos_sim_loss + 2 * (1 - lamb) * rec_scl_loss
        loss.backward()
        op.step()
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
            eeg, ein, img = batch
            eeg = eeg.to(self.device).float()
            ein = ein.to(self.device).int()
            img = img.to(self.device).float()
            eeg_emb = self.eeg_enc(eeg, ein)
            img_emb = self.img_enc(img)
            p_img = self.lat_dec(eeg_emb)
            cos_sim_loss = 1 - F.cosine_similarity(eeg_emb, img_emb).mean()
            _rec_scl_loss = {}
            for i in range(self.num_scales):
                sf = 0.5**i
                if i == 0:
                    _rec_scl_loss[f"rec_scl_loss{sf}"] = F.mse_loss(p_img, img)
                else:
                    _rec_scl_loss[f"rec_scl_loss{sf}"] = F.mse_loss(
                        F.interpolate(p_img, scale_factor=sf, mode="bilinear"),
                        F.interpolate(img, scale_factor=sf, mode="bilinear"),
                    )
            ns = self.num_scales
            weights = [(i + 1) / sum(range(1, ns + 1)) for i in range(ns)]
            rec_scl_loss = sum(
                wt * closs for wt, closs in zip(weights, _rec_scl_loss.values())
            )
        return {
            "cos_sim_loss": cos_sim_loss.item(),
            "rec_scl_loss": rec_scl_loss.item(),
            **{key: value.item() for key, value in _rec_scl_loss.items()},
        }

    def lambda_wt(self, iteration):
        # i, f = iteration, self.fbsi
        # return 0.1 + 0.8 * (math.sin((i / f) * math.pi - (math.pi / 2)) + 1) / 2
        if self.freeze_eeg_enc is True:
            return 0.1
        else:
            return 0.9

    def fire(
        self,
        max_epochs=100,
        log_rec_interval=16,
        log_dir="logs",
        branch_switch_interval=128,
    ):
        # NOTE: At the start, EEG encoder is frozen and Image encoder is not.
        self.freeze_eeg_enc = True
        self.freeze_img_enc = False
        self.freeze_lat_dec = False
        self.fbsi = branch_switch_interval
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

        @trainer.on(Events.ITERATION_COMPLETED)
        def _log_lambda_wt(engine):
            i = engine.state.iteration
            tb_logger.writer.add_scalar("Lambda", self.lambda_wt(i), i)

        @trainer.on(Events.EPOCH_COMPLETED)
        def _run_evaluator(engine):
            k = engine.state.output.keys()
            avmetrics = {key: [] for key in k if key != "loss"}
            for batch in self.val_loader:
                eeg, ein, img = batch
                eeg = eeg.to(self.device).float()
                ein = ein.to(self.device).int()
                img = img.to(self.device).float()
                batch_metrics = self._eval_step(engine, (eeg, ein, img))
                for key, value in batch_metrics.items():
                    avmetrics[key].append(value)
            avmetrics = {
                key: sum(value) / len(value) for key, value in avmetrics.items()
            }
            for key, value in avmetrics.items():
                tb_logger.writer.add_scalar(
                    f"validation/{key}", value, engine.state.iteration
                )

            self._log_images(engine, tb_logger, "Validation")

        @trainer.on(Events.ITERATION_COMPLETED(every=branch_switch_interval))
        def _switch_branch(engine):
            if self.freeze_eeg_enc is True:
                self.freeze_eeg_enc = False
                self.freeze_img_enc = True
            else:
                self.freeze_eeg_enc = True
                self.freeze_img_enc = False

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
                "optim_eeg": self.ope,
                "optim_img": self.opi,
            },
        )

        trainer.run(self.trn_loader, max_epochs=max_epochs)

        return {
            "eeg_enc": self.eeg_enc,
            "img_enc": self.img_enc,
            "lat_dec": self.lat_dec,
        }
