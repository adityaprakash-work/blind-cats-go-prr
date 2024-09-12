import math
import wandb
import numpy as np
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
from ignite.contrib.handlers import WandBLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ..losses import StochasticPatchMSELoss


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
        self.ope = Adam(list(eeg_enc.parameters()))
        self.opi = Adam(list(img_enc.parameters()) + list(lat_dec.parameters()))
        self.fbsi = None
        self.celoss = float("inf")
        self.curr_eval_branch = None
        self.lambda_bounds = (0.0, 1.0)

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
        branch = "I" if self.freeze_eeg_enc is True else "E"
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
        # NOTE: Both losses are combined with a weighted sum.
        # NOTE: The factor '100'is empirical.
        lamb = self.lambda_wt(engine.state.iteration)
        loss = lamb * cos_sim_loss + 10 * (1 - lamb) * rec_scl_loss
        loss.backward()
        op.step()
        return {
            f"{branch}/cos_sim_loss": cos_sim_loss.item(),
            f"{branch}/rec_scl_loss": rec_scl_loss.item(),
            f"{branch}/loss": loss.item(),
            **{f"{branch}/{k}": v.item() for k, v in _rec_scl_loss.items()},
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
            branch = self.curr_eval_branch
            if branch == "I":
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
            ns = self.num_scales
            weights = [(i + 1) / sum(range(1, ns + 1)) for i in range(ns)]
            rec_scl_loss = sum(
                wt * closs for wt, closs in zip(weights, _rec_scl_loss.values())
            )
            lamb = 0.0 if branch == "I" else 1.0
            loss = lamb * cos_sim_loss + 10 * (1 - lamb) * rec_scl_loss
        return {
            f"{branch}/cos_sim_loss": cos_sim_loss.item(),
            f"{branch}/rec_scl_loss": rec_scl_loss.item(),
            f"{branch}/loss": loss.item(),
            **{f"{branch}/{k}": v.item() for k, v in _rec_scl_loss.items()},
        }

    def lambda_wt(self, iteration):
        if self.freeze_eeg_enc is True:
            return self.lambda_bounds[0]
        else:
            return self.lambda_bounds[1]

    def fire(
        self,
        max_epochs=100,
        log_rec_interval=16,
        log_dir="logs",
        branch_switch_interval=128,
        lambda_bounds=(0.0, 1.0),
    ):
        self.lambda_bounds = lambda_bounds
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
            ky = [k.split("/")[1] for k in engine.state.output.keys()]
            avmetrics = {f"E/{k}": [] for k in ky} | {f"I/{k}": [] for k in ky}

            for batch in self.val_loader:
                eeg, ein, img = batch
                eeg = eeg.to(self.device).float()
                ein = ein.to(self.device).int()
                img = img.to(self.device).float()
                self.curr_eval_branch = "I"
                batch_metrics_i = self._eval_step(engine, (eeg, ein, img))
                self.curr_eval_branch = "E"
                batch_metrics_e = self._eval_step(engine, (eeg, ein, img))
                self.celoss = batch_metrics_e["E/loss"]
                for key, value in batch_metrics_i.items():
                    avmetrics[key].append(value)
                for key, value in batch_metrics_e.items():
                    avmetrics[key].append(value)
            avmetrics = {
                key: sum(value) / len(value) for key, value in avmetrics.items()
            }
            for key, value in avmetrics.items():
                tb_logger.writer.add_scalar(
                    f"validation/{key}", value, engine.state.epoch
                )

            self._log_images(engine, tb_logger, "Validation")

        @trainer.on(Events.ITERATION_COMPLETED(every=branch_switch_interval))
        def _switch_branch(engine):
            if self.freeze_eeg_enc is True:
                self.freeze_eeg_enc = False
                self.freeze_img_enc = True
                self.freeze_lat_dec = True
            else:
                self.freeze_eeg_enc = True
                self.freeze_img_enc = False
                self.freeze_lat_dec = False

        checkpoint_handler = ModelCheckpoint(
            log_dir,
            n_saved=1,
            require_empty=False,
            score_function=lambda _: -self.celoss,
        )

        trainer.add_event_handler(
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


class KaggleTwinTrainer:
    def __init__(
        self,
        eeg_enc,
        img_enc,
        lat_dec,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eeg_enc = eeg_enc
        self.img_enc = img_enc
        self.lat_dec = lat_dec
        self.eeg_enc.to(self.device)
        self.img_enc.to(self.device)
        self.lat_dec.to(self.device)
        self.ope = Adam(eeg_enc.parameters())
        self.opi = Adam(list(img_enc.parameters()) + list(lat_dec.parameters()))
        self.spl_e = StochasticPatchMSELoss(16)
        self.spl_i = StochasticPatchMSELoss(16)
        self.trn_eng_e = None
        self.evl_eng_e = None
        self.trn_eng_i = None
        self.evl_eng_i = None

    def _train_step_e(self, engine, batch):
        for param in self.eeg_enc.parameters():
            param.requires_grad = True
        for param in self.img_enc.parameters():
            param.requires_grad = False
        for param in self.lat_dec.parameters():
            param.requires_grad = False
        # NOTE: The other models can be set to `eval` mode if needed in other
        # architectures.
        self.eeg_enc.train()

        self.ope.zero_grad()

        eeg, img = batch
        eeg = eeg.to(self.device).float()
        img = img.to(self.device).float()
        eeg_emb = self.eeg_enc(eeg)
        mu, logvar = self.img_enc(img)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        img_emb = eps.mul(std).add_(mu)

        cos_sim_loss = 1 - F.cosine_similarity(eeg_emb, img_emb).mean()
        p_img = self.lat_dec(eeg_emb)
        # TODO: Multi-scale weighted loss can be added here.
        rec_loss = self.spl_e(img, p_img)

        loss = cos_sim_loss + rec_loss
        loss.backward()

        self.ope.step()

        return {
            "E/cos_sim_loss": cos_sim_loss.item(),
            "E/rec_loss": rec_loss.item(),
            "E/loss": loss.item(),
        }

    def _eval_step_e(self, engine, batch):
        self.eeg_enc.eval()
        self.img_enc.eval()
        self.lat_dec.eval()
        with torch.no_grad():
            eeg, img = batch
            eeg = eeg.to(self.device).float()
            img = img.to(self.device).float()
            eeg_emb = self.eeg_enc(eeg)
            mu, logvar = self.img_enc(img)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            img_emb = eps.mul(std).add_(mu)

            cos_sim_loss = 1 - F.cosine_similarity(eeg_emb, img_emb).mean()
            p_img = self.lat_dec(eeg_emb)
            rec_loss = self.spl_e(img, p_img)

            loss = cos_sim_loss + rec_loss

        return {
            "E/cos_sim_loss": cos_sim_loss.item(),
            "E/rec_loss": rec_loss.item(),
            "E/loss": loss.item(),
        }

    def _train_step_i(self, engine, batch):
        for param in self.img_enc.parameters():
            param.requires_grad = True
        for param in self.lat_dec.parameters():
            param.requires_grad = True
        self.img_enc.train()
        self.lat_dec.train()

        self.opi.zero_grad()

        _, img = batch
        img = img.to(self.device).float()
        mu, logvar = self.img_enc(img)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        img_emb = eps.mul(std).add_(mu)
        p_img = self.lat_dec(img_emb)
        rec_loss = self.spl_i(img, p_img)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = rec_loss + kld_loss
        loss.backward()

        self.opi.step()

        return {
            "I/rec_loss": rec_loss.item(),
            "I/kld_loss": kld_loss.item(),
            "I/loss": loss.item(),
        }

    def _eval_step_i(self, engine, batch):
        self.eeg_enc.eval()
        self.img_enc.eval()
        self.lat_dec.eval()
        with torch.no_grad():
            _, img = batch
            img = img.to(self.device).float()
            mu, logvar = self.img_enc(img)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            img_emb = eps.mul(std).add_(mu)
            p_img = self.lat_dec(img_emb)
            rec_loss = self.spl_i(img, p_img)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = rec_loss + kld_loss

        return {
            "I/rec_loss": rec_loss.item(),
            "I/kld_loss": kld_loss.item(),
            "I/loss": loss.item(),
        }

    # TODO: Merge the two `fire` methods into one.
    def fire_e(
        self,
        trn_loader,
        val_loader,
        max_epochs=10,
        patch_size=16,
        refesh_engines=False,
        wb_name="epoch-m-n",
        wb_group="eeg-branch",
        wb_project="blind-cats-go-prr",
    ):
        self.spl_e = StochasticPatchMSELoss(patch_size)
        if refesh_engines:
            self.trn_eng_e = Engine(self._train_step_e)
            self.evl_eng_e = Engine(self._eval_step_e)
            tpbar = ProgressBar()
            epbar = ProgressBar()
            tpbar.attach(self.trn_eng_e)
            epbar.attach(self.evl_eng_e)
            wandb_logger = WandBLogger(
                name=wb_name,
                project=wb_project,
                group=wb_group,
            )
            wandb_logger.attach_output_handler(
                self.trn_eng_e,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda x: x,
            )

            wandb_logger.attach_output_handler(
                self.evl_eng_e,
                event_name=Events.EPOCH_COMPLETED,
                tag="evaluation",
                output_transform=lambda x: x,
                global_step_transform=lambda *_: self.trn_eng_e.state.iteration,
            )

            @self.trn_eng_e.on(Events.EPOCH_COMPLETED)
            def _run_evaluator(engine):
                self.evl_eng_e.run(val_loader)

            @self.trn_eng_e.on(Events.ITERATION_COMPLETED(every=32))
            def _log_images_trn(engine):
                self._log_images(engine, trn_loader, "Training")

            @self.evl_eng_e.on(Events.ITERATION_COMPLETED(every=32))
            def _log_images_evl(engine):
                self._log_images(engine, val_loader, "Validation")

        self.trn_eng_e.run(trn_loader, max_epochs=max_epochs)

    def fire_i(
        self,
        trn_loader,
        val_loader,
        max_epochs=10,
        patch_size=16,
        refesh_engines=False,
        wb_name="epoch-m-n",
        wb_group="img-branch",
        wb_project="blind-cats-go-prr",
    ):
        self.spl_i = StochasticPatchMSELoss(patch_size)
        if refesh_engines:
            self.trn_eng_i = Engine(self._train_step_i)
            self.evl_eng_i = Engine(self._eval_step_i)
            tpbar = ProgressBar()
            epbar = ProgressBar()
            tpbar.attach(self.trn_eng_i)
            epbar.attach(self.evl_eng_i)
            wandb_logger = WandBLogger(
                name=wb_name,
                project=wb_project,
                group=wb_group,
            )
            wandb_logger.attach_output_handler(
                self.trn_eng_i,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda x: x,
            )

            wandb_logger.attach_output_handler(
                self.evl_eng_i,
                event_name=Events.EPOCH_COMPLETED,
                tag="evaluation",
                output_transform=lambda x: x,
                global_step_transform=lambda *_: self.trn_eng_i.state.iteration,
            )

            @self.trn_eng_i.on(Events.EPOCH_COMPLETED)
            def _run_evaluator(engine):
                self.evl_eng_i.run(val_loader)

        self.trn_eng_i.run(trn_loader, max_epochs=max_epochs)

    def _log_images(self, engine, loader, set_name):
        eeg, img = next(iter(loader))
        eeg = eeg.to(self.device).float()
        img = img.to(self.device).float()
        eeg_emb = self.eeg_enc(eeg)
        mu, logvar = self.img_enc(img)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        img_emb = eps.mul(std).add_(mu)
        p_img_e = self.lat_dec(eeg_emb)
        p_img_i = self.lat_dec(img_emb)
        img_grid = torch.cat((img, p_img_i, p_img_e), dim=3)
        img_grid = img_grid.clamp(0, 1).detach().cpu().numpy()
        iter_num = engine.state.iteration
        wandb.log(
            {f"{set_name} Set Reconstruction: {iter_num}": [wandb.Image(img_grid)]}
        )
