# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import sys
from dataclasses import dataclass, field
import numpy as np
import tensorflow as tf
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import models.reg as rm
import include.callbacks as callbacks
import include.generators as generators
import include.loss as loss
import include.optimisers as optimisers
import models.seg as sm
from Data import Data_train
from saver import Saver
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import wandb


@dataclass
class Trainer:
    """Train Convolutional Neural Networks models."""

    generator: generators.Generator
    callbacks: list = field(default_factory=lambda: [])
    batch_size: int = 32
    N_epochs: int = 10

    def train_model(self, model: tf.keras.Model, data: Data_train) -> tf.keras.Model:
        """Train model and evaluate validation."""
        self.history = model.fit(
            self.generator,
            epochs=self.N_epochs,
            validation_data=(data.x.valid_data, data.y.valid_data),
            verbose=1,
            # steps_per_epoch=int(
            #     np.floor(data.x.train.shape[0] / self.batch_size)),
            callbacks=self.callbacks,
            shuffle=False,
        )

        return model

    def get_history(self):
        """Return history."""
        return self.history

    def get_val_loss(self):
        """Return best val loss."""
        return np.min(self.history.history["val_loss"])


class Model_training_manager:
    """Manager for segmentation, inversion with CNN models."""

    def __init__(self, cfg: DictConfig) -> None:
        self.prepare_data(cfg)
        self.build_model(cfg)
        self.prepare_training(cfg)
        self.saver = Saver()
        self.cfg = cfg

    def prepare_data(self, cfg: DictConfig) -> None:
        """Prepare Data inputs to the neural network and outputs (=labels, targets)."""

        self.data = instantiate(cfg.data.init)

        if cfg.model.type == "segmentation":
            self.data.prepare_input(
                cfg.data.input.chan_0,
                cfg.data.input.chan_1,
                cfg.data.input.chan_2,
                cfg.data.input.chan_3,
                cfg.data.input.chan_4,
            )
            self.data.prepare_output_segmentation(
                cfg.data.output.curve,
                cfg.data.output.min_w,
                cfg.data.output.max_w,
                cfg.data.output.param_curve,
            )
        if cfg.model.type == "embedding":
            self.data.prepare_input(
                cfg.data.input.chan_0,
                cfg.data.input.chan_1,
                cfg.data.input.chan_2,
                cfg.data.input.chan_3,
                cfg.data.input.chan_4,
            )
            self.data.prepare_output_embedding()
        if cfg.model.type in ("inversion"):
            self.data.prepare_input(
                cfg.data.input.chan_0,
                cfg.data.input.chan_1,
                cfg.data.input.chan_2,
                cfg.data.input.chan_3,
                cfg.data.input.chan_4,
                cfg.data.input.dir_seg_models,
            )
            self.data.prepare_output_inversion(cfg.data.output.N_emissions)

    def build_model(self, cfg: DictConfig) -> None:
        """Build the inversion or segmentation model."""

        if "load_weights" in cfg:
            self.model = tf.keras.models.load_model(
                os.path.join(cfg.dir_res, cfg.load_weights,
                             "weights_cp_best.h5"),
                compile=True,
            )

        else:
            if cfg.model.type.startswith("segmentation"):
                seg_builder = sm.Seg_model_builder(
                    cfg.model.name,
                    self.data.x.fields_input_shape,
                    self.data.y.classes,
                    self.data.x.n_layer,
                    self.data.x.xco2_noisy_chans,
                    cfg.model.dropout_rate,
                )
                self.model = seg_builder.get_model()
            elif cfg.model.type == "embedding":
                reg_builder = rm.Reg_model_builder(
                    name=cfg.model.name,
                    input_shape=self.data.x.fields_input_shape,
                    n_layer=self.data.x.n_layer,
                    noisy_chans=self.data.x.xco2_noisy_chans,
                    config=cfg
                )
                self.model = reg_builder.get_model()

            elif cfg.model.type == "inversion":
                reg_builder = rm.Reg_model_builder(
                    cfg.model.name,
                    self.data.x.fields_input_shape,
                    self.data.y.classes,
                    self.data.x.n_layer,
                    self.data.x.xco2_noisy_chans,
                    cfg.model.dropout_rate,
                    cfg.model.scaling_coefficient,
                    self.data.x.window_length,
                    cfg
                )
                self.model = reg_builder.get_model()
            else:
                print(f"Unknown model type: {cfg.model.type}")
                sys.exit()

            self.model.compile(
                optimizer=optimisers.define_optimiser(
                    cfg.training.optimiser, cfg.training.learning_rate
                ),
                loss=loss.define_loss(cfg.model.loss_func),
                metrics=loss.define_metrics(cfg.model.type),
                run_eagerly=True,
            )

    def prepare_training(self, cfg: DictConfig) -> None:
        """Prepare the training phase."""
        if cfg.model.type.startswith("segmentation"):
            gen_machine = generators.Generator(
                cfg.model.type,
                cfg.training.batch_size,
                cfg.augmentations.rot.range,
                cfg.augmentations.shift.range,
                cfg.augmentations.flip.bool,
                cfg.augmentations.shear.range,
                cfg.augmentations.zoom.range,
                cfg.augmentations.shuffle,
            )
            generator = gen_machine.flow(
                self.data.x.train_data[self.data.x.train_data_indexes], self.data.y.train)
        if cfg.model.type == "embedding":
            gen_machine = generators.Generator(
                cfg.model.type,
                cfg.training.batch_size,
                cfg.augmentations.rot.range,
                cfg.augmentations.shift.range,
                cfg.augmentations.flip.bool,
                cfg.augmentations.shear.range,
                cfg.augmentations.zoom.range,
                cfg.augmentations.shuffle,
            )
            generator = gen_machine.flow(
                self.data.x.train_data[self.data.x.train_data_indexes], self.data.y.train_data[self.data.y.train_data_indexes])
        elif cfg.model.type == "inversion":
            generator = generators.ScaleDataGen(
                self.data.x.train_data,
                self.data.x.train_data_indexes,
                self.data.x.plumes_train,
                self.data.x.xco2_back_train,
                self.data.x.xco2_alt_anthro_train,
                self.data.y.train_data,
                [False]*5,
                # self.data.x.scale_bool,
                self.data.x.fields_input_shape,
                cfg.training.batch_size,
                plume_scaling_min=cfg.augmentations.plume_scaling_min,
                plume_scaling_max=cfg.augmentations.plume_scaling_max,
                window_length=self.data.x.window_length,
            )
        else:
            print(f"Unknown model type: {cfg.model.type}")
            sys.exit()

        cbs = callbacks.get_lrscheduler(
            cfg.callbacks.learning_rate_monitor, [])
        cbs = callbacks.get_earlystopping(cfg.callbacks.early_stopping, cbs)
        self.trainer = Trainer(
            generator,
            cbs,
            cfg.training.batch_size,
            cfg.training.max_epochs,
        )

    def run(self) -> None:
        """Train the model with the training data."""
        wandb.login(key=self.cfg.wandb.key)
        config = OmegaConf.to_container(self.cfg, resolve=True)
        import pdb;pdb.set_trace()
        with wandb.init(project=self.cfg.wandb.project_name,
                        name=self.cfg.exp_name, config=config) as run:

            self.trainer.callbacks.append(
                callbacks.get_modelcheckpoint(self.cfg.callbacks.model_checkpoint, []))
            self.trainer.callbacks.append(WandbMetricsLogger())
            # self.trainer.callbacks.append(WandbModelCheckpoint("models"))
            self.model = self.trainer.train_model(self.model, self.data)
            run.save(os.path.abspath("config.yaml"))
        return self.trainer.get_val_loss()

    def save(self) -> None:
        """Save results of the run."""
        print("Saving at:", os.getcwd())
        self.saver.save_model_and_weights(self.model)
        # self.saver.save_weights(self.model)_util
