import os
import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (Trainer, seed_everything)

from src import utils


@hydra.main(version_base='1.0', config_path=f"./configs", config_name="config.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    log = utils.get_logger(__name__)
    config = utils.extras(config)
    seed_everything(config.seed, workers=True)

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = not config.train.get("force_restart", False) and config.train.get("ckpt_path")
    if ckpt_path:
        ckpt_path = utils.resolve_ckpt_path(ckpt_dir=config.paths.ckpt_dir, ckpt_path=ckpt_path)
        if os.path.exists(ckpt_path):
            log.info(f"Resuming checkpoint from <{ckpt_path}>")
        else:
            log.info(f"Failed to resume checkpoint from <{ckpt_path}>: file not exists. Skip.")
            ckpt_path = None

    # loading pipeline
    datamodule, pl_module, logger, callbacks = utils.common_pipeline(config)
    print('loading successfully')

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial",
                                               accelerator="gpu" if torch.cuda.is_available() else "cpu")

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(config=config, datamodule=datamodule, model=pl_module, trainer=trainer, callbacks=callbacks, logger=logger)

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=pl_module, datamodule=datamodule, ckpt_path=ckpt_path)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception("Metric for hyperparameter optimization not found! "
                        "Make sure the `optimized_metric` in `hparams_search` config is correct!")
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test"):
        log.info("Starting testing!")
        best_ckpt_path = os.path.join(config.paths.ckpt_dir, 'best.ckpt')
        trainer.test(model=pl_module, datamodule=datamodule, ckpt_path=best_ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(config=config, model=pl_module, datamodule=datamodule, trainer=trainer, callbacks=callbacks, logger=logger, )

    # Print path to best checkpoint
    log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    # TODO: automatically test after training
    return score


if __name__ == "__main__":
    main()
