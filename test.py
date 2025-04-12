import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from src import utils
from src.tasks import on_prediction_mode

log = utils.get_logger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="test.yaml")
def main(config: DictConfig):
    """ Contains minimal example of the testing/prediction pipeline. Evaluates given checkpoint on a testset.  """
    # resolve user provided config
    config = utils.resolve_experiment_config(config)
    # Applies optional utilities
    config = utils.extras(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    seed_everything(config.get("seed", 0), workers=True)

    # Convert relative ckpt path to absolute path if necessary
    if not os.path.isabs(config.ckpt_path):
        config.ckpt_path = utils.resolve_ckpt_path(ckpt_dir=config.paths.ckpt_dir, ckpt_path=config.ckpt_path)
    datamodule, pl_module, logger, callbacks = utils.common_pipeline(config)  # loading pipeline
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)  # Init lightning trainer
    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": config.ckpt_path})   # Log hyperparameters

    # Start prediction
    mode = config.mode
    log.info(f"Starting on mode='{mode}'!")

    # (1) Specify test dataset by configuring datamodule.test_split
    data_split = config.get('data_split') or config.datamodule.get('test_split', 'test')
    datamodule.hparams.test_split = data_split
    log.info(f"Loading test data from '{data_split}' dataset...")

    # Pytorch Lightning treat predict differently compared to what we commonly think of.
    # Must use this context manager and trainer.test to run prediction as expected.
    with on_prediction_mode(pl_module, enable=mode == 'predict'):
        trainer.test(model=pl_module, datamodule=datamodule, ckpt_path=config.ckpt_path)

    log.info(f"Finished mode='{mode}' on '{data_split}' dataset.")


if __name__ == "__main__":
    main()
