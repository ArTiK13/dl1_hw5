import warnings

import hydra
import torch
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


def build_optimizer(config, model):
    admm_lr = config.optimizer.get("admm_lr")
    if admm_lr is None:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        return instantiate(config.optimizer, params=trainable_params)

    base_params = []
    admm_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("reconstructor.raw_"):
            admm_params.append(param)
        else:
            base_params.append(param)

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params})
    if admm_params:
        param_groups.append({"params": admm_params, "lr": admm_lr})

    optimizer_config = OmegaConf.to_container(config.optimizer, resolve=True)
    optimizer_cls = get_class(optimizer_config["_target_"])
    optimizer_kwargs = {
        key: value
        for key, value in optimizer_config.items()
        if key not in ("_target_", "admm_lr")
    }
    return optimizer_cls(param_groups, **optimizer_kwargs)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(config, model)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
    )

    trainer.train()


if __name__ == "__main__":
    main()
