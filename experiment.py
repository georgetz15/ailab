import lightning.pytorch.loggers as pl_loggers
import lightning.pytorch as L


def get_default_logger(dataset_name, model_name, save_dir='./results'):
    return pl_loggers.TensorBoardLogger(save_dir=save_dir,
                                        name=dataset_name,
                                        sub_dir=model_name)


def get_default_checkpoint_callback(monitored_metric, mode):
    return L.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor=monitored_metric,
        mode=mode,
        filename="best-{epoch:02d}",
        save_last=True,
    )


def train(model, data_module, max_epochs, monitored_metric, mode, **trainer_kwargs):
    checkpoint_callback = get_default_checkpoint_callback(monitored_metric, mode)
    tb_logger = get_default_logger(data_module.dataset_name, model.__class__.__name__)
    trainer = L.Trainer(max_epochs=max_epochs,
                        callbacks=[
                            checkpoint_callback,
                        ],
                        logger=tb_logger,
                        **trainer_kwargs,
                        )
    trainer.fit(model, data_module)

    return trainer