import pytorch_lightning as pl

# 1. Import the new CocoDataModule instead of MOT15DataModule
from objtracker.datasets.coco import CocoDataModule
from objtracker.models.rf_detr import RFDETRLightning

if __name__ == "__main__":
    print("Initializing Data Pipeline...")
    # 2. Use the new module!
    data_module = CocoDataModule(batch_size=4)

    print("Initializing Model...")
    model = RFDETRLightning(model_size="nano", lr=1e-4)

    print("Starting Lightning Trainer...")
    trainer = pl.Trainer(max_epochs=10, accelerator="auto", log_every_n_steps=1)

    trainer.fit(model, datamodule=data_module)
