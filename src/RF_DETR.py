import rfdetr as rfd
import torch


class RFDETRT:
    def __init__(
        self,
        model_size="nano",
    ):
        models = {"nano": rfd.RFDETRNano(),
                "small": rfd.RFDETRSmall(),
                "medium": rfd.RFDETRMedium()}

        self.model = models[model_size]
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def train(
        self,
        dataset_dir,
        epochs=10,
        batch_size=4,
        lr=1e-4,
        output_dir="runs/train",
        wandb_project = None,
        wandb_run = None
    ):
        print(f"Start training | device: {self.device}")

        self.model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=4,
            lr=lr,
            output_dir=output_dir,
            wandb=wandb_project is not None,
            project=wandb_project if wandb_project else None,
            run=wandb_run if wandb_run else None
        )

    def predict(self, image_path):
        return self.model.predict(image_path)

    def predict_batch(self, image_paths):
        results = []
        for path in image_paths:
            results.append(self.predict(path))
        return results

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
