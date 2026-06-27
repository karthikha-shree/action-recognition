import torch
import torchvision.models.video as video_models
import logging

logger = logging.getLogger(__name__)

class ActionRecognitionModel:
    """
    Modular wrapper for loading torchvision pre-trained action recognition models
    and running inference.
    """
    def __init__(self, model_name: str = "r3d_18", use_gpu: bool = True):
        self.model_name = model_name
        self.device = torch.device(
            "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self) -> torch.nn.Module:
        """
        Loads the pre-trained torchvision model.
        """
        logger.info(f"Loading pre-trained {self.model_name} model from torchvision...")
        
        if self.model_name == "r3d_18":
            weights = video_models.R3D_18_Weights.DEFAULT
            model = video_models.r3d_18(weights=weights)
        elif self.model_name == "mc3_18":
            weights = video_models.MC3_18_Weights.DEFAULT
            model = video_models.mc3_18(weights=weights)
        elif self.model_name == "r2plus1d_18":
            weights = video_models.R2Plus1D_18_Weights.DEFAULT
            model = video_models.r2plus1d_18(weights=weights)
        elif self.model_name == "s3d":
            weights = video_models.S3D_Weights.DEFAULT
            model = video_models.s3d(weights=weights)
        else:
            raise ValueError(
                f"Unsupported model: {self.model_name}. "
                f"Choose from: r3d_18, mc3_18, r2plus1d_18, s3d"
            )
            
        self.categories = weights.meta["categories"]
        logger.info(f"Successfully loaded {self.model_name} model.")
        return model

    def predict(self, video_tensor: torch.Tensor, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Runs model inference on the preprocessed video tensor.
        
        Args:
            video_tensor: PyTorch tensor of shape (1, C, T, H, W).
            top_k: Number of top predictions to return.
            
        Returns:
            List of tuples: (predicted_class_id, confidence_score)
        """
        video_tensor = video_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(video_tensor)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
        # Get top-k scores and indices
        top_scores, top_indices = torch.topk(probabilities, k=top_k)
        
        results = []
        for score, idx in zip(top_scores, top_indices):
            results.append((int(idx.item()), float(score.item())))
            
        return results
