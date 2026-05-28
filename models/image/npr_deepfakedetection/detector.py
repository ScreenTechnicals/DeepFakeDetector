import os
import sys
import base64
import io
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from deepsafe_sdk import ImageModel, PredictionResult

# Add model code to path for resnet50 import
MODEL_REPO_SUBDIR = "npr_deepfakedetection"
current_dir = os.path.dirname(os.path.abspath(__file__))
model_code_path = os.path.join(current_dir, MODEL_REPO_SUBDIR)
if model_code_path not in sys.path:
    sys.path.insert(0, model_code_path)

from networks.resnet import resnet50


class NPRDetector(ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        use_gpu = os.environ.get("USE_GPU", "false").lower() == "true"
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.target_layer = None

    def _find_last_conv_layer(self):
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def load(self):
        weights_path = self.weights_path("npr_deepfakedetection/weights/NPR.pth")
        net = resnet50(num_classes=1)
        state_dict = torch.load(
            weights_path, map_location=self.device, weights_only=False
        )
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)
        net.to(self.device)
        net.eval()
        self.model = net
        self.target_layer = self._find_last_conv_layer()

    def _build_gradcam_overlay(self, image: Image.Image, tensor: torch.Tensor) -> str:
        if self.target_layer is None:
            return None

        activations = []
        gradients = []

        def forward_hook(_module, _input, output):
            activations.append(output)

        def backward_hook(_module, _grad_input, grad_output):
            gradients.append(grad_output[0])

        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

        try:
            self.model.zero_grad(set_to_none=True)
            logit = self.model(tensor)
            probability = torch.sigmoid(logit).flatten()[0]
            logit.flatten()[0].backward()

            if not activations or not gradients:
                return None

            activation = activations[-1].detach()
            gradient = gradients[-1].detach()
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1)).squeeze()
            cam = cam.cpu().numpy()

            cam_min = float(cam.min())
            cam_max = float(cam.max())
            if cam_max - cam_min < 1e-8:
                return None
            cam = (cam - cam_min) / (cam_max - cam_min)

            cam_image = Image.fromarray(np.uint8(cam * 255), mode="L").resize(
                image.size, Image.Resampling.BILINEAR
            )
            cam_arr = np.asarray(cam_image).astype(np.float32) / 255.0

            rgba = np.zeros((cam_arr.shape[0], cam_arr.shape[1], 4), dtype=np.uint8)
            rgba[..., 0] = np.clip(255 * np.minimum(1.0, cam_arr * 1.8), 0, 255)
            rgba[..., 1] = np.clip(255 * np.maximum(0.0, 1.0 - np.abs(cam_arr - 0.55) * 2.0), 0, 255)
            rgba[..., 2] = np.clip(120 * np.maximum(0.0, 1.0 - cam_arr * 1.4), 0, 255)
            rgba[..., 3] = np.clip(205 * np.power(cam_arr, 0.75), 0, 205)

            overlay = Image.fromarray(rgba, mode="RGBA")
            buffer = io.BytesIO()
            overlay.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
        finally:
            forward_handle.remove()
            backward_handle.remove()

    def predict(self, input_data: str, threshold: float) -> PredictionResult:
        image = self.decode_image(input_data)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        heatmap = self._build_gradcam_overlay(image, tensor)
        with torch.no_grad():
            logit = self.model(tensor)
            probability = torch.sigmoid(logit).item()
        result = self.make_result(probability=probability, threshold=threshold)
        result.heatmap = heatmap
        result.heatmap_type = "gradcam"
        result.heatmap_model = "npr_deepfakedetection"
        return result
