import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path

class CurveTraceNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.enc(x)
        out = self.dec(h)
        return out.squeeze(1)


class AITracer:
    def __init__(self, model_path="curve_trace_model.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_h = 256
        self.input_w = 128
        
        # Load model if available
        p = Path(model_path)
        if p.exists():
            try:
                ckpt = torch.load(str(p), map_location=self.device, weights_only=False)
                self.input_h = ckpt.get('input_h', 256)
                self.input_w = ckpt.get('input_w', 128)
                
                self.model = CurveTraceNet().to(self.device)
                self.model.load_state_dict(ckpt['state_dict'])
                self.model.eval()
                print(f"[OK] AI Model loaded from {model_path} (Device: {self.device})")
            except Exception as e:
                print(f"[WARN] Failed to load AI model: {e}")
        else:
            print(f"[WARN] AI model not found at {model_path}")

    def is_available(self):
        return self.model is not None

    @torch.no_grad()
    def trace(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        Runs the AI model on a cropped BGR image of the curve track.
        Returns a 1D numpy array of x-coordinates for each row in the original roi.
        """
        if self.model is None:
            raise RuntimeError("AI model not loaded.")
            
        orig_h, orig_w = roi_bgr.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return np.array([])

        # Preprocess: convert to grayscale, resize to expected input size, normalize to 0-1
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        
        # Add batch and channel dimensions: [1, 1, H, W]
        x_tensor = torch.from_numpy(roi_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        x_tensor = x_tensor.to(self.device)

        # Predict
        pred_tensor = self.model(x_tensor) # Shape: [1, input_h]
        pred_norm = pred_tensor.squeeze().cpu().numpy() # Shape: [input_h], values in [0, 1]

        # Scale prediction back to original image coordinates
        # 1. Scale width from [0,1] to [0, orig_w - 1]
        pred_x_small = pred_norm * max(0, orig_w - 1)
        
        # 2. Interpolate from input_h back to orig_h
        y_small = np.linspace(0, orig_h - 1, self.input_h)
        y_orig = np.arange(orig_h)
        pred_x_orig = np.interp(y_orig, y_small, pred_x_small)

        return pred_x_orig
