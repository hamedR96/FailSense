import torch
import torch.nn as nn
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import PeftModel
import os

def process_input(images, text):
    num_images = len(images) if isinstance(images, list) else 1
    prompt = " ".join(["<image>"] * num_images) + " evaluate en " + text
    return prompt


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection


class MLP_BLOCK(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.block = nn.Sequential(
            ResidualBlock(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.BatchNorm1d(dim_out),
            nn.GELU(),
            nn.Dropout(0.4),
        )
    def forward(self, x):
        return self.block(x)  # Residual connection



class FailSense(nn.Module):
    def __init__(self, vlm_model_id: str, feature_dim: int = 2304, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)

        # Load VLM components
        self.processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

        # Load base model and PEFT adapter
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            "google/paligemma2-3b-mix-224",
            device_map=self.device
        )
        self.vlm_model = PeftModel.from_pretrained(base_model, vlm_model_id)

        # Freeze VLM parameters
        for param in self.vlm_model.parameters():
            param.requires_grad = False

        # self.features: [batch_size, seq_len, hidden_dim]
        self.attention_layer = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=3, batch_first=True).to(self.device)

        # Binary classifier head
        self.classifier = nn.Sequential(
            MLP_BLOCK(feature_dim,512),
            MLP_BLOCK(512,256),
            MLP_BLOCK(256,128),
            nn.Linear(128, 1)  # Binary classification
        ).to(self.device)

        # Hook to capture middle layer features from language model
        self.features = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        """Register hook to capture middle layer features from language model"""
        # Access the language model (text decoder) layers
        if hasattr(self.vlm_model.base_model.model, 'language_model'):
            # PaliGemma structure
            language_model = self.vlm_model.base_model.model.language_model
            if hasattr(language_model, 'model') and hasattr(language_model.model, 'layers'):
                layers = language_model.model.layers
            else:
                layers = language_model.layers


        else:
            # Fallback - try to find transformer layers
            if hasattr(self.vlm_model.base_model.model, 'layers'):
                layers = self.vlm_model.base_model.model.layers
            else:
                # Last resort - find any transformer-like layers
                layers = self.vlm_model.base_model.model.transformer.h

        # Get middle layer (approximately half way through)
        middle_idx = len(layers) // 2
        target_layer = layers[middle_idx]


        def hook_fn(module, input, output):
            # Store the output features (hidden states)
            if isinstance(output, tuple):
                self.features = output[0]  # First element is usually hidden states
            else:
                self.features = output

        self.hook_handle = target_layer.register_forward_hook(hook_fn)

    def extract_features(self, images, text_prompts):
        """Extract features from language model middle layer"""
        model_inputs = self.processor(
            text=text_prompts,
            images=images,
            return_tensors="pt",
            padding="longest"
        ).to(self.device)

        with torch.no_grad():
            # Forward pass through VLM to trigger hook
            _ = self.vlm_model(**model_inputs)

        # Process captured features from language model
        if self.features is not None:
            token_features = self.features.to(self.device)  # [B, L, D]
            attn_output, attn_weights = self.attention_layer(token_features, token_features, token_features)
            pooled = attn_output.mean(dim=1)
            return pooled

        else:
            raise RuntimeError("No features captured. Check hook registration on language model layers.")

    def forward(self, images, text_prompts):
        """Forward pass through VLM + classifier"""
        # Extract features from VLM
        features = self.extract_features(images, text_prompts)

        # Pass through classifier
        logits = self.classifier(features)
        return logits

    def predict(self, images, text_prompts):
        """Make predictions (0 or 1)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, text_prompts)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).int().squeeze()
        return predictions, probs

    def cleanup(self):
        """Remove hooks"""
        if self.hook_handle:
            self.hook_handle.remove()

    def save_classifier(self, path="./failsense_classifier"):
        os.makedirs(path, exist_ok=True)
        torch.save({
            'classifier': self.classifier.state_dict(),
            'attention_layer': self.attention_layer.state_dict()
        }, f"{path}/head_components.pt")

    def load_classifier(self,path):
        checkpoint = torch.load(path, map_location=self.device)
        self.classifier.load_state_dict(checkpoint['classifier'])
        self.attention_layer.load_state_dict(checkpoint['attention_layer'])

