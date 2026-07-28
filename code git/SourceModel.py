import copy
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn

class MobileNetV2Source(nn.Module):
    """MobileNetV2Source with customizable output classes for source domain.

    Args:
        source_num_classes (int): Number of output classes.
        dropout_p (float, optional): Dropout probability.

    Attributes:
        encoder (torch.nn.Module): MobileNetV2Source feature encoder.
        fc (torch.nn.Module): Fully connected layer for classification.
        dim (int): Dimensionality of the classifier's input features.

    Methods:
        forward(x, return_feats=False): Forward pass through the network.
        resume_from_ckpt(ckpt_path=None): Load model weights from a checkpoint file.
    """

    def __init__(self, source_num_classes, dropout_p=0.1):
        """
        Initializes MobileNetV2Source

        Args:
            source_num_classes (int): Number of output classes.
            dropout_p (float, optional): Dropout probability.
        """
        super(MobileNetV2Source, self).__init__()

        # Load pre-trained MobileNetV2 model, with weights trained on ImageNet-1K.
        self.encoder = models.__dict__['mobilenet_v2'](weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Clone the final classification head from the pre-trained model.
        self.fc = copy.deepcopy(self.encoder.classifier)

        # Extract the input feature size of the last Linear layer in the classifier.
        # This will be used to create a new classifier head
        self.dim = list(self.fc.children())[-1].in_features

        # Replace the original classifier with an identity layer, meaning the encoder now only outputs features, not classification scores.
        self.encoder.classifier = nn.Identity()

        # New classification head with dropout
        if source_num_classes != 1000: # If the desired number of output classes ≠ 1000 (ImageNet), replace the classifier.
            self.fc = nn.Sequential(
                nn.Dropout(p=dropout_p, inplace=False),
                nn.Linear(self.dim, source_num_classes) # creates a fully connected layer. input: self.dim features (output from the encoder), output: source_num_classes

            )

    # Forward Pass
    def forward(self, x, return_feats=False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.
            return_feats (bool, optional): If True, returns intermediate features. Defaults to False.

        Returns:
            torch.Tensor or tuple: Model output tensor or tuple of output tensor and intermediate features.
        """
        # Extract deep features using the encoder (MobileNetV2 backbone)
        feats = self.encoder(x)

        # Pass features through the custom classifier to get class scores
        scores = self.fc(feats)

        if return_feats:
            return scores, feats

        return scores

    def resume_from_ckpt(self, ckpt_path: str = None):
        """
        Load model weights from a checkpoint file.

        Args:
            ckpt_path (str, optional): Path to the checkpoint file. Defaults to None.
        """
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt
            current_keys = self.state_dict()
            new_dict = {}

            for k, v in state_dict.items():
                # Adjust key names for loading pretrained weights
                if k not in current_keys:
                    k = k.replace("features", "encoder.features")
                    k = k.replace("classifier", "fc")
                    k = k.replace("ext.", "encoder.")

                new_dict[k] = v

            # Load the model's state dictionary
            self.load_state_dict(new_dict, strict=True)
  