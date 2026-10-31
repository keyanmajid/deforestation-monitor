import segmentation_models_pytorch as smp


def build_model(in_channels=5, encoder="resnet34", pretrained=True):
    """U-Net with a ResNet34 encoder. Outputs raw logits (1 channel)."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,   # smp adapts the first conv layer to 5 bands
        classes=1,
        activation=None,
    )