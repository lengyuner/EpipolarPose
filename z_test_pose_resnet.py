import torch
import yaml
from easydict import EasyDict
from lib.models.pose3d_resnet import get_pose_net
import torch.nn.functional as F

def create_dummy_config():
    config = {
        'MODEL': {
            'EXTRA': {
                'NUM_LAYERS': 50,  # ResNet50
                'NUM_DECONV_LAYERS': 3,
                'NUM_DECONV_FILTERS': [256, 256, 256],
                'NUM_DECONV_KERNELS': [4, 4, 4],
                'FINAL_CONV_KERNEL': 1,
                'DECONV_WITH_BIAS': False
            },
            'INIT_WEIGHTS': False,
            'PRETRAINED': '',
            'VOLUME': False
        }
    }
    return EasyDict(config)

def test_pose_resnet():
    # Create configuration
    cfg = create_dummy_config()
    
    # Create model
    model = get_pose_net(cfg, is_train=True)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_length = 30
    orig_height = 229
    orig_width = 461
    target_height = 256
    target_width = 480
    channels = 3
    
    # Create random input tensor
    x = torch.randn(batch_size, seq_length, orig_height, orig_width, channels)
    print(f"Original input shape: {x.shape}")
    
    # Reshape and resize input
    x = x.permute(0, 1, 4, 2, 3)  # (B, T, H, W, C) -> (B, T, C, H, W)
    x = x.contiguous().view(-1, channels, orig_height, orig_width)  # Reshape to (B*T, C, H, W)
    x = F.interpolate(
        x, 
        size=(target_height, target_width), 
        mode='bilinear', 
        align_corners=True
    )
    x = x.view(batch_size, seq_length, channels, target_height, target_width)
    x = x.permute(0, 1, 3, 4, 2)  # Back to (B, T, H, W, C)
    x = x.contiguous()  # Make memory contiguous
    
    print(f"Resized input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print("\nShape verification:")
    print(f"Expected H: {target_height}, Got H: {output.shape[2]}")
    print(f"Expected W: {target_width}, Got W: {output.shape[3]}")
    
    # Verify output dimensions
    assert output.shape == (batch_size, seq_length, target_height, target_width), \
        f"Expected output shape {(batch_size, seq_length, target_height, target_width)}, but got {output.shape}"
    
    print("\nTest passed successfully!")
    
    return x, output

if __name__ == "__main__":
    try:
        x, output = test_pose_resnet()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")