def test_pose_resnet():
    # ... (previous code remains the same) ...
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output tensor size: {output.numel()}")
    print(f"Expected size: {batch_size * seq_length * height * width}")
    
    # Verify output dimensions
    assert output.shape == (batch_size, seq_length, height, width), \
        f"Expected output shape {(batch_size, seq_length, height, width)}, but got {output.shape}"
    
    # Add tensor statistics
    print(f"\nOutput statistics:")
    print(f"Min value: {output.min().item():.4f}")
    print(f"Max value: {output.max().item():.4f}")
    print(f"Mean value: {output.mean().item():.4f}")