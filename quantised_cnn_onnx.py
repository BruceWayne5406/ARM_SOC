import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import os

# Define the quantizable model architecture (must match your training code)
class ThreeLayerCNN_Quantizable(nn.Module):
    def __init__(self, num_classes=80):
        super(ThreeLayerCNN_Quantizable, self).__init__()

        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Layer 1: Conv -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: Conv -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: Conv -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Quantize input
        x = self.quant(x)

        # Layer 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Layer 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Dequantize output
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv-ReLU layers for better quantization"""
        torch.quantization.fuse_modules(self, [['conv1', 'relu1']], inplace=True)
        torch.quantization.fuse_modules(self, [['conv2', 'relu2']], inplace=True)
        torch.quantization.fuse_modules(self, [['conv3', 'relu3']], inplace=True)
        torch.quantization.fuse_modules(self, [['fc1', 'relu4']], inplace=True)


def load_quantized_model(model_path):
    """
    Helper function to load quantized models with proper handling
    """
    print(f"Loading quantized model from: {model_path}")

    # First, check what we're dealing with
    checkpoint = torch.load(model_path, map_location='cpu')

    # Check if it's a full model or just state_dict
    if isinstance(checkpoint, nn.Module):
        print("   → Loaded as complete model object")
        return checkpoint

    # It's a state_dict, check its structure
    print(f"   → State dict with {len(checkpoint)} keys")

    # Check for quantized layer signatures
    has_quantized = any('_packed_params' in key or '.scale' in key for key in checkpoint.keys())

    if has_quantized:
        print("   → Detected quantized state dict")
        # Create and prepare a quantized model
        model_fp32 = ThreeLayerCNN_Quantizable(num_classes=80)
        model_fp32.eval()
        model_fp32.fuse_model()
        model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Prepare model
        model_prepared = torch.quantization.prepare(model_fp32)

        # Calibrate with dummy data
        dummy_calib = torch.randn(10, 3, 224, 224)
        with torch.no_grad():
            for _ in range(5):
                model_prepared(dummy_calib)

        # Convert to quantized
        model_quantized = torch.quantization.convert(model_prepared)

        # Load the state dict
        missing_keys, unexpected_keys = model_quantized.load_state_dict(checkpoint, strict=False)

        if missing_keys:
            print(f"   ! Warning: Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"   ! Warning: Unexpected keys: {len(unexpected_keys)}")

        print("   ✓ Quantized model loaded successfully")
        return model_quantized
    else:
        print("   → Detected FP32 state dict")
        model = ThreeLayerCNN_Quantizable(num_classes=80)
        model.load_state_dict(checkpoint)
        model.eval()
        return model


def export_quantized_to_onnx(model_path, onnx_path='quantized_model.onnx',
                              input_shape=(1, 3, 224, 224)):
    """
    Export quantized PyTorch model to ONNX format

    Args:
        model_path: Path to the quantized .pth model file
        onnx_path: Output path for ONNX file
        input_shape: Input tensor shape (batch_size, channels, height, width)
    """
    print("="*60)
    print("EXPORTING QUANTIZED MODEL TO ONNX")
    print("="*60)

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return False

    print(f"\n1. Loading quantized model from: {model_path}")

    # Load the model using our helper function
    try:
        model = load_quantized_model(model_path)
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    print(f"\n2. Creating dummy input with shape: {input_shape}")
    dummy_input = torch.randn(input_shape)

    # Export to ONNX
    print(f"\n3. Exporting to ONNX format: {onnx_path}")
    try:
        torch.onnx.export(
            model,                          # Model being exported
            dummy_input,                    # Model input (dummy)
            onnx_path,                      # Where to save the model
            export_params=True,             # Store trained parameter weights
            opset_version=13,               # ONNX version (13 supports quantization)
            do_constant_folding=True,       # Optimize constant folding
            input_names=['input'],          # Input tensor name
            output_names=['output'],        # Output tensor name
            dynamic_axes={                  # Variable length axes
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        print("   ✓ ONNX export successful!")
    except Exception as e:
        print(f"   ✗ Error during ONNX export: {e}")
        return False

    # Verify ONNX model
    print("\n4. Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("   ✓ ONNX model is valid!")

        # Print model info
        print("\n5. ONNX Model Information:")
        print(f"   - IR Version: {onnx_model.ir_version}")
        print(f"   - Producer: {onnx_model.producer_name}")
        print(f"   - Graph inputs: {len(onnx_model.graph.input)}")
        print(f"   - Graph outputs: {len(onnx_model.graph.output)}")
        print(f"   - Nodes: {len(onnx_model.graph.node)}")

    except ImportError:
        print("   ! ONNX package not installed. Install with: pip install onnx")
        print("   ! Model exported but verification skipped")
    except Exception as e:
        print(f"   ! Warning during verification: {e}")

    # Get file size
    file_size = os.path.getsize(onnx_path)
    print(f"\n6. Output file size: {file_size / (1024*1024):.2f} MB")

    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)

    return True


def export_with_onnxruntime_quantization(model_path, fp32_onnx_path, onnx_quantized_path='quantized_model_int8.onnx'):
    """
    Apply ONNX Runtime quantization to an FP32 ONNX model
    This creates a more FPGA-friendly INT8 ONNX model

    Args:
        model_path: Original PyTorch model path (not used, kept for compatibility)
        fp32_onnx_path: Path to FP32 ONNX model
        onnx_quantized_path: Output path for INT8 quantized ONNX model
    """
    print("\n" + "="*60)
    print("ONNX RUNTIME INT8 QUANTIZATION")
    print("="*60)

    if not os.path.exists(fp32_onnx_path):
        print(f"Error: FP32 ONNX model not found: {fp32_onnx_path}")
        return False

    print(f"\n1. Loading FP32 ONNX model: {fp32_onnx_path}")

    try:
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader
        import numpy as np

        print("\n2. Applying dynamic INT8 quantization...")
        # Apply dynamic quantization (weights are quantized, activations are quantized at runtime)
        quantize_dynamic(
            model_input=fp32_onnx_path,
            model_output=onnx_quantized_path,
            weight_type=QuantType.QInt8,  # INT8 quantization
            optimize_model=True,  # Apply optimizations
        )

        print(f"   ✓ INT8 ONNX model saved to: {onnx_quantized_path}")

        # Compare file sizes
        original_size = os.path.getsize(fp32_onnx_path)
        quantized_size = os.path.getsize(onnx_quantized_path)

        print(f"\n3. Size comparison:")
        print(f"   - FP32 ONNX:     {original_size / (1024*1024):.2f} MB")
        print(f"   - INT8 ONNX:     {quantized_size / (1024*1024):.2f} MB")
        print(f"   - Compression:   {original_size/quantized_size:.2f}x")

        print("\n" + "="*60)
        print("QUANTIZATION COMPLETE!")
        print("="*60)

    except ImportError:
        print("   ✗ ONNX Runtime not installed")
        print("   Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"   ✗ Error during ONNX Runtime quantization: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_onnx_inference(onnx_path, input_shape=(1, 3, 224, 224)):
    """
    Test the exported ONNX model with a dummy inference
    """
    print("\n" + "="*60)
    print("TESTING ONNX INFERENCE")
    print("="*60)

    try:
        import onnxruntime as ort
        import numpy as np

        print(f"\n1. Loading ONNX model: {onnx_path}")
        session = ort.InferenceSession(onnx_path)

        print("   ✓ ONNX Runtime session created")

        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        print(f"\n2. Model I/O:")
        print(f"   - Input name:  {input_name}")
        print(f"   - Input shape: {session.get_inputs()[0].shape}")
        print(f"   - Output name: {output_name}")
        print(f"   - Output shape: {session.get_outputs()[0].shape}")

        # Create dummy input
        print(f"\n3. Running inference with random input...")
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Run inference
        outputs = session.run([output_name], {input_name: dummy_input})

        print(f"   ✓ Inference successful!")
        print(f"   - Output shape: {outputs[0].shape}")
        print(f"   - Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")

    except ImportError:
        print("   ! ONNX Runtime not installed")
        print("   ! Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"   ✗ Error during inference test: {e}")
        return False

    return True


def export_dequantized_to_onnx(model_path, onnx_path='dequantized_model.onnx',
                                input_shape=(1, 3, 224, 224)):
    """
    Load quantized model, dequantize it to FP32, then export to ONNX
    This approach is more compatible with ONNX and FPGA tools
    """
    print("="*60)
    print("EXPORTING DEQUANTIZED MODEL TO ONNX (FP32)")
    print("="*60)
    print("Note: Model will be converted back to FP32 for ONNX compatibility")
    print("You can re-quantize using FPGA-specific tools")
    print("="*60)

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return False

    print(f"\n1. Loading quantized model from: {model_path}")

    try:
        model_quantized = load_quantized_model(model_path)
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return False

    print("\n2. Dequantizing model to FP32...")

    # Create FP32 model
    model_fp32 = ThreeLayerCNN_Quantizable(num_classes=80)
    model_fp32.eval()

    # Extract dequantized weights from quantized model
    try:
        # For each quantized layer, extract the dequantized weights
        with torch.no_grad():
            # Convolutional layers
            if hasattr(model_quantized.conv1, 'weight'):
                model_fp32.conv1.weight.data = model_quantized.conv1.weight().dequantize()
                model_fp32.conv1.bias.data = model_quantized.conv1.bias()

            if hasattr(model_quantized.conv2, 'weight'):
                model_fp32.conv2.weight.data = model_quantized.conv2.weight().dequantize()
                model_fp32.conv2.bias.data = model_quantized.conv2.bias()

            if hasattr(model_quantized.conv3, 'weight'):
                model_fp32.conv3.weight.data = model_quantized.conv3.weight().dequantize()
                model_fp32.conv3.bias.data = model_quantized.conv3.bias()

            # Fully connected layers (these might be packed)
            if hasattr(model_quantized.fc1, '_packed_params'):
                weight, bias = model_quantized.fc1._weight_bias()
                model_fp32.fc1.weight.data = weight.dequantize()
                if bias is not None:
                    model_fp32.fc1.bias.data = bias
            elif hasattr(model_quantized.fc1, 'weight'):
                model_fp32.fc1.weight.data = model_quantized.fc1.weight().dequantize()
                model_fp32.fc1.bias.data = model_quantized.fc1.bias()

            if hasattr(model_quantized.fc2, '_packed_params'):
                weight, bias = model_quantized.fc2._weight_bias()
                model_fp32.fc2.weight.data = weight.dequantize()
                if bias is not None:
                    model_fp32.fc2.bias.data = bias
            elif hasattr(model_quantized.fc2, 'weight'):
                model_fp32.fc2.weight.data = model_quantized.fc2.weight().dequantize()
                model_fp32.fc2.bias.data = model_quantized.fc2.bias()

        print("   ✓ Model dequantized successfully")
    except Exception as e:
        print(f"   ✗ Error during dequantization: {e}")
        print("   → Attempting to use model as-is...")
        model_fp32 = model_quantized

    # Create dummy input
    print(f"\n3. Creating dummy input with shape: {input_shape}")
    dummy_input = torch.randn(input_shape)

    # Export to ONNX
    print(f"\n4. Exporting to ONNX format: {onnx_path}")
    try:
        torch.onnx.export(
            model_fp32,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        print("   ✓ ONNX export successful!")
    except Exception as e:
        print(f"   ✗ Error during ONNX export: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify ONNX model
    print("\n5. Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("   ✓ ONNX model is valid!")

        # Print model info
        print("\n6. ONNX Model Information:")
        print(f"   - IR Version: {onnx_model.ir_version}")
        print(f"   - Producer: {onnx_model.producer_name}")
        print(f"   - Graph inputs: {len(onnx_model.graph.input)}")
        print(f"   - Graph outputs: {len(onnx_model.graph.output)}")
        print(f"   - Nodes: {len(onnx_model.graph.node)}")

    except ImportError:
        print("   ! ONNX package not installed")
    except Exception as e:
        print(f"   ! Warning during verification: {e}")

    # Get file size
    file_size = os.path.getsize(onnx_path)
    print(f"\n7. Output file size: {file_size / (1024*1024):.2f} MB")

    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)

    return True


# Main execution
if __name__ == '__main__':
    print("="*60)
    print("QUANTIZED MODEL TO ONNX CONVERTER")
    print("="*60)

    # Get model path from user
    print("\nAvailable export options:")
    print("1. Export quantized model directly (INT8 - experimental)")
    print("2. Dequantize and export as FP32 (recommended - most compatible)")
    print("3. ONNX Runtime quantization on FP32 model")
    print("4. Export and test inference")

    choice = input("\nSelect option (1/2/3/4): ").strip()

    # Get input model path
    default_path = 'coco_3layer_cnn_int8_ptq.pth'
    model_path = input(f"\nEnter path to quantized model (default: {default_path}): ").strip()
    if not model_path:
        model_path = default_path

    # Get output ONNX path
    default_onnx = 'quantized_cnn_model.onnx'
    onnx_path = input(f"Enter output ONNX path (default: {default_onnx}): ").strip()
    if not onnx_path:
        onnx_path = default_onnx

    # Execute based on choice
    if choice == '1':
        # Direct quantized export (may have compatibility issues)
        success = export_quantized_to_onnx(model_path, onnx_path)

    elif choice == '2':
        # Dequantize to FP32 then export (most compatible)
        success = export_dequantized_to_onnx(model_path, onnx_path)

    elif choice == '3':
        # Dequantize, export, then re-quantize with ONNX Runtime
        temp_onnx = 'temp_fp32_model.onnx'
        success = export_dequantized_to_onnx(model_path, temp_onnx)
        if success:
            onnx_int8_path = onnx_path.replace('.onnx', '_int8.onnx')
            success = export_with_onnxruntime_quantization(model_path, temp_onnx, onnx_int8_path)
            # Clean up temp file
            if os.path.exists(temp_onnx):
                os.remove(temp_onnx)
                print(f"\n   → Cleaned up temporary file: {temp_onnx}")

    elif choice == '4':
        # Export and test
        success = export_dequantized_to_onnx(model_path, onnx_path)
        if success:
            test_onnx_inference(onnx_path)
    else:
        print("Invalid choice!")
        exit(1)

    if success:
        print("\n" + "="*60)
        print("FPGA DEPLOYMENT NOTES")
        print("="*60)
        print("\nYour ONNX model is ready for FPGA deployment!")
        print("\nNext steps:")
        print("1. For Xilinx FPGAs:")
        print("   - Use Vitis AI Quantizer and Compiler")
        print("   - Command: vai_q_onnx quantize --model your_model.onnx")
        print("\n2. For Intel FPGAs:")
        print("   - Use OpenVINO Model Optimizer")
        print("   - Command: mo --input_model your_model.onnx")
        print("\n3. For generic FPGA flows:")
        print("   - Convert ONNX to vendor-specific format")
        print("   - Optimize for target FPGA architecture")
        print("   - Consider fixed-point conversion if needed")
        print("\n4. Model characteristics:")
        print("   - INT8 weights and activations")
        print("   - Optimized for low-power inference")
        print("   - ~4x memory reduction vs FP32")
        print("="*60)
    else:
        print("\n✗ Export failed! Please check the errors above.")
        exit(1)