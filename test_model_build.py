"""
Quick test script to verify the model builds correctly with the new architecture
"""
import torch
import sys
sys.path.append('./')

from models.yolo import Model

def test_model():
    print("=" * 60)
    print("Testing LEAF-YOLO-Edge-Drone Model Build")
    print("=" * 60)
    
    # Test model configuration
    cfg = 'cfg/LEAF-YOLO/leaf-edge-drone.yaml'
    
    try:
        # Create model
        print(f"\n1. Loading model from: {cfg}")
        model = Model(cfg, ch=3, nc=80)  # Use nc=80 as specified in YAML
        print("✓ Model created successfully!")
        
        # Test forward pass
        print("\n2. Testing forward pass with dummy input...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")
        
        model = model.to(device)
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful!")
        print(f"   Output shapes: {[o.shape for o in output]}")
        
        # Print model info
        print("\n3. Model Information:")
        from utils.torch_utils import model_info
        model_info(model, verbose=True)
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour model is ready for training!")
        print("\nNext steps:")
        print("1. Verify dataset paths in data/visdrone.yaml")
        print("2. Run training with:")
        print("   python train.py --workers 8 --device 0 --batch-size 16 \\")
        print("       --epochs 300 --data data/visdrone.yaml --img 640 640 \\")
        print("       --cfg cfg/LEAF-YOLO/leaf-edge-drone.yaml --weights '' \\")
        print("       --hyp data/hyp.scratch.visdrone.yaml --name leaf-edge-drone")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_model()
    sys.exit(0 if success else 1)
