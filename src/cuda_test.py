
# complete_test.py
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")

# 测试所有 PyG 组件
components = [
    ('torch_geometric', lambda: __import__('torch_geometric')),
    ('torch_scatter', lambda: __import__('torch_scatter')),
    ('torch_sparse', lambda: __import__('torch_sparse')),
    ('torch_cluster', lambda: __import__('torch_cluster')),
]

all_good = True
for name, import_func in components:
    try:
        module = import_func()
        print(f"✓ {name} loaded successfully")
        if hasattr(module, '__version__'):
            print(f"  Version: {module.__version__}")
    except Exception as e:
        print(f"✗ {name} failed to load: {e}")
        all_good = False

# 功能测试
if all_good:
    try:
        from torch_scatter import scatter
        # 简单的功能测试
        x = torch.randn(10, 16).cuda() if torch.cuda.is_available() else torch.randn(10, 16)
        index = torch.randint(0, 4, (10,)).cuda() if torch.cuda.is_available() else torch.randint(0, 4, (10,))
        result = scatter(x, index, dim=0, reduce="sum")
        print("✓ Scatter operation works correctly")
        print("All components verified successfully!")
    except Exception as e:
        print(f"✗ Function test failed: {e}")
else:
    print("Some components failed to load. Please check installation.")