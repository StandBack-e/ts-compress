import torch
import time

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    try:
        # 创建一个足够大的张量，确保它能占用显著的显存
        print("\nAllocating a large tensor on GPU...")
        large_tensor = torch.randn(10000, 10000, device=device)
        print("Tensor allocated successfully.")
        print(f"Tensor device: {large_tensor.device}")
        print("Starting a 30-second computation loop. PLEASE RUN nvidia-smi NOW in another terminal.")
        
        # 持续进行计算，确保GPU核心被使用
        start_time = time.time()
        while time.time() - start_time < 30:
            # 执行一个耗时的操作
            torch.matmul(large_tensor, large_tensor)
        
        print("\nTest finished successfully!")

    except Exception as e:
        print(f"\nAn error occurred during the GPU test: {e}")

else:
    print("\nCUDA is not available on this system.")