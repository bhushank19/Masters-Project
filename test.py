import torch

# Check if MPS (Apple Silicon) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Example: Moving a tensor to the MPS device
x = torch.tensor([1, 2, 3], device=device)
print(x)
