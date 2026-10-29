import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

print("Knee MRI AI Project")
print("-------------------")

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("PyTorch version:", torch.__version__)

print("GPU available:", torch.cuda.is_available())