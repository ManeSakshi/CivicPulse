import pandas as pd
import torch
from transformers import pipeline

print("[OK] Pandas version:", pd.__version__)
print("[OK] Torch version:", torch.__version__)

classifier = pipeline("sentiment-analysis")
print(classifier("Traffic in Sangli is horrible today"))
