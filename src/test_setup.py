import pandas as pd
import torch
from transformers import pipeline

print("✅ Pandas version:", pd.__version__)
print("✅ Torch version:", torch.__version__)

classifier = pipeline("sentiment-analysis")
print(classifier("Traffic in Sangli is horrible today"))
