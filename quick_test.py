# from src.config import *
# from src.data.preprocessing import load_data
# from transformers import AutoTokenizer
#
# df = load_data(TRAIN_PATH)
# print(df.head())
#
# AutoTokenizer.from_pretrained(MODEL_NAME)

# from src.models.transformer import TransformerModel
# from src.config import *

# import torch

# model = TransformerModel(MODEL_NAME, NUM_CLASSES)

# input_ids = torch.randint(0, 100, (2, 128))
# attention_mask = torch.ones((2, 128))

# outputs = model(input_ids, attention_mask)

# print(outputs.shape)


from src.data.preprocessing import load_data

df = load_data("data/EXIST2023_training.json")
print(df.head())