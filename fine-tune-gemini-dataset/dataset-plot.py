import os
import random
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
from items import Item
from loaders import ItemLoader
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import pickle

load_dotenv(override=True)
token = os.getenv('HF_TOKEN')
google_api_key = os.getenv('GOOGLE_GENERATIVE_AI_API_KEY')
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ['HF_TOKEN'] = token
login(token)

# dataset_names = [
#     "Automotive",
#     "Electronics",
#     "Office_Products",
#     "Tools_and_Home_Improvement",
#     "Cell_Phones_and_Accessories",
#     "Toys_and_Games",
#     "Appliances",
#     "Musical_Instruments",
# ]

# items = []
# for dataset_name in dataset_names:
#     loader = ItemLoader(dataset_name)
#     items.extend(loader.load())

# slots = defaultdict(list)
# for item in items:
#     slots[round(item.price)].append(item)

# np.random.seed(42)
# random.seed(42)
# sample = []
# for i in range(1, 1000):
#     slot = slots[i]
#     if i>=240:
#         sample.extend(slot)
#     elif len(slot) <= 1200:
#         sample.extend(slot)
#     else:
#         weights = np.array([1 if item.category=='Automotive' else 5 for item in slot])
#         weights = weights / np.sum(weights)
#         selected_indices = np.random.choice(len(slot), size=1200, replace=False, p=weights)
#         selected = [slot[i] for i in selected_indices]
#         sample.extend(selected)

# print(f"There are {len(sample):,} items in the sample")

# random.seed(42)
# random.shuffle(sample)
# train_data = int(len(sample) * 0.8)
# train = sample[:train_data]
# test = sample[train_data:]
# print(f"Divided into a training set of {len(train):,} items and test set of {len(test):,} items")

# train_prompts = [item.prompt for item in train]
# train_prices = [item.price for item in train]
# test_prompts = [item.test_prompt() for item in test]
# test_prices = [item.price for item in test]

# train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
# test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
# dataset = DatasetDict({
#     "train": train_dataset,
#     "test": test_dataset
# })

# with open('train.pkl', 'wb') as file:
#     pickle.dump(train, file)

# with open('test.pkl', 'wb') as file:
#     pickle.dump(test, file)

train_data = []
test_data = []
with open('train.pkl', 'rb') as file:
    train_data = pickle.load(file, encoding="utf-8")

with open('test.pkl', 'rb') as file:
    test_data = pickle.load(file, encoding="utf-8")

print(f"loaded training set of {len(train_data):,} items and test set of {len(test_data):,} items")

print(train_data[0].prompt)
print(test_data[0].prompt)

prices = [float(item.price) for item in test_data[:1000]]
plt.figure(figsize=(15, 6))
plt.title(f"Avg {sum(prices)/len(prices):.2f} and highest {max(prices):,.2f}\n")
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.hist(prices, rwidth=0.7, color="darkblue", bins=range(0, 1000, 10))
plt.show()