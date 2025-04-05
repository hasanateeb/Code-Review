from datasets import load_dataset
import os

# Define dataset name and subset
dataset_name = "code_search_net"
subset = "python"
save_path = "./data"

# Load dataset (trust remote code)
print("📦 Loading dataset... This may take some time.")
ds = load_dataset(dataset_name, subset, trust_remote_code=True)

# Ensure save folder exists
os.makedirs(save_path, exist_ok=True)

# Limit number of samples (optional: speeds up test/dev)
sample_size = 2000  # Change this to a higher number if needed
train_sample = ds["train"].select(range(min(sample_size, len(ds["train"]))))
test_sample = ds["test"].select(range(min(sample_size, len(ds["test"]))))

# Save dataset to CSV
train_csv = os.path.join(save_path, "code_search_net_train.csv")
test_csv = os.path.join(save_path, "code_search_net_test.csv")

train_sample.to_csv(train_csv, index=False)
test_sample.to_csv(test_csv, index=False)

print(f"✅ Dataset saved to: {save_path}")
