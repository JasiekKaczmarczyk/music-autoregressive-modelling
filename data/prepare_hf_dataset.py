import glob
import os
import torch
from datasets import Dataset, Features, Value, Array2D
from tqdm import tqdm

if __name__ == "__main__":
    token = os.environ["HUGGINGFACE_TOKEN"]
    fp = glob.glob("quantized/**.pt")

    records = []

    for f in tqdm(fp):
        name = f.split("/")[-1].split(".")[0]
        embedding = torch.load(f)

        # shape [num_quantizers, len] -> [len, num_quantizers]
        embedding = embedding.squeeze()
        embedding = embedding.permute(1, 0)

        print(f"Embedding shape: {embedding.shape}")

        record = {
            "name": name,
            "embedding": embedding
        }

        records.append(record)

    features = Features(
        {
            "name": Value(dtype="string"),
            "embedding": Array2D(dtype="int16", shape=(None, 4)),
        }
    )

    dataset = Dataset.from_list(records, features=features)

    dataset.push_to_hub("JasiekKaczmarczyk/audio-quantized", token=token)

    
