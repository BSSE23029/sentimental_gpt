import argparse
import torch
import os
import pickle

def load_merge(input_dir, output_path):
    tensors_dir = os.path.join(input_dir, "tensors")
    meta_path = os.path.join(input_dir, "metadata.pkl")

    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    assembled = dict(metadata)

    for fname in os.listdir(tensors_dir):
        if fname.endswith(".pt"):
            key = fname[:-3]  # strip .pt
            tensor_path = os.path.join(tensors_dir, fname)
            assembled[key] = torch.load(tensor_path, map_location="cpu")

    torch.save(assembled, output_path)
    print(f"Reconstructed checkpoint saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with tensors/ and metadata.pkl")
    parser.add_argument("--output", required=True, help="Output .pth file path")
    args = parser.parse_args()

    load_merge(args.input_dir, args.output)
