import argparse
import torch
import os
import pickle

def save_split(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    tensors_dir = os.path.join(output_dir, "tensors")
    os.makedirs(tensors_dir, exist_ok=True)

    checkpoint = torch.load(input_path, map_location="cpu")

    metadata = {}

    for key, value in checkpoint.items():
        safe_key = key.replace("/", "_")
        if torch.is_tensor(value):
            tensor_path = os.path.join(tensors_dir, f"{safe_key}.pt")
            torch.save(value, tensor_path)
        else:
            metadata[key] = value

    meta_path = os.path.join(output_dir, "metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Split complete. Output stored in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .pth file")
    parser.add_argument("--output_dir", required=True, help="Directory to save output parts")
    args = parser.parse_args()

    save_split(args.input, args.output_dir)
