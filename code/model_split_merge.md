# Model Split & Merge Toolkit

This toolkit lets you losslessly break a PyTorch `.pth` checkpoint into separate files (metadata + individual tensors) and later reconstruct the exact same checkpoint. No quantization, no pruning, no changes to the data.

---

## Features

* Split a `.pth` file into:
  * `metadata.pkl` for all non-tensor objects
  * One `.pt` file per tensor stored under `tensors/`
* Merge all pieces back into a single `.pth` file
* Perfectly reversible
* Safe for large models

---

## Installation

You only need PyTorch:

```bash
pip install torch
```

---

## Usage

### 1. Split a model

Run the script to break a checkpoint into metadata and per-layer tensor files.

```bash
python save_split.py --input sentinel_model.pth --output_dir model_parts/
```

After running, the structure will look like:

```
model_parts/
    metadata.pkl
    tensors/
        layer1.weight.pt
        layer1.bias.pt
        ...
```

---

### 2. Merge the model back

Reassemble everything into a single `.pth` checkpoint.

```bash
python load_merge.py --input_dir model_parts --output sentinel_model.pth
```

This produces:

```
sentinel_model.pth
```

The output file is identical to the original.

---

## Scripts

### save_split.py

Splits the checkpoint into metadata and tensor files.

### load_merge.py

Reassembles the checkpoint from metadata and tensor files.

---

## Notes

* All operations are strictly lossless.
* Tensor filenames have `/` replaced with `_` to avoid filesystem issues.
* Useful when you need to store or compress model parts separately.
