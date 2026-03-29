# inspect_dataset.py — Check IndicVoices column structure

from datasets import load_dataset

ds = load_dataset(
    "ai4bharat/IndicVoices",
    "hindi",
    split="train",
    streaming=True,
)

# Print first item fully
for i, item in enumerate(ds):
    print(f"All keys: {list(item.keys())}")
    for k, v in item.items():
        print(f"  {k}: type={type(v).__name__}, value={str(v)[:120]}")
    break