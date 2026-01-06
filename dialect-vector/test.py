import torch

pt_path = "dialect-vector/phi3_medium_dialect_vectors.pt" 
obj = torch.load(pt_path, map_location="cpu")

print(type(obj))
print(obj.keys())

results = obj["results"]

print("results type:", type(results))
print("num layers saved:", len(results))
print("layer keys (head):", sorted(results.keys())[:5])
print("layer keys (tail):", sorted(results.keys())[-5:])

layer_id = 38
entry = results[layer_id]

print("entry type:", type(entry))
print("entry keys:", entry.keys())

v = entry["v_aae"]

print("vector type:", type(v))
print("vector shape:", v.shape)
print("vector dtype:", v.dtype)
print("vector norm:", float(v.norm()))

stats = entry["stats"]
for k, v in stats.items():
    print(f"{k}: {v}")

import torch

layer_id = 38
entry = obj["results"][layer_id]

v = entry["v_aae"]   # already unit-norm

torch.save(
    {
        "v_aae_unit": v,
        "layer_id": layer_id,
        "model_name": obj["model_name"],
        "stats": entry["stats"],
    },
    "phi3_medium_layer38_vector.pt"
)

print("Saved phi3_medium_layer38_vector.pt")
print("norm:", float(v.norm()))