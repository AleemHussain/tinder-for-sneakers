import open_clip
from PIL import Image
import torch
import numpy as np
import os
from annoy import AnnoyIndex
import matplotlib.pyplot as plt

# Load model and preprocessing
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model.eval()

# Load query image
query_path = "query/test_shoe6.jpg"
image = preprocess(Image.open(query_path).convert("RGB")).unsqueeze(0)
with torch.no_grad():
    query_embedding = model.encode_image(image).cpu().numpy()[0]

# Load index and filename mapping
index = AnnoyIndex(512, "angular")
index.load("shoe_index.ann")
id_to_filename = np.load("id_to_filename.npy", allow_pickle=True)

# Search for similar images
top_k = 5
indices = index.get_nns_by_vector(query_embedding, top_k)

# Display results
fig, axs = plt.subplots(1, top_k, figsize=(15, 5))
for i, idx in enumerate(indices):
    img_path = os.path.join("shoes", id_to_filename[idx])
    img = Image.open(img_path)
    axs[i].imshow(img)
    axs[i].axis("off")
    axs[i].set_title(f"Result {i+1}")
plt.suptitle("Most Similar Sneakers", fontsize=16)
plt.tight_layout()
plt.show()
