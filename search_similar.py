import open_clip
from PIL import Image
import torch
import numpy as np
import os
from annoy import AnnoyIndex
import matplotlib.pyplot as plt

# Config
query_path = "query/test_shoe1.jpg"  # Set to None if no image is provided
text_prompt = "red sole white shoelaces"  # Set to "" if no text is provided
alpha = 0.5  # Weight for image vs. text. Ignored if only one modality is used
top_k = 5
image_dir = "shoes"

# Load model
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)

# Initialize empty embedding variables
image_embedding = None
text_embedding = None

# Compute image embedding if image is given
if query_path and os.path.exists(query_path):
    image = preprocess(Image.open(query_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = model.encode_image(image)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

# Compute text embedding if text prompt is given
if text_prompt.strip():
    text_tokens = tokenizer([text_prompt]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

# Combine or select embeddings
if image_embedding is not None and text_embedding is not None:
    combined = alpha * image_embedding + (1 - alpha) * text_embedding
elif image_embedding is not None:
    combined = image_embedding
elif text_embedding is not None:
    combined = text_embedding
else:
    raise ValueError("Please provide at least an image or a text prompt.")

# Normalize and prepare query vector
combined /= combined.norm(dim=-1)
query_vector = combined.squeeze(0).cpu().numpy()

# Load index and filenames
index = AnnoyIndex(512, "angular")
index.load("shoe_index.ann")
id_to_filename = np.load("id_to_filename.npy", allow_pickle=True)

# Search and show results
indices = index.get_nns_by_vector(query_vector, top_k)
fig, axs = plt.subplots(1, top_k, figsize=(15, 5))
for i, idx in enumerate(indices):
    img_path = os.path.join(image_dir, id_to_filename[idx])
    img = Image.open(img_path)
    axs[i].imshow(img)
    axs[i].axis("off")
    axs[i].set_title(f"Result {i+1}")
query_type = "Image + Text" if image_embedding is not None and text_embedding is not None else "Text" if image_embedding is None else "Image"
plt.suptitle(f"Search Type: {query_type}", fontsize=14)
plt.tight_layout()
plt.show()
