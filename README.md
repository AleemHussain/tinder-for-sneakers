# Tinder-for-Sneakers (FashionCLIP Version)

A prototype for a swipe-based sneaker recommendation system. Inspired by the user experience of TikTok and Tinder, users can explore sneaker options quickly by swiping through images — finding the right pair in just a few interactions.

This project uses **FashionCLIP** to convert sneaker images and text descriptions into embeddings, and **Annoy** for fast similarity search in the vector space.  
It also integrates a **SQLite database** to store sneaker metadata (filename, brand, color, description) and allows enhanced display during search.

---

## 📄 Documentation

You can find the original conceptual summary of the project (in German) here:

📘 [Bildbasiertes Empfehlungssystem mit FashionCLIP (PDF)](Bildbasiertes_Empfehlungssystem_mit_FashionCLIP.pdf)

---

## 📁 Dataset Used

We use the following sneaker dataset from Kaggle:
**[Nike, Adidas & Converse Image Dataset](https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged)**

It includes sneaker images from well-known brands like Nike, Adidas, and Converse.

---

## ⚙️ Installation

```bash
git clone https://github.com/AleemHussain/tinder-for-sneakers.git
cd tinder-for-sneakers
```

```bash
# (Optional) Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

```bash
# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 1. Generate image embeddings and build search index

```bash
python build_index.py
```

This script will:

* Extract image embeddings using FashionCLIP
* Store them in an Annoy index (`shoe_index.ann`)
* Save a filename mapping (`id_to_filename.npy`)
* Store sneaker metadata in a local SQLite database (`sneakers.db`)

### 2. Search by image, text, or both

Configure and run:

```bash
python search_similar.py
```

You can set:

* `query_path`: path to an input sneaker image (e.g. `query/test_shoe1.jpg`)
* `text_prompt`: a natural language query (e.g. `"white sneaker with red sole"`)
* `alpha`: weight between image and text (e.g. `0.5` = 50/50)

The script will:

* Compute image/text embeddings
* Search nearest matches using Annoy
* Load metadata from the database
* Display results with image and description using `matplotlib`

---

## 🧠 Technologies Used

* [HuggingFace Transformers](https://huggingface.co) — `patrickjohncyh/fashion-clip`
* [Annoy (Spotify)](https://github.com/spotify/annoy) — Fast vector search
* [SQLite3](https://www.sqlite.org/index.html) — Lightweight metadata database
* Python 3.10+

---

## 📂 Project Structure

```
tinder-for-sneakers/
├── shoes/                   # Sneaker image dataset
├── query/                   # Example input queries
│   └── test_shoe1.jpg
│
├── build_index.py           # Builds Annoy index + DB
├── search_similar.py        # Performs similarity search
├── db.py                    # SQLite helper functions
│
├── shoe_index.ann           # Saved Annoy index (auto-generated)
├── id_to_filename.npy       # Filename mapping (auto-generated)
├── sneakers.db              # SQLite DB with sneaker metadata
├── requirements.txt
├── README.md
```

---

## 📄 License

For research and prototyping purposes only.
Data usage subject to the license of the referenced Kaggle dataset.

---
