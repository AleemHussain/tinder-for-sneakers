# Tinder-for-Sneakers (OpenCLIP Version)

A prototype for a swipe-based sneaker recommendation system. Inspired by TikTok and Tinder, users can explore sneaker options quickly by swiping through images — finding the right pair in just a few interactions.

This project uses **OpenCLIP** to embed sneaker images and **Annoy** to search for similar styles in a high-dimensional vector space. The sneaker images come from a public Kaggle dataset.

---

## 📁 Dataset Used

We use this sneaker dataset from Kaggle:
👉 **[Nike, Adidas &amp; Converse Image Dataset](https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged)**

The images are manually extracted and placed into the `shoes/` folder.

---

## 📄 Documentation

Concept and design are documented in German in the following file:
📘 [Bildbasiertes Empfehlungssystem mit FashionCLIP (PDF)](Bildbasiertes_Empfehlungssystem_mit_FashionCLIP.pdf)

---

## 📂 Folder Structure

```
TFS/
├── shoes/                   # Sneaker database (from Kaggle dataset)
├── query/                   # Image to search with
│   └── test_shoe.jpg        # Already given some images to test with
│
├── build_index.py           # Embeds all shoes and builds Annoy index
├── search_similar.py        # Finds similar shoes from query image
│
├── shoe_index.ann           # Generated Annoy index (auto)
├── id_to_filename.npy       # Mapping of image filenames (auto)
├── .gitignore
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/AleemHussain/tinder-for-sneakers.git
cd tinder-for-sneakers
```

```bash
# (Optional) Create a Conda environment
conda create -n sneakerclip python=3.10 -y
conda activate sneakerclip
```

```bash
# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Use

### 1. Add your images

- All reference sneaker images go into `shoes/`
- Your query image goes into `query/` (e.g. `test_shoe.jpg`)

### 2. Build the index

```bash
python build_index.py
```

### 3. Search for similar sneakers

```bash
python search_similar.py
```

This will:

- Embed your query image
- Retrieve the 5 most similar sneakers using Annoy
- Display them side-by-side with `matplotlib`

---

## 🧠 Technologies Used

- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Annoy](https://github.com/spotify/annoy)
- [Pillow](https://pillow.readthedocs.io/)
- Python 3.10+

---

## 📄 License

This project is for research and prototyping only.
All image content is based on the [referenced Kaggle dataset](https://www.kaggle.com/datasets/die9origephit/nike-adidas-and-converse-imaged).
