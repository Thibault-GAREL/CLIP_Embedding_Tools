# 🔤 CLIP Embedding Tools

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-red.svg)
![Torch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-red.svg)

![License](https://img.shields.io/badge/license-MIT-green.svg)  
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)

<p align="center">
  <img src="img/Logo-Embedding_Tools.png" alt="Logo Embedding Tools">
</p>

## 📝 Project Description
Three powerful tools for exploring semantic word relationships using OpenAI's CLIP model. Perform vector operations like "king - man + woman ≈ queen" or find semantic opposites by negating embeddings. Features a cached mode with 49K token vocabulary for instant loading! 🧠✨

---

## 🚀 Features
🔍 **Tool 1**: Find semantic opposites by negating embeddings

➕ **Tool 2**: Vector arithmetic with curated 500-word vocabulary

⚡ **Tool 3**: Full 49K token vocabulary with disk caching (Recommended!)

🎯 Direct access to CLIP's token embeddings (no slow encoding!)

📊 Interactive CLI with complex expression support

---

## ⚙️ How it works

🤖 Uses OpenAI's CLIP (ViT-B/32) to generate 512-dimensional semantic embeddings.

🔢 Performs vector operations: addition, subtraction, negation on word embeddings.

🎯 Directly accesses token embedding layer (instant – no text encoding needed!).

📈 Computes cosine similarity to find nearest neighbors in semantic space.

💾 **Tool 3**: One-time 2-3 min computation, then instant loading from cache!

## 🗺️ Examples

### Example 1: Opposite Embedding
```
Enter a word: hot

Top 10 tokens closest to the opposite embedding:
------------------------------------------------------------
 1. 'cold'        (similarity: 0.7234)
 2. 'cool'        (similarity: 0.6543)
 3. 'freezing'    (similarity: 0.6102)
```

### Example 2: Vector Arithmetic
```
Enter command: calc king - man + woman

Top 10 tokens closest to the result:
------------------------------------------------------------
 1. 'queen'       (similarity: 0.8234)
 2. 'princess'    (similarity: 0.7543)
 3. 'monarch'     (similarity: 0.7123)
```

### Example 3: Performance Comparison

| Tool | Vocabulary | First Run | Subsequent | Quality |
|------|-----------|-----------|------------|---------|
| Tool 2 (limited) | 500 words | 10-20 sec | Instant | Good |
| **Tool 3 (cached)** | **49K tokens** | **2-3 min** | **~1 sec** | **Excellent** |

⏱️ First run computes all embeddings (one-time setup). All future runs load instantly from `clip_token_embeddings.npz`!

---

## 📂 Repository structure
```bash
├── img/                           # Images for the README
│   └── Logo-Embedding_tools.png
│
├── opposite_embedding.py          # Tool 1: Find semantic opposites
├── embedding_arithmetic.py        # Tool 2: Vector arithmetic (500 words)
├── embedding_arithmetic_cached.py # Tool 3: Full vocabulary with caching (Recommended!)
│
├── clip_token_embeddings.npz      # Cached embeddings (generated on first run)
│
├── requirements.txt               # Python dependencies
├── LICENSE                        # Project license
├── README.md                      # Main documentation
```

---

## 💻 Run it on Your PC
Clone the repository and install dependencies:
```bash
git clone https://github.com/Thibault-GAREL/clip-embedding-tools.git
cd clip-embedding-tools

python -m venv .venv  # if you don't have a virtual environment
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt

# Run Tool 3 (Recommended - Full vocabulary with caching)
python embedding_arithmetic_cached.py

# Or try other tools:
python opposite_embedding.py          # Tool 1
python embedding_arithmetic.py        # Tool 2
```

### Interactive Commands
Once running, try these commands:
```bash
add man kingdom              # Vector addition
sub king man                 # Vector subtraction
calc king - man + woman      # Complex expressions
```

---

## 📖 Inspiration / Sources
I code it with Claude Code 😆 !

Code created by me 😎, Thibault GAREL - [Github](https://github.com/Thibault-GAREL)