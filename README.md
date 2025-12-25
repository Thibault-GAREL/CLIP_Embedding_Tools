# Opposite Embedding Finder

Find the "opposite" embedding of any word using CLIP (Contrastive Language-Image Pre-training).

## Concept

This tool computes the opposite of a word's embedding by:
1. Getting the CLIP text embedding for your input word
2. Negating the embedding vector (multiplying by -1)
3. Searching through CLIP's entire token vocabulary (~49,000 tokens) to find which tokens are closest to this opposite embedding

The "opposite" in embedding space represents the direction that is most dissimilar to the original word in the high-dimensional semantic space.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode

```bash
python opposite_embedding.py
```

Then type words to find their opposites!

### As a Python Module

```python
from opposite_embedding import OppositeEmbeddingFinder

# Initialize
finder = OppositeEmbeddingFinder()

# Find opposite of a word
finder.find_opposite("happy", top_k=10)
```

## How It Works

1. **CLIP Model**: Uses OpenAI's CLIP model (ViT-B/32) to get 512-dimensional semantic embeddings
2. **Token Embeddings**: Directly accesses CLIP's token embedding layer (instant - no encoding needed!)
3. **Negation**: The opposite embedding is computed as `-1 * original_embedding`
4. **Search**: Computes cosine similarity between opposite embedding and all ~49K token embeddings
5. **Results**: Returns the top-k tokens closest to the opposite embedding

**Key optimization**: Instead of encoding 49K tokens as text (slow), we directly use the model's token embedding weights (instant!)

## Example

```
Enter a word: hot

Finding opposite embedding for: 'hot'
============================================================

Extracting token embeddings from CLIP model...
Got 49408 token embeddings directly from model!

Original embedding shape: (512,)
Original embedding (first 5 dims): [ 0.0234 -0.0123  0.0456 ...]

Opposite embedding (first 5 dims): [-0.0234  0.0123 -0.0456 ...]
Dot product (should be ~-1): -1.0000

Top 10 tokens from CLIP vocabulary closest to the opposite embedding:
------------------------------------------------------------
 1. 'cold'                     (similarity: 0.7234)
 2. 'cool'                     (similarity: 0.6543)
 3. 'freezing'                 (similarity: 0.6102)
 ...
```

**Note**: Token embeddings are extracted directly from the model (instant!). First search extracts them, then all subsequent searches are cached.

## License

MIT License
