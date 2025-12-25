#!/usr/bin/env python3
"""
Opposite Embedding Finder using CLIP
Finds the opposite embedding of a given token/word by searching CLIP's token vocabulary.
"""

import torch
import clip
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class OppositeEmbeddingFinder:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize the CLIP model for finding opposite embeddings.

        Args:
            model_name: CLIP model to use (default: "ViT-B/32")
            device: Device to run on (cuda/cpu), auto-detected if None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        self.token_embeddings = None
        print("Model loaded! Ready to find opposite embeddings.")

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get CLIP embedding for a text input.

        Args:
            text: Input text/word

        Returns:
            Normalized embedding vector
        """
        text_input = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

    def get_opposite_embedding(self, text: str) -> np.ndarray:
        """
        Get the opposite embedding by negating the original embedding.

        Args:
            text: Input text/word

        Returns:
            Opposite (negated) embedding vector
        """
        embedding = self.get_text_embedding(text)
        opposite = -embedding
        # Normalize
        opposite = opposite / np.linalg.norm(opposite)
        return opposite

    def get_token_embeddings(self):
        """
        Get the token embedding weights directly from CLIP's model.
        This is instant - no need to encode each token individually!

        Returns:
            Token embeddings from the model's embedding layer
        """
        if self.token_embeddings is not None:
            return self.token_embeddings

        print("Extracting token embeddings from CLIP model...")

        # Get the token embedding layer from CLIP's transformer
        # The embedding layer contains the raw token embeddings
        token_embedding_layer = self.model.token_embedding.weight

        # Get embeddings and normalize them
        embeddings = token_embedding_layer.detach().cpu().numpy()

        # Normalize embeddings (same as CLIP does for text)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.token_embeddings = embeddings / norms

        print(f"Got {len(self.token_embeddings)} token embeddings directly from model!")
        return self.token_embeddings

    def find_nearest_tokens(self, embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find the nearest tokens to a given embedding from CLIP's vocabulary.

        Args:
            embedding: Target embedding vector
            top_k: Number of top results to return

        Returns:
            List of (token, similarity_score) tuples
        """
        # Get token embeddings from the model (instant!)
        token_embeds = self.get_token_embeddings()

        # Compute cosine similarities
        similarities = np.dot(token_embeds, embedding)

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Map back to token strings
        tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        decoder = {v: k for k, v in tokenizer.encoder.items()}  # id -> token

        results = []
        for idx in top_indices:
            token_str = decoder.get(idx, f"<unk_{idx}>")
            # Clean up the token display
            token_str = token_str.replace('</w>', '')
            results.append((token_str, float(similarities[idx])))

        return results

    def find_opposite(self, word: str, top_k: int = 10):
        """
        Find and display the opposite embedding of a word.

        Args:
            word: Input word
            top_k: Number of results to display
        """
        print(f"\n{'='*60}")
        print(f"Finding opposite embedding for: '{word}'")
        print(f"{'='*60}\n")

        # Get original embedding
        original_emb = self.get_text_embedding(word)
        print(f"Original embedding shape: {original_emb.shape}")
        print(f"Original embedding (first 5 dims): {original_emb[:5]}")

        # Get opposite embedding
        opposite_emb = self.get_opposite_embedding(word)
        print(f"\nOpposite embedding (first 5 dims): {opposite_emb[:5]}")

        # Verify they are opposite
        dot_product = np.dot(original_emb, opposite_emb)
        print(f"Dot product (should be ~-1): {dot_product:.4f}")

        # Find nearest tokens to opposite embedding
        print(f"\nTop {top_k} tokens from CLIP vocabulary closest to the opposite embedding:")
        print("-" * 60)
        results = self.find_nearest_tokens(opposite_emb, top_k)

        for i, (token, score) in enumerate(results, 1):
            print(f"{i:2d}. '{token}' {' ' * (25 - len(token))} (similarity: {score:.4f})")

        print(f"\n{'='*60}\n")
        return results


def main():
    """Main function for interactive use."""
    # Initialize the finder
    finder = OppositeEmbeddingFinder()

    print("\n" + "="*60)
    print("CLIP Opposite Embedding Finder")
    print("="*60)
    print("\nThis tool finds the 'opposite' of a word's embedding by")
    print("negating its CLIP embedding vector and searching CLIP's")
    print("token embedding layer (~49K tokens) to find which tokens")
    print("are closest to that opposite direction in embedding space.")
    print("\nNote: Uses model's embedding layer directly - instant results!")
    print("\nType 'quit' or 'exit' to stop.\n")

    # Interactive loop
    while True:
        word = input("Enter a word: ").strip()

        if word.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not word:
            print("Please enter a valid word.")
            continue

        try:
            finder.find_opposite(word, top_k=10)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
