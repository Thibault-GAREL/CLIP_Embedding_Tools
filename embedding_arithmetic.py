#!/usr/bin/env python3
"""
Embedding Arithmetic using CLIP
Perform vector operations like: word1 + word2 = ?
Example: man + kingdom ≈ king
"""

import torch
import clip
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


class EmbeddingArithmetic:
    def __init__(self, model_name: str = "ViT-B/32", device: str = None, vocab_size: int = 10000):
        """
        Initialize the CLIP model for embedding arithmetic.

        Args:
            model_name: CLIP model to use (default: "ViT-B/32")
            device: Device to run on (cuda/cpu), auto-detected if None
            vocab_size: Number of most common words to use (default: 10000)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # Load vocabulary
        print(f"Loading vocabulary of {vocab_size} common words...")
        self.vocabulary = self._load_vocabulary(vocab_size)
        print(f"Loaded {len(self.vocabulary)} words")

        self.vocab_embeddings = None
        print("Model loaded! Ready for embedding arithmetic.")

    def _load_vocabulary(self, vocab_size: int) -> List[str]:
        """Load a vocabulary of common English words."""
        # Comprehensive word list organized by category
        words = [
            # Core concepts
            "man", "woman", "boy", "girl", "person", "people", "human", "child", "adult",
            "father", "mother", "son", "daughter", "brother", "sister", "family",

            # Royalty & leadership
            "king", "queen", "prince", "princess", "monarch", "emperor", "empress",
            "ruler", "leader", "chief", "lord", "lady", "duke", "duchess",
            "kingdom", "empire", "throne", "crown", "royal", "nobility",

            # Occupations
            "doctor", "nurse", "teacher", "student", "worker", "engineer", "scientist",
            "artist", "musician", "writer", "poet", "actor", "director", "chef",
            "farmer", "builder", "driver", "pilot", "soldier", "police", "firefighter",

            # Animals
            "dog", "cat", "horse", "cow", "pig", "sheep", "goat", "chicken",
            "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit", "mouse",
            "bird", "eagle", "owl", "dove", "crow", "fish", "whale", "shark",

            # Nature & Places
            "sun", "moon", "star", "sky", "cloud", "rain", "snow", "wind",
            "mountain", "hill", "valley", "river", "lake", "ocean", "sea", "beach",
            "forest", "tree", "flower", "grass", "plant", "desert", "island",
            "city", "town", "village", "country", "nation", "world", "earth",

            # Buildings & Structures
            "house", "home", "building", "tower", "castle", "palace", "temple",
            "church", "school", "hospital", "library", "museum", "theater",
            "bridge", "road", "street", "park", "garden", "farm", "factory",

            # Objects
            "book", "pen", "paper", "computer", "phone", "car", "bicycle",
            "table", "chair", "bed", "door", "window", "wall", "floor",
            "food", "water", "bread", "meat", "fruit", "vegetable",
            "clothes", "shirt", "pants", "dress", "shoes", "hat",

            # Emotions & States
            "happy", "sad", "angry", "afraid", "excited", "bored", "tired",
            "love", "hate", "joy", "sorrow", "peace", "war", "hope", "fear",
            "strong", "weak", "brave", "cowardly", "kind", "cruel", "wise", "foolish",

            # Qualities
            "big", "small", "large", "tiny", "huge", "little",
            "tall", "short", "high", "low", "deep", "shallow",
            "wide", "narrow", "thick", "thin", "fat", "slim",
            "hot", "cold", "warm", "cool", "freezing", "burning",
            "fast", "slow", "quick", "rapid", "swift",
            "hard", "soft", "rough", "smooth", "sharp", "dull",
            "heavy", "light", "dense", "loose",
            "bright", "dark", "light", "dim", "shiny", "dull",
            "loud", "quiet", "silent", "noisy",
            "clean", "dirty", "pure", "filthy",
            "new", "old", "young", "ancient", "modern",
            "good", "bad", "great", "terrible", "excellent", "poor",
            "beautiful", "ugly", "pretty", "handsome",
            "rich", "poor", "wealthy", "broke",
            "full", "empty", "complete", "incomplete",

            # Actions
            "walk", "run", "jump", "fly", "swim", "climb", "fall",
            "eat", "drink", "sleep", "wake", "rest", "work",
            "speak", "talk", "shout", "whisper", "sing", "laugh", "cry",
            "think", "know", "learn", "teach", "understand", "forget",
            "see", "look", "watch", "hear", "listen", "smell", "taste", "feel",
            "give", "take", "buy", "sell", "trade", "steal",
            "make", "create", "build", "destroy", "break", "fix",
            "open", "close", "start", "stop", "begin", "end",
            "go", "come", "leave", "arrive", "enter", "exit",
            "push", "pull", "throw", "catch", "hit", "kick",
            "love", "hate", "like", "dislike", "want", "need",

            # Time
            "day", "night", "morning", "afternoon", "evening", "noon", "midnight",
            "today", "tomorrow", "yesterday", "now", "then", "soon", "late", "early",
            "year", "month", "week", "hour", "minute", "second",
            "past", "present", "future", "history",
            "spring", "summer", "autumn", "winter", "season",

            # Directions & Positions
            "up", "down", "left", "right", "forward", "backward",
            "north", "south", "east", "west",
            "inside", "outside", "above", "below", "over", "under",
            "near", "far", "close", "distant", "here", "there",
            "front", "back", "side", "top", "bottom", "middle", "center",

            # Numbers & Quantities
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "many", "few", "some", "all", "none", "more", "less", "most", "least",
            "first", "second", "third", "last", "next", "previous",

            # Abstract Concepts
            "life", "death", "birth", "age", "time", "space",
            "truth", "lie", "fact", "fiction", "real", "fake",
            "idea", "thought", "mind", "soul", "spirit", "body",
            "power", "force", "energy", "strength", "weakness",
            "right", "wrong", "good", "evil", "justice", "crime",
            "freedom", "slavery", "liberty", "prison",
            "success", "failure", "victory", "defeat", "win", "lose",
            "problem", "solution", "question", "answer",
            "beginning", "end", "start", "finish",
            "part", "whole", "piece", "fragment",

            # Colors
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "black", "white", "gray", "brown", "gold", "silver",

            # Materials
            "wood", "stone", "metal", "iron", "steel", "gold", "silver",
            "glass", "plastic", "paper", "cloth", "leather",
            "water", "ice", "fire", "earth", "air",
        ]

        # Deduplicate and limit to vocab_size
        words = list(dict.fromkeys(words))  # Remove duplicates while preserving order
        return words[:vocab_size]

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

    def compute_vocabulary_embeddings(self, batch_size: int = 64):
        """
        Compute embeddings for all vocabulary words through the full CLIP model.
        Uses batching for efficiency.

        Args:
            batch_size: Number of words to encode in each batch
        """
        if self.vocab_embeddings is not None:
            return

        print(f"Computing embeddings for {len(self.vocabulary)} words...")
        print("This will take about 10-20 seconds...")

        embeddings = []

        # Process in batches for efficiency
        for i in tqdm(range(0, len(self.vocabulary), batch_size), desc="Encoding vocabulary"):
            batch_words = self.vocabulary[i:i + batch_size]

            # Tokenize batch
            text_inputs = clip.tokenize(batch_words).to(self.device)

            # Encode through full CLIP model
            with torch.no_grad():
                text_features = self.model.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            embeddings.append(text_features.cpu().numpy())

        # Combine all batches
        self.vocab_embeddings = np.vstack(embeddings)
        print(f"Vocabulary embeddings computed! Shape: {self.vocab_embeddings.shape}")

    def find_nearest_words(self, embedding: np.ndarray, top_k: int = 10, exclude_words: List[str] = None) -> List[Tuple[str, float]]:
        """
        Find the nearest words to a given embedding from the vocabulary.

        Args:
            embedding: Target embedding vector
            top_k: Number of top results to return
            exclude_words: List of words to exclude from results

        Returns:
            List of (word, similarity_score) tuples
        """
        # Compute vocabulary embeddings if not done yet
        self.compute_vocabulary_embeddings()

        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Compute cosine similarities
        similarities = np.dot(self.vocab_embeddings, embedding)

        # Get top k (get more than needed in case we need to filter)
        top_indices = np.argsort(similarities)[::-1][:top_k * 3]

        results = []
        exclude_set = set([w.lower() for w in exclude_words]) if exclude_words else set()

        for idx in top_indices:
            word = self.vocabulary[idx]

            # Skip excluded words
            if word.lower() in exclude_set:
                continue

            results.append((word, float(similarities[idx])))

            if len(results) >= top_k:
                break

        return results

    def add_vectors(self, word1: str, word2: str, top_k: int = 10):
        """
        Add two word embeddings and find nearest words.
        Example: man + kingdom ≈ king

        Args:
            word1: First word
            word2: Second word
            top_k: Number of results to display
        """
        print(f"\n{'='*60}")
        print(f"Vector Addition: '{word1}' + '{word2}' = ?")
        print(f"{'='*60}\n")

        # Get embeddings
        emb1 = self.get_text_embedding(word1)
        emb2 = self.get_text_embedding(word2)

        print(f"'{word1}' embedding (first 5 dims): {emb1[:5]}")
        print(f"'{word2}' embedding (first 5 dims): {emb2[:5]}")

        # Add vectors
        result_emb = emb1 + emb2
        print(f"\nResult embedding (first 5 dims): {result_emb[:5]}")

        # Find nearest words (exclude input words)
        print(f"\nTop {top_k} words closest to '{word1}' + '{word2}':")
        print("-" * 60)
        results = self.find_nearest_words(result_emb, top_k, exclude_words=[word1, word2])

        for i, (word, score) in enumerate(results, 1):
            print(f"{i:2d}. '{word}' {' ' * (25 - len(word))} (similarity: {score:.4f})")

        print(f"\n{'='*60}\n")
        return results

    def subtract_vectors(self, word1: str, word2: str, top_k: int = 10):
        """
        Subtract two word embeddings and find nearest words.
        Example: king - man ≈ queen (conceptually)

        Args:
            word1: First word (minuend)
            word2: Second word (subtrahend)
            top_k: Number of results to display
        """
        print(f"\n{'='*60}")
        print(f"Vector Subtraction: '{word1}' - '{word2}' = ?")
        print(f"{'='*60}\n")

        # Get embeddings
        emb1 = self.get_text_embedding(word1)
        emb2 = self.get_text_embedding(word2)

        print(f"'{word1}' embedding (first 5 dims): {emb1[:5]}")
        print(f"'{word2}' embedding (first 5 dims): {emb2[:5]}")

        # Subtract vectors
        result_emb = emb1 - emb2
        print(f"\nResult embedding (first 5 dims): {result_emb[:5]}")

        # Find nearest words (exclude input words)
        print(f"\nTop {top_k} words closest to '{word1}' - '{word2}':")
        print("-" * 60)
        results = self.find_nearest_words(result_emb, top_k, exclude_words=[word1, word2])

        for i, (word, score) in enumerate(results, 1):
            print(f"{i:2d}. '{word}' {' ' * (25 - len(word))} (similarity: {score:.4f})")

        print(f"\n{'='*60}\n")
        return results

    def complex_arithmetic(self, operations: str, top_k: int = 10):
        """
        Perform complex vector arithmetic with multiple operations.
        Example: "king - man + woman"

        Args:
            operations: String with operations like "king - man + woman"
            top_k: Number of results to display
        """
        print(f"\n{'='*60}")
        print(f"Complex Arithmetic: {operations}")
        print(f"{'='*60}\n")

        # Parse the operations
        import re
        # Split by + and - while keeping the operators
        tokens = re.split(r'(\+|\-)', operations)
        tokens = [t.strip() for t in tokens if t.strip()]

        if not tokens:
            print("Error: No valid operations found")
            return []

        # Get all words involved for exclusion
        words = [t for t in tokens if t not in ['+', '-']]

        # Start with the first word
        result_emb = self.get_text_embedding(tokens[0])
        print(f"Starting with '{tokens[0]}'")

        # Process operations
        i = 1
        while i < len(tokens):
            if i + 1 < len(tokens):
                operator = tokens[i]
                word = tokens[i + 1]

                emb = self.get_text_embedding(word)

                if operator == '+':
                    result_emb = result_emb + emb
                    print(f"  + '{word}'")
                elif operator == '-':
                    result_emb = result_emb - emb
                    print(f"  - '{word}'")

                i += 2
            else:
                break

        print(f"\nResult embedding (first 5 dims): {result_emb[:5]}")

        # Find nearest words (exclude input words)
        print(f"\nTop {top_k} words closest to the result:")
        print("-" * 60)
        results = self.find_nearest_words(result_emb, top_k, exclude_words=words)

        for i, (word, score) in enumerate(results, 1):
            print(f"{i:2d}. '{word}' {' ' * (25 - len(word))} (similarity: {score:.4f})")

        print(f"\n{'='*60}\n")
        return results


def main():
    """Main function for interactive use."""
    arithmetic = EmbeddingArithmetic()

    print("\n" + "="*60)
    print("CLIP Embedding Arithmetic")
    print("="*60)
    print("\nPerform vector operations on word embeddings!")
    print("\nCommands:")
    print("  add <word1> <word2>       - Add two vectors")
    print("  sub <word1> <word2>       - Subtract vectors (word1 - word2)")
    print("  calc <expression>         - Complex expression (e.g., 'king - man + woman')")
    print("  quit/exit                 - Exit program")
    print("\nExamples:")
    print("  add man kingdom")
    print("  sub king man")
    print("  calc king - man + woman")
    print()

    while True:
        try:
            user_input = input("Enter command: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            parts = user_input.split(None, 1)
            if len(parts) == 0:
                continue

            command = parts[0].lower()

            if command == 'add' and len(parts) > 1:
                words = parts[1].split()
                if len(words) >= 2:
                    arithmetic.add_vectors(words[0], words[1])
                else:
                    print("Error: Need two words for addition. Example: add man kingdom")

            elif command == 'sub' and len(parts) > 1:
                words = parts[1].split()
                if len(words) >= 2:
                    arithmetic.subtract_vectors(words[0], words[1])
                else:
                    print("Error: Need two words for subtraction. Example: sub king man")

            elif command == 'calc' and len(parts) > 1:
                expression = parts[1]
                arithmetic.complex_arithmetic(expression)

            else:
                print("Unknown command. Type 'quit' to exit or use: add/sub/calc")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
