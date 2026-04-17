"""Bidirectional LSTM baseline for personality classification.

Architecture:
  - GloVe / random word embeddings (frozen or fine-tuned)
  - Bidirectional LSTM with optional stacking
  - Attention pooling over hidden states
  - Dropout + fully-connected classification head

Supports the same three classification modes as the transformer baseline:
  - "16class": single 16-way softmax
  - "4dim":    four independent binary classifiers (MBTI dimensions)
  - "ocean_binary": five independent binary classifiers (Big Five traits)
"""

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("torch not installed — LSTMBaseline unavailable")

try:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 f1_score, precision_score, recall_score)

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LSTMConfig:
    vocab_size: int = 30000
    embed_dim: int = 300
    hidden_dim: int = 256
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    attention: bool = True
    max_length: int = 512        # tokens (whitespace-split words)
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 20
    early_stopping_patience: int = 5
    loss_weighting: str = "none"  # none | balanced | sqrt_balanced
    glove_path: str | None = None  # path to GloVe .txt file; None = random init
    freeze_embeddings: bool = False
    seed: int = 42
    output_dir: str = "outputs/models/lstm"


# ---------------------------------------------------------------------------
# Tokenizer (simple whitespace + frequency-based vocab)
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    PAD, UNK = 0, 1

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}

    def build_vocab(self, texts: list[str]) -> None:
        from collections import Counter
        counts: Counter = Counter()
        for text in texts:
            counts.update(text.lower().split())
        vocab = ["<PAD>", "<UNK>"] + [w for w, _ in counts.most_common(self.vocab_size - 2)]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        logger.info(f"Vocab size: {len(self.word2idx)}")

    def encode(self, text: str, max_length: int) -> list[int]:
        tokens = text.lower().split()[:max_length]
        ids = [self.word2idx.get(t, self.UNK) for t in tokens]
        ids += [self.PAD] * (max_length - len(ids))
        return ids

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)

    @classmethod
    def load(cls, path: str, vocab_size: int = 30000) -> "SimpleTokenizer":
        tok = cls(vocab_size)
        with open(path, "rb") as f:
            data = pickle.load(f)
        tok.word2idx = data["word2idx"]
        tok.idx2word = data["idx2word"]
        return tok


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class TextDataset(Dataset):
        def __init__(self, token_ids: list[list[int]], labels: list[int]):
            self.token_ids = torch.tensor(token_ids, dtype=torch.long)
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            return self.token_ids[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class _AttentionPool(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attn = nn.Linear(hidden_dim, 1)

        def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # hidden: (B, T, H)  mask: (B, T)  — 1 for real tokens, 0 for padding
            scores = self.attn(hidden).squeeze(-1)        # (B, T)
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = torch.softmax(scores, dim=-1)       # (B, T)
            return (weights.unsqueeze(-1) * hidden).sum(dim=1)  # (B, H)

    class LSTMClassifier(nn.Module):
        def __init__(self, config: LSTMConfig, num_labels: int, pretrained_embeddings: "torch.Tensor | None" = None):
            super().__init__()
            self.config = config
            actual_vocab = pretrained_embeddings.shape[0] if pretrained_embeddings is not None else config.vocab_size

            self.embedding = nn.Embedding(actual_vocab, config.embed_dim, padding_idx=0)
            if pretrained_embeddings is not None:
                self.embedding.weight = nn.Parameter(pretrained_embeddings)
            self.embedding.weight.requires_grad = not config.freeze_embeddings

            self.lstm = nn.LSTM(
                input_size=config.embed_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                bidirectional=config.bidirectional,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
            self.attn_pool = _AttentionPool(out_dim) if config.attention else None
            self.dropout = nn.Dropout(config.dropout)
            self.classifier = nn.Linear(out_dim, num_labels)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            mask = (input_ids != 0).float()               # (B, T)
            emb = self.dropout(self.embedding(input_ids)) # (B, T, E)
            out, _ = self.lstm(emb)                       # (B, T, H)
            if self.attn_pool is not None:
                pooled = self.attn_pool(out, mask)        # (B, H)
            else:
                # mean pooling over non-padding positions
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths
            return self.classifier(self.dropout(pooled))  # (B, num_labels)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LSTMBaseline:
    """Bidirectional LSTM with attention for personality classification."""

    def __init__(self, config: LSTMConfig | None = None):
        if not HAS_TORCH:
            raise ImportError("torch is required: pip install torch")
        self.config = config or LSTMConfig()
        self.tokenizer: SimpleTokenizer | None = None
        self.model: "LSTMClassifier | None" = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_labels(self, labels: list[str]) -> None:
        unique = sorted(set(labels))
        self.label2id = {l: i for i, l in enumerate(unique)}
        self.id2label = {i: l for l, i in self.label2id.items()}

    def _encode(self, texts: list[str]) -> list[list[int]]:
        return [self.tokenizer.encode(t, self.config.max_length) for t in texts]

    def _load_glove(self, vocab: dict[str, int]) -> "torch.Tensor":
        glove_path = self.config.glove_path
        embed = torch.zeros(len(vocab), self.config.embed_dim)
        nn.init.normal_(embed, std=0.01)
        embed[0].zero_()  # PAD
        found = 0
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                if word in vocab:
                    embed[vocab[word]] = torch.tensor([float(x) for x in parts[1:]])
                    found += 1
        logger.info(f"GloVe: loaded {found}/{len(vocab)} vectors from {glove_path}")
        return embed

    def _compute_class_weights(self, labels: list[str]) -> "torch.Tensor | None":
        weighting = self.config.loss_weighting
        if not weighting or weighting == "none":
            return None
        from sklearn.utils.class_weight import compute_class_weight
        n = len(self.id2label)
        ids = np.array([self.label2id[l] for l in labels], dtype=np.int64)
        w = compute_class_weight("balanced", classes=np.arange(n), y=ids)
        if weighting == "sqrt_balanced":
            counts = np.bincount(ids, minlength=n).astype(float)
            counts = np.where(counts == 0, 1.0, counts)
            w = np.sqrt(counts.sum() / (n * counts))
            w = np.clip(w, 0.5, 2.0)
        return torch.tensor(w, dtype=torch.float32)

    def _make_loader(self, texts: list[str], labels: list[str], shuffle: bool) -> "DataLoader":
        ids = self._encode(texts)
        label_ids = [self.label2id[l] for l in labels]
        ds = TextDataset(ids, label_ids)
        return DataLoader(ds, batch_size=self.config.batch_size, shuffle=shuffle, num_workers=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str],
        val_labels: list[str],
        output_dir: str | None = None,
    ) -> None:
        self._setup_labels(train_labels + val_labels)
        output_dir = output_dir or self.config.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(Path(output_dir) / "label_map.json", "w") as f:
            json.dump({"label2id": self.label2id, "id2label": self.id2label}, f)

        # Build vocab and tokenizer
        self.tokenizer = SimpleTokenizer(self.config.vocab_size)
        self.tokenizer.build_vocab(train_texts)
        self.tokenizer.save(str(Path(output_dir) / "tokenizer.pkl"))

        # Pretrained embeddings
        pretrained = None
        if self.config.glove_path and Path(self.config.glove_path).exists():
            pretrained = self._load_glove(self.tokenizer.word2idx)

        # Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMClassifier(self.config, len(self.label2id), pretrained).to(device)
        logger.info(
            f"LSTMClassifier: {sum(p.numel() for p in self.model.parameters()):,} params | device={device}"
        )

        class_weights = self._compute_class_weights(train_labels)
        weight = class_weights.to(device) if class_weights is not None else None
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=2, factor=0.5
        )

        train_loader = self._make_loader(train_texts, train_labels, shuffle=True)
        val_loader = self._make_loader(val_texts, val_labels, shuffle=False)

        best_val_acc, best_epoch, patience_count = -1.0, 0, 0

        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            self.model.train()
            total_loss = 0.0
            for input_ids, labels_t in train_loader:
                input_ids, labels_t = input_ids.to(device), labels_t.to(device)
                optimizer.zero_grad()
                logits = self.model(input_ids)
                loss = criterion(logits, labels_t)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            # Validate
            val_acc, val_f1 = self._eval_epoch(val_loader, device)
            scheduler.step(val_acc)
            logger.info(
                f"Epoch {epoch}/{self.config.num_epochs} | "
                f"loss={total_loss / len(train_loader):.4f} | "
                f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc, best_epoch, patience_count = val_acc, epoch, 0
                torch.save(self.model.state_dict(), Path(output_dir) / "best_model.pt")
            else:
                patience_count += 1
                if patience_count >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                    break

        # Reload best checkpoint
        self.model.load_state_dict(torch.load(Path(output_dir) / "best_model.pt", map_location=device))
        logger.info(f"Training complete. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")

        # Save config for load()
        cfg_dict = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                    for k, v in vars(self.config).items()}
        with open(Path(output_dir) / "lstm_config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

    def _eval_epoch(self, loader: "DataLoader", device: "torch.device") -> tuple[float, float]:
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for input_ids, labels_t in loader:
                logits = self.model(input_ids.to(device))
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels_t.numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return acc, f1

    def predict(self, texts: list[str], batch_size: int = 64) -> list[str]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")
        device = next(self.model.parameters()).device
        self.model.eval()
        all_preds: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            ids = torch.tensor(self._encode(batch), dtype=torch.long).to(device)
            with torch.no_grad():
                logits = self.model(ids)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(self.id2label[int(p)] for p in preds)
        return all_preds

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        preds = self.predict(texts)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "precision_weighted": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "recall_weighted": recall_score(labels, preds, average="weighted", zero_division=0),
            "classification_report": classification_report(labels, preds, zero_division=0),
        }

    @classmethod
    def load(cls, checkpoint_dir: str) -> "LSTMBaseline":
        checkpoint_dir = Path(checkpoint_dir)

        with open(checkpoint_dir / "lstm_config.json") as f:
            raw = json.load(f)
        cfg = LSTMConfig(**{
            k: v for k, v in raw.items() if k in LSTMConfig.__dataclass_fields__
        })

        instance = cls(cfg)
        instance.tokenizer = SimpleTokenizer.load(str(checkpoint_dir / "tokenizer.pkl"), cfg.vocab_size)

        with open(checkpoint_dir / "label_map.json") as f:
            mapping = json.load(f)
        instance.label2id = mapping["label2id"]
        instance.id2label = {int(k): v for k, v in mapping["id2label"].items()}

        pretrained = None
        if cfg.glove_path and Path(cfg.glove_path).exists():
            pretrained = instance._load_glove(instance.tokenizer.word2idx)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.model = LSTMClassifier(cfg, len(instance.label2id), pretrained).to(device)
        instance.model.load_state_dict(
            torch.load(checkpoint_dir / "best_model.pt", map_location=device)
        )
        logger.info(f"Loaded LSTM model from {checkpoint_dir}")
        return instance
