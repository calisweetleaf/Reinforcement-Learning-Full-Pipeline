# Context Compression

## Overview

Long-context language models are limited by quadratic attention complexity. Context compression reduces extended contexts to fixed-size representations, enabling processing of arbitrary-length sequences.

## Motivation

**Problem**: Standard attention is O(n²) in sequence length

- 4K context: manageable
- 128K context: expensive
- 1M+ context: prohibitive

**Solution**: Compress long contexts into dense representations

## Compression Architecture

### 1. Attention-Based Compression

```
Long Context (N tokens)
    ↓
[Chunk 1] [Chunk 2] ... [Chunk K]  (K chunks of size C)
    ↓           ↓             ↓
[Encoder] [Encoder] ... [Encoder]  (Process each chunk)
    ↓           ↓             ↓
[Memory 1] [Memory 2] ... [Memory K]  (M memory tokens per chunk)
    ↓
Concatenate memories + compress
    ↓
Fixed-size compressed representation
```

**Implementation**:

```python
class AttentionCompressor(nn.Module):
    def __init__(self, chunk_size=512, memory_tokens=32):
        self.chunk_size = chunk_size
        self.memory_tokens = memory_tokens
        self.memory_embeds = nn.Parameter(
            torch.randn(1, memory_tokens, hidden_dim)
        )
    
    def compress(self, long_context):
        """Compress long context into fixed memory."""
        # Split into chunks
        chunks = long_context.split(self.chunk_size, dim=1)
        
        compressed_memories = []
        for chunk in chunks:
            # Cross-attention: memory queries attend to chunk
            memory = self.cross_attention(
                query=self.memory_embeds,
                key=chunk,
                value=chunk
            )
            compressed_memories.append(memory)
        
        # Final compression
        return self.final_compress(torch.cat(compressed_memories, dim=1))
```

### 2. Hierarchical Compression

Multi-level compression for very long contexts:

```
Level 0: Raw tokens (1M tokens)
    ↓ Compress (100:1)
Level 1: 10K segment representations
    ↓ Compress (100:1)
Level 2: 100 document representations
    ↓ Compress (10:1)
Level 3: 10 global representations
```

### 3. Learned Compression

Train a compression network end-to-end:

```python
class LearnedCompressor(nn.Module):
    def __init__(self, target_compression=64):
        self.encoder = TransformerEncoder(layers=6)
        self.compression_token = nn.Parameter(
            torch.randn(1, target_compression, hidden_dim)
        )
        self.compressor = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=16
        )
    
    def forward(self, context):
        # Encode context
        encoded = self.encoder(context)
        
        # Compress to fixed size
        compressed = self.compressor(
            self.compression_token,
            encoded
        )
        return compressed
```

## ContextCompressor Class

The pipeline includes a `ContextCompressor` module:

```python
class ContextCompressor(nn.Module):
    """
    Compresses long contexts for efficient processing.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        compression_ratio: int = 16,
        method: str = "attention"  # "attention", "pooling", "learned"
    ):
        self.base_model = base_model
        self.compression_ratio = compression_ratio
        self.method = method
    
    def compress(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress long sequences.
        
        Returns:
            compressed_ids: Compressed token representations
            compressed_mask: Attention mask for compressed tokens
        """
        if input_ids.size(1) <= self.max_context:
            # Short context, no compression needed
            return input_ids, attention_mask
        
        # Apply compression based on method
        if self.method == "attention":
            return self._attention_compress(input_ids, attention_mask)
        elif self.method == "pooling":
            return self._pooling_compress(input_ids, attention_mask)
        else:
            return self._learned_compress(input_ids, attention_mask)
    
    def decompress(
        self,
        compressed: torch.Tensor,
        query_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve relevant information from compressed representation.
        """
        # Cross-attention to extract needed information
        return self.cross_attention(query_positions, compressed)
```

## Usage in Pipeline

### Automatic Compression

```python
from rlhf import RLHFOrchestrator

orchestrator = RLHFOrchestrator(
    use_context_compression=True,
    compression_config={
        "method": "attention",
        "compression_ratio": 16,
        "chunk_size": 512
    }
)

# Long context is automatically compressed
response = orchestrstrator.generate(
    long_prompt,  # Can be 100K+ tokens
    max_new_tokens=1024
)
```

### Manual Compression

```python
# Compress once, use multiple times
compressor = ContextCompressor(model)
compressed = compressor.compress(long_document)

# Query compressed representation multiple times
for question in questions:
    response = model.generate_with_compressed(
        question,
        compressed_context=compressed
    )
```

## Integration with Policy Models

```python
class PolicyModelWithCompression(PolicyModel):
    def __init__(self, *args, use_compression=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.compressor = ContextCompressor(
            self.model,
            compression_ratio=16
        ) if use_compression else None
    
    def generate(self, prompts, **kwargs):
        if self.compressor and len(prompts[0]) > self.max_length:
            # Compress long prompts
            compressed = self.compressor.compress(prompts)
            return self.generate_from_compressed(compressed, **kwargs)
        return super().generate(prompts, **kwargs)
```

## Training with Compression

When training with context compression:

```python
class CompressionTrainer:
    def train_step(self, batch):
        # Compress contexts
        compressed_contexts = self.compressor.compress(
            batch['contexts']
        )
        
        # Standard training with compressed representations
        outputs = self.model(
            input_ids=batch['queries'],
            compressed_context=compressed_contexts
        )
        
        # Reconstruction loss (optional)
        if self.use_reconstruction:
            recon_loss = self.reconstruction_loss(
                compressed_contexts,
                batch['contexts']
            )
            loss = task_loss + self.recon_weight * recon_loss
        
        return loss
```

## Compression Methods Comparison

| Method | Compression | Quality | Speed | Use Case |
|--------|-------------|---------|-------|----------|
| Attention | High | High | Medium | General purpose |
| Mean Pooling | Very High | Medium | Fast | Document-level |
| Max Pooling | Very High | Low | Fast | Key phrase extraction |
| Hierarchical | Adjustable | High | Slow | Very long contexts |
| Learned | Adjustable | Highest | Medium | Fixed domain |

## Performance Characteristics

### Memory Savings

```
Without compression:
  Memory = O(batch_size × seq_len²)

With compression (ratio R):
  Memory = O(batch_size × (seq_len/R)²)
  
For R=16, 128K context:
  - Uncompressed: O(128K²) = O(16B) operations
  - Compressed: O(8K²) = O(64M) operations
  - Speedup: ~250×
```

### Quality Trade-offs

Compression inherently loses information. Mitigation strategies:

1. **Query-aware compression**: Compress based on what will be queried
2. **Hierarchical access**: Keep some tokens uncompressed (recent/first/last)
3. **Iterative refinement**: Decompress → process → recompress

## Advanced Techniques

### 1. Selective Compression

Keep important tokens uncompressed:

```python
def selective_compress(tokens, importance_scores):
    # Keep top-k important tokens
    important_indices = importance_scores.topk(k).indices
    
    # Compress rest
    to_compress = tokens[~important_indices]
    compressed = compressor.compress(to_compress)
    
    # Concatenate
    return torch.cat([tokens[important_indices], compressed], dim=1)
```

### 2. Query-Conditional Compression

Compress differently based on the query:

```python
class QueryConditionalCompressor:
    def compress(self, context, query):
        # Use query to guide compression
        attention_weights = self.compute_query_context_attention(
            query, context
        )
        
        # Sample compression based on attention
        return self.attention_guided_compress(context, attention_weights)
```

### 3. Recursive Compression

For extremely long contexts (millions of tokens):

```python
def recursive_compress(context, levels=3):
    for level in range(levels):
        context = compress_level(context, level)
    return context
```

## Evaluation

Measure compression quality:

```python
def evaluate_compression(compressor, test_data):
    metrics = {
        'reconstruction_loss': 0,
        'downstream_accuracy': 0,
        'compression_ratio': 0
    }
    
    for example in test_data:
        original = example['context']
        compressed = compressor.compress(original)
        reconstructed = compressor.decompress(compressed)
        
        # Reconstruction quality
        metrics['reconstruction_loss'] += mse(reconstructed, original)
        
        # Downstream task performance
        pred = model.predict_with_compressed(compressed, example['question'])
        metrics['downstream_accuracy'] += (pred == example['answer'])
        
        # Compression stats
        metrics['compression_ratio'] += len(original) / len(compressed)
    
    return {k: v / len(test_data) for k, v in metrics.items()}
```

## Applications

### Long Document QA

```python
# Compress 100K token document
doc_compressed = compressor.compress(long_document)

# Answer multiple questions efficiently
for question in questions:
    answer = model.answer(question, context=doc_compressed)
```

### Conversation History

```python
# Compress conversation history as it grows
conversation = []
compressed_history = None

for turn in dialogue:
    if compressed_history is None:
        compressed_history = compressor.compress(turn)
    else:
        # Incremental compression
        compressed_history = compressor.incremental_compress(
            compressed_history, turn
        )
```

### Multi-Document Retrieval

```python
# Compress each document
doc_embeddings = [compressor.compress(doc) for doc in documents]

# Efficient similarity search
scores = [similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
```
