from pathlib import Path
import polars as pl
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
from tqdm import tqdm
import gc
import os
import torch.cuda

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)

# Configuration
PATH = Path("/dtu/blackhole/0c/215532/ebnerd_data").expanduser()
EMBEDDINGS_PATH = Path("llama_embeddings").resolve()
EMBEDDINGS_PATH.mkdir(exist_ok=True, parents=True)

DATASPLIT = "ebnerd_small"
MAX_TITLE_LENGTH = 30
HISTORY_SIZE = 20
BATCH_SIZE = 4  # Reduced from 32
USE_HALF_PRECISION = True  # Add this to use FP16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    """Load ebnerd behaviors data"""
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .collect()
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
    )
    return df_behaviors, df_history

def create_llama_embeddings_batch(texts, tokenizer, model, max_length):
    """Create embeddings for a batch of texts"""
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling of last hidden states as embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
    # Clear cache after each batch
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
        
    return embeddings

def main():
    print("Loading data...")
    # Load all necessary data
    df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))
    df_train_behaviors, df_train_history = ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "train"), 
        history_size=HISTORY_SIZE
    )
    df_val_behaviors, df_val_history = ebnerd_from_path(
        PATH.joinpath(DATASPLIT, "validation"), 
        history_size=HISTORY_SIZE
    )

    # Collect all unique article IDs
    print("Collecting unique article IDs...")
    required_article_ids = set()
    
    # From training set
    required_article_ids.update(
        df_train_history[DEFAULT_HISTORY_ARTICLE_ID_COL].explode().to_list() +
        df_train_behaviors[DEFAULT_INVIEW_ARTICLES_COL].explode().to_list()
    )
    
    # From validation set
    required_article_ids.update(
        df_val_history[DEFAULT_HISTORY_ARTICLE_ID_COL].explode().to_list() +
        df_val_behaviors[DEFAULT_INVIEW_ARTICLES_COL].explode().to_list()
    )

    # Filter articles
    df_articles_filtered = df_articles.filter(pl.col('article_id').is_in(required_article_ids))
    print(f"Found {len(df_articles_filtered)} unique articles")

    # Load LLaMA model
    print("Loading LLaMA model...")
    llama_model_name = "meta-llama/Llama-2-7b-hf"
    
    # Load with low memory usage configuration
    config = AutoConfig.from_pretrained(llama_model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    model = AutoModel.from_pretrained(
        llama_model_name,
        config=config,
        torch_dtype=torch.float16 if USE_HALF_PRECISION else torch.float32,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    
    if USE_HALF_PRECISION:
        model = model.half()
    
    # Optional: delete unused objects
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    # Create embeddings in smaller batches
    print("Creating embeddings...")
    embeddings = []
    article_ids = []
    
    try:
        for i in tqdm(range(0, len(df_articles_filtered), BATCH_SIZE)):
            batch_df = df_articles_filtered.slice(i, BATCH_SIZE)
            
            # Concatenate title and subtitle for each article
            texts = [
                " ".join([
                    str(row[col]) 
                    for col in [DEFAULT_TITLE_COL, DEFAULT_SUBTITLE_COL] 
                    if row[col] is not None
                ])
                for row in batch_df.iter_rows(named=True)
            ]
            
            # Get embeddings for batch
            batch_embeddings = create_llama_embeddings_batch(
                texts, 
                tokenizer, 
                model, 
                MAX_TITLE_LENGTH
            )
            
            embeddings.append(batch_embeddings)
            article_ids.extend(batch_df['article_id'].to_list())
            
            # Optional: Clear CUDA cache periodically
            if i % (BATCH_SIZE * 10) == 0:
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("Interrupted by user")

    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    
    # Save embeddings and article IDs
    print("Saving embeddings...")
    np.save(EMBEDDINGS_PATH / "llama_embeddings.npy", embeddings)
    np.save(EMBEDDINGS_PATH / "article_ids.npy", np.array(article_ids))
    
    # Save embedding dimension for reference
    with open(EMBEDDINGS_PATH / "embedding_dim.txt", "w") as f:
        f.write(str(embeddings.shape[1]))

    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    print(f"Embedding shape: {embeddings.shape}")

if __name__ == "__main__":
    main()