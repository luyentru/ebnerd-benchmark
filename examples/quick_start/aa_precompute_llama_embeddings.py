from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm
import polars as pl
import numpy as np
import torch

from ebrec.utils._nlp import generate_embeddings_with_transformers
from ebrec.utils._python import batch_items_generator
from ebrec.utils._polars import concat_str_columns

TRANSFORMER_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

DATA_PATH = Path("/dtu/blackhole/0c/215532/ebnerd_data/ebnerd_small").expanduser()
DUMP_DIR = DATA_PATH.joinpath("artifacts", TRANSFORMER_MODEL_NAME.replace("/", "_"))
DUMP_DIR.mkdir(parents=True, exist_ok=True)
print(f"Embeddings will be stored at: {DUMP_DIR}")

df_articles = pl.read_parquet(DATA_PATH.joinpath("articles.parquet"))
print(df_articles.head(5))

print(df_articles.shape)

DEMO = False
if DEMO:
    df_articles = df_articles[:10]

concat_columns = ["title", "subtitle"]

model = AutoModel.from_pretrained(
    TRANSFORMER_MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

# Set the pad_token to the eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

df_articles, col_name = concat_str_columns(df_articles, concat_columns)

print(df_articles.select(col_name).head(5))

BATCH_SIZE = 4
n_batches = int(np.ceil(df_articles.height / BATCH_SIZE))

chunked_text_list = batch_items_generator(df_articles[col_name].to_list(), BATCH_SIZE)
embeddings = (
    generate_embeddings_with_transformers(
        model=model,
        tokenizer=tokenizer,
        text_list=text_list,
        batch_size=BATCH_SIZE,
        disable_tqdm=True,
    )
    for text_list in tqdm(
        chunked_text_list, desc="Encoding", total=n_batches, unit="text"
    )
)
embeddings = torch.vstack(list(embeddings))

embeddings_name = f"{col_name}-{TRANSFORMER_MODEL_NAME}"
series_emb = pl.Series(embeddings_name, embeddings.to("cpu").numpy())
df_emb = df_articles.select("article_id").with_columns(series_emb)

file_path = DUMP_DIR.joinpath(f"{embeddings_name.replace('/', '_')}.parquet")
df_emb.write_parquet(file_path)
print(f"Embeddings saved to: {file_path}")
