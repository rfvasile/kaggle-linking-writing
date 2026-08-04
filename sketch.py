# %%
from pandas import DataFrame, read_parquet
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    DebertaV2Model,
    DebertaV2TokenizerFast,
)
from transformers.modeling_outputs import BaseModelOutput

from scripts.silver_bullet_feats_v1 import get_essay_df

# %%
df = read_parquet("datamount/train_folds.parquet").set_index("id", drop=False)
indices = df.index.unique()

"""
# Good to know:
df
df.loc[indices[1]]
df.loc["001519c8"]  # Series
df.loc["001519c8", "fold"]  # single scalar
df.loc[["001519c8"]]  # df row
"""

# %%
essay_id = "001519c8"
print(f"Time dimension essay {essay_id}:")
print(len(df.loc[essay_id]))

essays = get_essay_df(df.reset_index(drop=True)).set_index("id")

# %%

model_id = "microsoft/deberta-v3-base"
tokenizer: DebertaV2TokenizerFast = AutoTokenizer.from_pretrained(model_id)

"""
essays.columns.to_list()
essays.columns[essays.columns.str.contains("essay")]

print(repr(tokenizer))
"""

essays.head()
text = essays.loc[essay_id, "essay"]

print(text)
print("Text len:", len(text))

tokens = tokenizer.tokenize(text)
print(tokens)
print("Tokenized len:", len(tokens))

inputs: BatchEncoding = tokenizer(text, return_tensors="pt")
print("Inputs", inputs)
print(f"Tokenizer input ids: {inputs['input_ids'].shape}")

# inputs: BatchEncoding = tokenizer(_text2, return_tensors="pt")
# print(f"Tokenizer input ids: {inputs['input_ids'].shape}")
# print(f"len(text): {len(_text2)}")

# %%
config = AutoConfig.from_pretrained(model_id)
model: DebertaV2Model = AutoModel.from_pretrained(model_id)


repr(model)
logits: BaseModelOutput = model(**inputs)
assert logits.last_hidden_state is not None
print(logits.last_hidden_state.shape)

# %%
"""
model.embeddings.word_embeddings.weight.shape

import inspect
from transformers import PreTrainedTokenizerBase
sig = inspect.signature(PreTrainedTokenizerBase.__call__)
print([p for p in sig.parameters if "offset" in p or "return_" in p])
"""
import inspect

inspect.signature(pad_sequence)
# %%

tokens = tokenizer.tokenize(text)
print(len(tokens))
inputs: BatchEncoding = tokenizer(text, return_tensors="pt")
print(f"Tokenizer input ids: {inputs['input_ids'].shape}")
print(f"len(text): {len(text)}")

# %%
# Trying to: determine
essay_id = df.index.unique()[0]

n_events = len(df.loc[essay_id])
n_tokens = len(tokenizer(essays.loc[essay_id, "essay"])["input_ids"])
print(essay_id, n_events, n_tokens)

# %%
id = df.index.unique()[0]


def gen_token_events(essay: str, events: DataFrame) -> DataFrame:
    input = tokenizer(essay, return_tensors="pt", return_offsets_mapping=True)
    offsets: Tensor = input["offset_mapping"]

    rows = []
    for s, f in offsets[0, :].tolist():
        # Parse events
        refs = events.iloc[s:f]
        if refs.empty:
            if rows:
                last = rows[-1]
                rows.append(
                    {
                        "up_time": last["up_time"],
                        "cursor_position": last["cursor_position"],
                        "action_time": last["action_time"],
                    }
                )
            else:
                rows.append({"up_time": 0, "cursor_position": 0, "action_time": 0})

            continue

        # 1. up_time
        up_time = refs["up_time"].mean()

        # 2. Cursor position
        cur_pos = refs.iloc[-1]["cursor_position"]

        # 3. Action time
        act_time = refs["action_time"].sum()

        # Assign new row
        rows.append({"up_time": up_time, "cursor_position": cur_pos, "action_time": act_time})

    return DataFrame(rows, columns=events.columns)


essay = essays.loc[id, "essay"]
events = df.loc[id][["up_time", "cursor_position", "action_time"]]
token_events = gen_token_events(essay, events)

input = tokenizer(essay, return_tensors="pt", return_offsets_mapping=True)
assert len(token_events) == model(input["input_ids"]).last_hidden_state.shape[1], "Not working"
