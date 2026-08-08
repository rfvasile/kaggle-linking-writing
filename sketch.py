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


def _gen_token_events(essay: str, events: DataFrame) -> DataFrame:
    """Version 1: consider atomic operations (i.e., no composite ones like paste)"""
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

        # Aggregate feastures
        up_time = refs["up_time"].mean()
        cur_pos = refs.iloc[-1]["cursor_position"]
        act_time = refs["action_time"].sum()

        # Assign new row
        rows.append({"up_time": up_time, "cursor_position": cur_pos, "action_time": act_time})

    return DataFrame(rows, columns=events.columns)


# %%
# Improved version which reconstructs the text to account for non-sequential operations
# i.e., replace, copy, cut, paste, normal insertion.
# This works by reconstructing the text from the available events, and saving the last
# event that corresponds to an individual token.


def reconstruct_essay(currTextInput):
    essayText = ""
    for Input in currTextInput.values:
        if Input[0] == "Replace":
            replaceTxt = Input[2].split(" => ")
            essayText = (
                essayText[: Input[1] - len(replaceTxt[1])]
                + replaceTxt[1]
                + essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]) :]
            )
            continue
        if Input[0] == "Paste":
            essayText = essayText[: Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]) :]
            continue
        if Input[0] == "Remove/Cut":
            essayText = essayText[: Input[1]] + essayText[Input[1] + len(Input[2]) :]
            continue
        if "M" in Input[0]:
            croppedTxt = Input[0][10:]
            splitTxt = croppedTxt.split(" To ")
            valueArr = [item.split(", ") for item in splitTxt]
            moveData = (
                int(valueArr[0][0][1:]),
                int(valueArr[0][1][:-1]),
                int(valueArr[1][0][1:]),
                int(valueArr[1][1][:-1]),
            )
            if moveData[0] != moveData[2]:
                if moveData[0] < moveData[2]:
                    essayText = (
                        essayText[: moveData[0]]
                        + essayText[moveData[1] : moveData[3]]
                        + essayText[moveData[0] : moveData[1]]
                        + essayText[moveData[3] :]
                    )
                else:
                    essayText = (
                        essayText[: moveData[2]]
                        + essayText[moveData[0] : moveData[1]]
                        + essayText[moveData[2] : moveData[0]]
                        + essayText[moveData[1] :]
                    )
            continue
        essayText = essayText[: Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]) :]
    return essayText


def replay_with_owner(events: DataFrame) -> tuple[str, list[int]]:
    """Replay the log, carrying a parallel per-character owner array."""
    text = ""
    owner: list[int] = []

    cols = events[["activity", "cursor_position", "text_change"]]
    for i, (activity, cursor, change) in enumerate(cols.itertuples(index=False)):
        if activity == "Replace":
            old, new = change.split(" => ")
            start = cursor - len(new)
            text = text[:start] + new + text[start + len(old) :]
            owner = owner[:start] + [i] * len(new) + owner[start + len(old) :]
        elif activity == "Remove/Cut":
            text = text[:cursor] + text[cursor + len(change) :]
            owner = owner[:cursor] + owner[cursor + len(change) :]
        elif "M" in activity:  # "Move From [a, b] To [c, d]"
            lhs, rhs = activity[10:].split(" To ")
            a, b = (int(v.strip("[] ")) for v in lhs.split(", "))
            c, d = (int(v.strip("[] ")) for v in rhs.split(", "))
            if a != c:
                if a < c:
                    text = text[:a] + text[b:d] + text[a:b] + text[d:]
                    owner = owner[:a] + owner[b:d] + owner[a:b] + owner[d:]
                else:
                    text = text[:c] + text[a:b] + text[c:a] + text[b:]
                    owner = owner[:c] + owner[a:b] + owner[c:a] + owner[b:]
        else:  # Input, Paste
            start = cursor - len(change)
            text = text[:start] + change + text[start:]
            owner = owner[:start] + [i] * len(change) + owner[start:]

        assert len(owner) == len(text), f"desync at event {i}: {len(owner)} != {len(text)}"

    return text, owner


def gen_token_events(essay: str, events: DataFrame, owner: list[int]) -> DataFrame:
    enc = tokenizer(essay, return_tensors="pt", return_offsets_mapping=True)
    rows = []
    for s, f in enc["offset_mapping"][0].tolist():
        # Extract overlapping events
        idx = sorted(set(owner[s:f]))

        if not idx:  # [CLS] / [SEP]
            rows.append({"up_time": 0, "cursor_position": 0, "action_time": 0, "n_events": 0})
            continue

        refs = events.iloc[idx]
        rows.append(
            {
                "up_time": refs["up_time"].iloc[-1],  # time at completion
                "cursor_position": refs["cursor_position"].iloc[-1],  # final position
                "action_time": refs["action_time"].sum(),  # a duration as a sum
                "n_events": len(idx),
            }
        )
    return DataFrame(rows)


essay = essays.loc[id, "essay"]
events = df.loc[id][["activity", "cursor_position", "text_change", "up_time", "action_time"]]
token_events = gen_token_events(essay, events)

input = tokenizer(essay, return_tensors="pt", return_offsets_mapping=True)
assert len(token_events) == model(input["input_ids"]).last_hidden_state.shape[1], "Not working"
