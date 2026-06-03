from transformers import GPT2TokenizerFast, GPT2Config, GPT2LMHeadModel

TOKENIZER_DIR = "tokenizer/joao-tokenizer-bpe"

tokenizer = GPT2TokenizerFast(
    vocab_file=f"{TOKENIZER_DIR}/vocab.json",
    merges_file=f"{TOKENIZER_DIR}/merges.txt",
    tokenizer_file=f"{TOKENIZER_DIR}/tokenizer.json",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=512,
    n_ctx=512,
    n_embd=384,
    n_layer=6,
    n_head=6,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

model = GPT2LMHeadModel(config)