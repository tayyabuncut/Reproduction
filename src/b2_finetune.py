"""Baseline 2 fine-tuning (Li, Yuan & Zhang, 2024, Sec. 5.1).

Core model (Llama-2-7b-chat-hf):
  * INT4 (QLoRA), LoRA rank 16, 5 epochs / max 1000 steps, batch size 8, lr 2e-4.
  * prompt template = App. B.3 (summarise, key points only).

Embedding model (mxbai-embed-large-v1):
  * SentenceTransformer + MultipleNegativesRankingLoss, 10 epochs with warmup.
  * paper also reports a run of 17,390 steps at batch size 8.

Both functions are no-ops without a GPU; they return the output adapter / model directory for the pipeline.
"""
from __future__ import annotations

from pathlib import Path


CORE_FT_SYSTEM = (
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to "
    "answer the question. If you don't know the answer, just say that you don't know. Summarize your answer "
    "and ensure the answer only contains key points."
)


def _format_core_example(tok, question: str, context: str, answer: str) -> str:
    msgs = [
        {"role": "system", "content": CORE_FT_SYSTEM},
        {"role": "user", "content": f"Question: {question}\nContext:\n{context}"},
        {"role": "assistant", "content": answer},
    ]
    return tok.apply_chat_template(msgs, tokenize=False)


def finetune_core_qlora(
    base_model: str,
    qa_train: list[dict],
    output_dir: str,
    hf_token: str | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lr: float = 2e-4,
    epochs: int = 5,
    max_steps: int = 1000,
    batch_size: int = 8,
    grad_accum: int = 1,
    max_seq_len: int = 1024,
    seed: int = 42,
) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"

    texts = [_format_core_example(tok, r["question"], r.get("context", ""), r["answer"]) for r in qa_train]
    ds = Dataset.from_dict({"text": texts})

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb, device_map="auto",
                                                 token=hf_token)
    model.config.use_cache = False
    peft_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    cfg = SFTConfig(output_dir=output_dir, per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=grad_accum, learning_rate=lr, num_train_epochs=epochs,
                    max_steps=max_steps, logging_steps=25, save_strategy="no", seed=seed,
                    max_length=max_seq_len, report_to=[], fp16=True, warmup_ratio=0.03,
                    dataset_text_field="text")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    return output_dir


def finetune_embedding(
    base_model: str,
    qa_train: list[dict],
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    max_steps: int | None = None,
    warmup_ratio: float = 0.1,
    seed: int = 42,
) -> str:
    """MultipleNegativesRankingLoss on (question, answer/context) positive pairs."""
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    model = SentenceTransformer(base_model)
    examples = []
    for r in qa_train:
        pos = r.get("context") or r.get("answer")
        if r.get("question") and pos:
            examples.append(InputExample(texts=[r["question"], pos]))
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    steps_per_epoch = len(loader)
    total = max_steps if max_steps else steps_per_epoch * epochs
    warmup = int(total * warmup_ratio)
    model.fit(train_objectives=[(loader, loss)],
              epochs=epochs if not max_steps else max(1, max_steps // max(1, steps_per_epoch)),
              warmup_steps=warmup, show_progress_bar=True, output_path=output_dir)
    return output_dir
