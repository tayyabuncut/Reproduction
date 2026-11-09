"""LLM generation wrappers.

HFGenerator loads a causal LM from HuggingFace in 4-bit NF4 (bitsandbytes) so that 7-8B models fit on a
free Colab T4 (16 GB). Both papers ran full/INT4 models on larger hardware; 4-bit inference is a documented
deviation for Baseline 1 (Baseline 2 itself fine-tuned in INT4).

MockLLM is a deterministic stand-in used only by the CPU unit tests (tests/run_cpu_tests.py) so the
harness logic can be checked without a GPU. It must never be used for reported results.
"""
from __future__ import annotations

import gc
import re
from typing import Sequence


class HFGenerator:
    def __init__(
        self,
        model_id: str,
        load_in_4bit: bool = True,
        adapter_path: str | None = None,
        max_input_tokens: int = 3800,
        hf_token: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self.tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.tok.padding_side = "left"  # required for batched decoder-only generation
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        qcfg = None
        if load_in_4bit:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=qcfg,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        # Llama-3 ends turns with <|eot_id|>; stop on it as well as eos.
        self.stop_ids = [self.tok.eos_token_id]
        eot = self.tok.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot != self.tok.unk_token_id and eot is not None:
            self.stop_ids.append(eot)

    def chat_to_prompt(self, messages: list[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 4,
    ) -> list[str]:
        """`prompts` are fully formatted strings (use chat_to_prompt for chat models)."""
        import torch

        outs: list[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = list(prompts[i : i + batch_size])
            enc = self.tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                add_special_tokens=not batch[0].startswith(self.tok.bos_token or "\x00"),
            ).to(self.model.device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id, eos_token_id=self.stop_ids)
            if temperature and temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kw.update(do_sample=False)
            with torch.inference_mode():
                gen = self.model.generate(**enc, **kw)
            new = gen[:, enc["input_ids"].shape[1] :]
            outs.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return [o.strip() for o in outs]

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def unload(self):
        """Free GPU memory. Idempotent: safe to call more than once (e.g. when a cell is re-run)."""
        import torch

        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MockLLM:
    """Deterministic fake LLM for CPU tests.

    Behaviour: if an injected instruction containing a canary token is visible in the prompt context and the
    prompt is NOT in attribution mode, it 'obeys' with probability-free rule (obeys when the payload contains
    the word 'reply' or 'begin'); otherwise it answers with the first sentence of the first context passage.
    This only exercises plumbing; numbers produced with it are meaningless.
    """

    model_id = "mock"
    _canary = re.compile(r"\b(CNRY-[A-Z0-9]{6})\b")

    def chat_to_prompt(self, messages: list[dict]) -> str:
        return "\n".join(f"[{m['role']}] {m['content']}" for m in messages)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompts, max_new_tokens=256, temperature=0.0, top_p=1.0, batch_size=4):
        outs = []
        for p in prompts:
            ctx = p.split("Context", 1)[-1]
            m = self._canary.search(ctx)
            attribution = "cite" in p.lower() and "every sentence" in p.lower()
            if m and not attribution and re.search(r"\b(reply|begin)\b", ctx, re.I):
                outs.append(f"{m.group(1)}")
                continue
            passages = re.findall(r"\[(\d+)\]\s*(.+)", ctx)
            if passages:
                n, txt = passages[0]
                sent = re.split(r"(?<=[.!?])\s", txt.strip())[0]
                outs.append(f"{sent} [{n}]")
                continue
            # Un-numbered context (Baseline 2 format): echo the first sentence of the context.
            body = ctx.split(":", 1)[-1].strip() if ":" in ctx else ctx.strip()
            if body and body not in ("(no external context provided)",):
                sent = re.split(r"(?<=[.!?])\s", body)[0]
                outs.append(sent[:200])
            else:
                outs.append("I don't know.")
        return outs

    def unload(self):
        pass
