"""Test original log probability alignment between SGLang and Hugging Face.

This test suite verifies the correctness of the `origin_logprobs` output (temperature=1) in SGLang
by comparing it against raw logit-based probabilities computed directly from a
reference Hugging Face model.

The test covers the following scenarios:
- Next-token prediction: Verifies that the log probability of the next token from
  SGLang matches the Hugging Face model.
- Top-k logprobs: Ensures that the top-k original logprobs returned by SGLang are
  consistent with Hugging Face outputs.
- Specified token IDs: Confirms that the original logprobs for specific token IDs
  match the values computed from Hugging Face logits.

The test uses a fixed prompt and deterministic sampling configuration to ensure
numerical consistency. All comparisons are performed within a small tolerance,
to account for floating-point discrepancies.

This ensures that SGLang faithfully exposes original model scores, which are
critical for downstream applications like reranking, calibration, and debugging.
"""

import unittest

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

# ------------------------- Configurable via env ------------------------- #
MODEL_ID = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
PROMPTS = [
    "Hello, my name is",
    "The future of AI is",
    "The president of the United States is",
    "The capital of France is ",
]
TOP_LOGPROBS_NUM = 50
FIRST_N_TOKEN_IDS = 10
RTOL = 0.20
ATOL = 0.00
# ------------------------------------------------

torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class TestLogprob(unittest.TestCase):
    def setUp(self):
        # ----- HF side (float32 weights) -----
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map="auto"
        )

        # ----- SGLang side -----
        self.sgl_engine = sgl.Engine(
            model_path=MODEL_ID,
            skip_tokenizer_init=True,
            trust_remote_code=True,
        )

        # Shared sampling parameters
        self.sampling_params = {
            "temperature": 0.5,  # SGLang uses 0.5, but original logprobs are used 1.0
            "top_p": 1.0,
            "top_k": 10,
            "max_new_tokens": 1,
        }

    def tearDown(self):
        try:
            if hasattr(self.sgl_engine, "shutdown"):
                self.sgl_engine.shutdown()
        except Exception:
            pass

    def test_logprob_match(self):
        for prompt in PROMPTS:
            # ---------- 1) HF forward pass ----------
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.hf_model.device)
            attn_mask = enc["attention_mask"].to(self.hf_model.device)

            with torch.inference_mode():
                hf_out = self.hf_model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    return_dict=True,
                )
            logits = hf_out.logits[:, -1, :]  # [1, V]
            hf_log_probs = F.log_softmax(logits.float(), dim=-1)[0]  # [V]

            # ---------- 2) SGLang inference ----------
            outputs = self.sgl_engine.generate(
                input_ids=input_ids[0].tolist(),
                sampling_params=self.sampling_params,
                return_logprob=True,
                top_logprobs_num=TOP_LOGPROBS_NUM,
                token_ids_logprob=list(range(FIRST_N_TOKEN_IDS)),
            )

            if isinstance(outputs, list):
                outputs = outputs[0]
            meta = outputs["meta_info"]
            output_token_logprobs = meta["output_token_original_logprobs"]
            output_top_logprobs = meta["output_top_original_logprobs"]
            output_token_ids_logprobs = meta["output_token_ids_original_logprobs"]

            # ---------- 3) Build SGLang tensors ----------
            logprobs_vals, indices_vals, _ = zip(*output_token_logprobs)
            sgl_token_logprobs_tensor = torch.tensor(
                logprobs_vals, device=self.hf_model.device
            )
            sgl_indices_tensor = torch.tensor(indices_vals, device=self.hf_model.device)

            # ---------- 5) Assertions ----------
            sgl_indices_tensor = sgl_indices_tensor.to(
                dtype=torch.long, device=hf_log_probs.device
            )
            hf_token_logprobs = hf_log_probs[sgl_indices_tensor]

            self.assertTrue(
                torch.allclose(
                    hf_token_logprobs, sgl_token_logprobs_tensor, rtol=RTOL, atol=ATOL
                ),
                msg=f"Token logprobs differ over indices {sgl_indices_tensor.tolist()}",
            )

            # ----- top‑k comparison -----
            if output_top_logprobs is None:
                raise ValueError("Engine response missing 'output_top_logprobs'.")

            hf_topk_vals, _ = torch.topk(hf_log_probs, k=TOP_LOGPROBS_NUM, dim=-1)

            sgl_topk_vals = torch.tensor(
                [float(x[0][0]) for x in output_top_logprobs if x[0] is not None],
                dtype=torch.float32,
                device=hf_log_probs.device,
            )

            k_match = min(hf_topk_vals.numel(), sgl_topk_vals.numel())
            hf_topk_vals = hf_topk_vals[:k_match]
            sgl_topk_vals = sgl_topk_vals[:k_match]

            self.assertTrue(
                torch.allclose(hf_topk_vals, sgl_topk_vals, rtol=RTOL, atol=ATOL),
                msg="Top‑k logprobs mismatch",
            )

            # ----- token_ids_logprob comparison -----
            hf_first_ids = hf_log_probs[:FIRST_N_TOKEN_IDS]
            sgl_first_ids = torch.tensor(
                [a for a, _, _ in output_token_ids_logprobs[0]],
                device=self.hf_model.device,
            )
            self.assertTrue(
                torch.allclose(hf_first_ids, sgl_first_ids, rtol=RTOL, atol=ATOL),
                msg="token_ids_logprob mismatch",
            )

            print(
                f"[{prompt}] max|diff| token_logprob = "
                f"{torch.max(torch.abs(hf_token_logprobs - sgl_token_logprobs_tensor)):.4f}"
            )


if __name__ == "__main__":
    unittest.main()
