# test_hf_sglang_logprob_unittest.py
import os
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import sglang as sgl

# ------------------------- Configurable via env ------------------------- #
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
PROMPTS = [
    "Hello, my name is",
    "The future of AI is",
    "The president of the United States is",
    "The capital of France is ",
]
TOP_LOGPROBS_NUM = int(os.environ.get("TOP_LOGPROBS_NUM", "50"))
FIRST_N_TOKEN_IDS = int(os.environ.get("FIRST_N_TOKEN_IDS", "10"))
RTOL = float(os.environ.get("RTOL", "0.20"))
ATOL = float(os.environ.get("ATOL", "0.00"))
# ------------------------------------------------

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def extract_sglang_outputs(output):
    """
    Normalize SGLang response to the fields we need.
    Adjust the keys here if your local build differs.
    """
    meta = output["meta_info"]
    return (
        meta["input_token_logprobs"],
        meta["input_top_logprobs"],
        meta["input_token_ids_logprobs"],
        meta["output_token_original_logprobs"],
        meta["output_top_original_logprobs"],
        meta["output_token_ids_original_logprobs"],
        [a for _, a, _ in meta["output_token_original_logprobs"]],
    )


class TestHFVsSGLangLogprob(unittest.TestCase):
    def setUp(self):
        # ----- HF side (float32 weights) -----
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, device_map="auto"
        ).eval()

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

            (
                input_token_logprobs,
                input_top_logprobs,
                input_token_ids_logprobs,
                output_token_logprobs,
                output_top_logprobs,
                output_token_ids_logprobs,
                output_ids_sglang,
            ) = extract_sglang_outputs(outputs)

            # ---------- 3) Build SGLang tensors ----------
            sgl_token_logprobs_tensor = torch.tensor(
                [a for a, _, _ in output_token_logprobs], device=self.hf_model.device
            )
            sgl_indices_tensor = torch.tensor(
                [b for _, b, _ in output_token_logprobs], device=self.hf_model.device
            )

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
