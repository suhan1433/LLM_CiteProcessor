"""
Selective Citation Text Generation
========================================

This script demonstrates how to use a custom LogitsProcessor to boost the probability of reference (chunk) tokens during text generation with a HuggingFace CausalLM model.

Usage (from command line):
    python3 selective_cite_processor.py \
        --model_name "Qwen/Qwen2-0.5B-Instruct" \
        --device cpu \
        --boost_factor 2.5 \
        --max_length 512 \
        --temperature 0.8 \
        --top_p 0.85 \
        --user_query "연차 신청은 어디서 하나요?" \
        --chunks "연차는 그룹웨어 시스템을 통해 신청할 수 있다. 로그인 후 '근태관리 > 휴가신청' 메뉴에서 작성하면 됨. 승인 여부는 팀장이 검토한 후 알림으로 전달됨. 연차 사용 내역은 마이페이지에서 확인 가능." \
        --system_prompt "당신은 회사 내 직원들의 질문에 답변하는 AI 도우미입니다. 아래 참고 문서를 기반으로 질문에 대해 정확하고 친절하게 답변해 주세요. 문서에 기반한 내용 외에는 추측하지 마세요.." \
        --content_template "{user_query}\n\n[참고 문서]\n{chunks}\n\n위 내용을 참고해서 사용자 질문에 친절하고 정확하게 답변해 주세요."
Or edit the parameters in the main() function below.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import argparse
from typing import Optional
import numpy as np
import random

seed = 80
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cite_tokens(
    tokenizer: AutoTokenizer,
    logits: torch.FloatTensor,
    chunk_token_ids: torch.Tensor,
    boost_factor: float = 1.0
) -> torch.FloatTensor:
    """
    Boost the logits of tokens present in the chunk to encourage citation.
    """
    vocab_size = logits.shape[1]
    batch_size = logits.shape[0]
    for i in range(batch_size):
        chunk_tokens = set(chunk_token_ids[i].tolist())
        chunk_tokens.add(tokenizer.eos_token_id)
        chunk_tokens = [t for t in chunk_tokens if t < vocab_size]
        logits[i, chunk_tokens] += boost_factor
    return logits


class SelectiveCiteProcessor(LogitsProcessor):
    """
    LogitsProcessor that boosts the probability of tokens from the reference chunk.
    """
    def __init__(self, tokenizer: AutoTokenizer, chunk_token_ids: torch.Tensor, boost_factor: float = 1.0):
        self.tokenizer = tokenizer
        self.chunk_token_ids = chunk_token_ids
        self.boost_factor = boost_factor

    def __call__(self, input_ids: torch.Tensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        return cite_tokens(self.tokenizer, logits, self.chunk_token_ids, self.boost_factor)


def run_inference(
    model_name: str,
    chunks: str,
    user_query: str,
    system_prompt: str,
    content_template: str,
    device: Optional[str] = None,
    boost_factor: float = 1.0,
    max_length: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95
) -> str:
    """
    Run inference with citation boosting on a CausalLM model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        model.to(device)
    except Exception as e:
        print(f"[ERROR] Failed to load model or tokenizer: {e}")
        raise

    user_content = content_template.format(user_query=user_query, chunks=chunks)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]

    chunk_token_ids = tokenizer.encode(chunks, return_tensors="pt").to(device)
    selective_cite_processor = SelectiveCiteProcessor(tokenizer, chunk_token_ids, boost_factor=boost_factor)

    gen_kwargs = {
        "max_length": max_length,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "logits_processor": LogitsProcessorList([selective_cite_processor]),
    }

    print("[INFO] Generation parameters:")
    print(f"  boost_factor: {boost_factor}")
    print(f"  max_length: {max_length}")
    print(f"  temperature: {temperature}")
    print(f"  top_p: {top_p}")
    print(f"  user_query: {user_query}")
    print(f"  chunks: {chunks}")
    print(f"  system_prompt: {system_prompt}")
    print(f"  content_template: {content_template}")

    output = model.generate(input_ids, **gen_kwargs)
    generated_text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Citation-Boosted Text Generation Experiment")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2-0.5B-Instruct", help='HuggingFace model name')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu)')
    parser.add_argument('--boost_factor', type=float, default=1.0, help='Logit boost factor for chunk tokens')
    parser.add_argument('--max_length', type=int, default=200, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling')
    parser.add_argument('--user_query', type=str, default="오케스트로에 대해 알려줘", help='User query')
    parser.add_argument('--chunks', type=str, default="오케스트로란 클라우드 회사다, 파크원에 위치해 있다, 전화번호는 111-111이다", help='Reference chunk text')
    parser.add_argument('--system_prompt', type=str, default="아래의 내용을 참고해서 답변을 해줘", help='System prompt for the model')
    parser.add_argument('--content_template', type=str, default="{user_query}\n\n참고: {chunks}", help='Template for user content. Use {user_query} and {chunks} as placeholders.')
    args = parser.parse_args()

    result = run_inference(
        model_name=args.model_name,
        chunks=args.chunks,
        user_query=args.user_query,
        system_prompt=args.system_prompt,
        content_template=args.content_template,
        device=args.device,
        boost_factor=args.boost_factor,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )
    print("\n--- 생성된 텍스트 ---")
    print(result)

if __name__ == "__main__":
    main()
