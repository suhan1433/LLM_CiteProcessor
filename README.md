# SelectiveCiteProcessor
SelectiveCiteProcessor는 프롬프트 중 사용자가 지정한 특정 부분만 인용하도록 유도하여, LLM이 정확하고 일관된 답변을 생성할 수 있도록 지원하는 LogitsProcessor입니다

## 개요
답변 생성 시, 특정 정보(예: 전화번호, 고유 명사 등)를 변형 없이 포함해야 하는 경우가 존재합니다.
하지만 LLM은 확률적으로 텍스트를 생성하기 때문에, 해당 정보가 왜곡되거나 누락되는 문제가 발생할 수 있습니다.

이를 해결하기 위해, 프롬프트 중 사용자가 지정한 특정 구간(e.g. Chunks)의 토큰 ID를 추출한 뒤, 해당 토큰들의 logit 값에 가중치(Boost)를 적용합니다.
이를 통해 모델은 지정된 정보를 우선적으로 참고하고, 정확하게 답변에 포함하도록 유도할 수 있습니다.

이 방식은 중요 정보를 정확하게 포함해야 하는 테스크에서 효과적인 결과를 보였습니다.

## 예시
```python
--model_name "Qwen/Qwen2-0.5B-Instruct" 
--device cpu 
--boost_factor 2.5 
--max_length 512 
--temperature 0.8 
--top_p 0.85 
--system_prompt "당신은 회사 내 직원들의 질문에 답변하는 AI 도우미입니다. 아래 참고 문서를 기반으로 질문에 대해 정확하고 친절하게 답변해 주세요. 문서에 기반한 내용 외에는 추측하지 마세요.." 
--user_query "연차 신청은 어디서 하나요?" 
--chunks "연차는 그룹웨어 시스템을 통해 신청할 수 있다. 로그인 후 '근태관리 > 휴가신청' 메뉴에서 작성하면 됨. 승인 여부는 팀장이 검토한 후 알림으로 전달됨. 연차 사용 내역은 마이페이지에서 확인 가능." 
--content_template "{user_query}\n\n[참고 문서]\n{chunks}\n\n위 내용을 참고해서 사용자 질문에 친절하고 정확하게 답변해 주세요."


--- SelectiveCiteProcessor 미적용 ---
죄송합니다, 저는 인공지능의 모델로 인식되지 않으며 현재 지원된 옵션이나 서비스에 대한 정보를 제공하기 어렵습니다. 직접 사용하여 연차 신청 또는 이메일을 보내주시면 감사하겠습니다.

--- SelectiveCiteProcessor 적용 ---
연차 신청은 그룹웨어 시스템을 통해 사용할 수 있습니다. 로그인 후 '근태관리 > 휴가신청' 메뉴에서 작성하면 됩니다. 승인 여부는 팀장이 검토한 후 알림으로 전달되며, 연차 사용 내역은 마이페이지에서 확인할 수 있습니다. 사용자에게는 이메일 또는 메시지 메뉴에서 연차 신청을 확인할 수 있습니다.
```

## 실행
```
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
```
## 설명
```python
class SelectiveCiteProcessor(LogitsProcessor):
    """
    LogitsProcessor that boosts the probability of tokens from the reference chunk.
    """
    def __init__(self, tokenizer: AutoTokenizer, chunk_token_ids: torch.Tensor, boost_factor: float = 1.0):
        self.tokenizer = tokenizer
        self.chunk_token_ids = chunk_token_ids
        self.boost_factor = boost_factor

    def __call__(self, input_ids: torch.Tensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        return self._cite_tokens(logits)

    def _cite_tokens(self, logits: torch.FloatTensor) -> torch.FloatTensor:
        """
        Boost the logits of tokens present in the chunk to encourage citation.
        """
        vocab_size = logits.shape[1]
        batch_size = logits.shape[0]
        for i in range(batch_size):
            chunk_tokens = set(self.chunk_token_ids[i].tolist()) # 설정한 구간의 토큰들의 Ids값 저장
            chunk_tokens.add(self.tokenizer.eos_token_id) # <eos>의 경우도 다른 토큰 Logit값이 올라가면서 무시되지 않도록 Boost
            chunk_tokens = [t for t in chunk_tokens if t < vocab_size] 
            logits[i, chunk_tokens] += self.boost_factor # 다음 생성 후보군 Ids의 Logit을 Boost
        return logits
```

## Reference
https://github.com/NVIDIA/logits-processor-zoo
