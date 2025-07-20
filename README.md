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
model = "Qwen/Qwen2-0.5B-Instruct"
chunk = "오케스트로란 클라우드 회사다, 파크원에 있다, 전화번호 111-111"
messages = [
    {"role": "system", "content": "아래의 내용을 참고해서 답변을 해줘"},
    {"role": "user", "content": f"오케스트로에 대해 알려줘?\n\n참고: {chunk}"}
]

--- CiteProcessor 적용 ---
오케스트로는 파크원에서 주로 활동하는 클라우드 회사입니다. 회사는 파크원에 위치해 있으며, 이에 대한 전화번호인 111-111은 그 회사에 대한 정보나 관리에 관한 연락처입니다. 그러나 실제 회사명이나 전화번호를 알 수 없습니다. 다른 질문이 있으시면 언제든지 말씀해주세요.

--- CiteProcessor 미적용 ---
오케스트로는 "Orchestra"라는 이름의 대표적인 공연장입니다. 이곳은 일반적으로 전 세계에서 최고의 오케스트리들나 유명한 가수들이 그들의 작품들을 보여주며, 다양한 음악流派를 다룬 공연소입니다.
그러나 전화번호와 AI가 들어있는 것은 사실이 아닙니다. 이 정보를 찾기 위해 온라인 검색이나 전문가의建议 등을 통해 확인해보시는 것이 좋습니다. 하지만, 주로 제공되는 정보에 따르면, 오케스트로는 인공지능으로 운영되어 있고, 각각의 예술家들 및 그들의 애플리케이션들이 진행하는 공연장이 될 수 있습니다.
```

## 실행
```
python retrievePipe.py \
  --model_name "Qwen/Qwen2-0.5B-Instruct" \
  --device cuda \
  --boost_factor 2.0 \
  --max_length 256 \
  --temperature 0.9 \
  --top_p 0.85 \
  --user_query "오케스트로를 설명해줘." \
  --chunks "대표 서비스: 클라우드, AI \n회사 위치: 파크원" \
  --system_prompt "너는 아래 참고 내용을 바탕으로 답변해." \
  --content_template "질문: {user_query}\n\n[참고자료]\n{chunks}\n\n답변:"
```
## 

## Reference
https://github.com/NVIDIA/logits-processor-zoo
