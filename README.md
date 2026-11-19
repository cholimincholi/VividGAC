# VividGAC

시각 정보로 오디오 캡셔닝을 유도하되, 실제 오디오로 확인된 단서만 전달해 오디오·비디오 질의응답을 수행합니다.

- `code/run_stage.py`: 질문 키워드 → 시각 캡션 → 후보 단서 → CLAP 근거 게이트 → 오디오 캡션 → 최종 추론의 단계별 파이프라인. CLAP 점수가 임계값 이상인 단서만 남기고, 전체 시각 캡션을 오디오 프롬프트에서 분리합니다. 통과한 단서가 없으면 일반 오디오 설명으로 전환합니다.
- `code/models.py`: VLM·오디오 모델·추론 LLM 래퍼와 CLIP·CLAP 정규화 임베딩 및 코사인 유사도 계산.
- `code/prompts.py`: 시각·오디오 캡셔닝 및 추론 프롬프트.
- `code/config.py`: 모델 위치, 생성 매개변수, 질문 관련성 임계값 `TAU_R`와 오디오 근거 임계값 `TAU_A`.
- `code/data.py`, `code/prepare_audio.py`: 질의응답 입력 형식 통일과 영상에서 16 kHz 모노 오디오 추출.

핵심 소스 발췌본입니다. 실행에는 별도 모델 가중치·입력 데이터와 PyTorch, Transformers, spaCy(`en_core_web_sm`), librosa, qwen-vl-utils, PyAV, SoundFile 등 사용 모델에 맞는 라이브러리가 필요합니다. 모델·데이터·출력 위치는 상대경로 또는 `VIVID_MODELS`, `VIVID_DATA`, `VIVID_OUTPUTS` 환경변수로 지정합니다.
