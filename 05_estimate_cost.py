import json
import os
import tiktoken
from tqdm import tqdm
from datetime import datetime
from config import Config

# ---------------------------------------------------------
# 1. 설정 및 초기화
# ---------------------------------------------------------
# 로그를 저장할 파일명 설정
# 로그 파일 경로(동적)
_current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_PATH = f"./logs/05_estimate_cost_{_current_time}.log"

# 로컬 토크나이저 로드
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens_local(text):
    """로컬에서 토큰 개수를 계산하는 함수"""
    if not text:
        return 0
    return len(tokenizer.encode(text))

def save_to_log(message):
    """화면 출력 및 파일 저장을 동시에 수행"""
    print(message)  # 화면 출력
    
    # 파일에 추가 모드('a')로 저장
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")

# ---------------------------------------------------------
# 2. 데이터 로드
# ---------------------------------------------------------
print(f"데이터셋 로드 중: {Config.DATA_FILE_PATH}")
with open(Config.DATA_FILE_PATH, "r", encoding="utf-8") as f:
    raw_dataset = json.load(f)

print(f"규정 문서 로드 중: {Config.DOC_FILE_PATH}")
with open(Config.DOC_FILE_PATH, "r", encoding="utf-8") as f:
    full_document_text = f.read()

# ---------------------------------------------------------
# 3. 토큰 계산 (Local Calculation)
# ---------------------------------------------------------
print("\n예상 토큰 사용량 계산 중 (Local - No API Call)...")

# 1) 시스템 프롬프트 + 문서 토큰
full_context = Config.SYSTEM_PROMPT + full_document_text
context_tokens = count_tokens_local(full_context)

# 2) 개별 Q&A 데이터 토큰 합산
total_qa_tokens = 0
estimated_output_tokens = 0

# tqdm은 진행률만 보여주므로 로그에 남기지 않음
for item in tqdm(raw_dataset):
    user_query = f"""
    [검증 대상 Q&A]
    - Category: {item.get('category', 'unknown')}
    - Question: {item['question']}
    - Answer: {item['answer']}
    """
    
    count = count_tokens_local(user_query)
    total_qa_tokens += count
    estimated_output_tokens += 300 

# ---------------------------------------------------------
# 4. 견적서 생성 및 로그 저장
# ---------------------------------------------------------
# 현재 시간
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 로그 메시지 포맷팅
log_message = f"""
==================================================
[{current_time}] 토큰 사용량 견적서
==================================================
1. 규정 문서(Context) 크기 : {context_tokens:,} Tokens
--------------------------------------------------
2. Q&A 데이터 입력 총합   : {total_qa_tokens:,} Tokens
3. 예상 출력(Output) 총합 : {estimated_output_tokens:,} Tokens
--------------------------------------------------
4. 총 처리 데이터 수      : {len(raw_dataset)} 건
==================================================
"""

# 화면 출력 및 파일 저장 실행
save_to_log(log_message)

print(f"\n견적 결과가 '{LOG_FILE_PATH}' 파일에 저장되었습니다.")