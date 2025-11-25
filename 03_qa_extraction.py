import os
import json
import re
import math
from tqdm import tqdm
from typing import List, Optional

# LangChain 관련 기능 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

"""
# [설정] 경로 및 vLLM 서버 정보
"""
# 컨텍스트 데이터 주소와 출력 파일명
DATA_DIR = "./data/chunking_chapters_len_preprocess_final"
OUTPUT_FILE = "./data/QA/qwen3-coder-A3B-instruct/qa_dataset.json"

# vLLM 서버 정보
OPENAI_BASE_URL = "http://localhost:8001/v1" # vLLM 서버 주소, 앞으로 보고 바꾸면 될듯?
OPENAI_API_KEY = "EMPTY" # 로컬에서 실행할 거라 비어있음 처리
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct" # 사용할 모델 이름

# 생성할 데이터 종류 및 개수 총 300개(150, 100, 50) -> 일단 처음에 잘 생성되는지 6, 4, 2로 실행
TOTAL_TARGETS = {
    "simple": 6,      # 단순 조회형
    "procedural": 4,  # 절차형
    "negative": 2      # 부정형
}

"""
# [구조 정의] llm이 출력해야 할 JSON 형식에 대한 정의
"""
class QAItem(BaseModel): # 설계도
    category: str = Field(description="질문 유형 (simple, procedural, negative 중 하나)")
    question: str = Field(description="생성된 질문 텍스트")
    answer: str = Field(description="질문에 대한 답변 텍스트")
    source: str = Field(description="답변의 근거가 되는 문장 혹은 '규정 없음' 등의 근거")

"""
# [유틸리티] 파일 로드 및 텍스트 전처리
"""
def clean_text(text): # 이미 진행했지만 혹시 모르니까 추가
    """불필요한 공백이나 특수문자 정리"""
    return re.sub(r'\s+', ' ', text).strip()

def load_files():
    """폴더 내의 txt 파일들을 읽어서 딕셔너리 {파일명: 내용} 형태로 반환"""
    context_dict = {}

    # 파일 가져오기
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])
    print(f"\n데이터 로드 중 ({len(files)}개 파일)")
    
    # 딕셔너리로 `파일명: 내용`` 형태로 저장
    for f_name in files:
        file_path = os.path.join(DATA_DIR, f_name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = clean_text(f.read())
            if content: # 내용이 있는 경우만 저장
                context_dict[f_name] = content
                print(f"   - {f_name}: {len(content)}자 (OK)")
    return context_dict

"""
# [Langchain 설정] 프롬프트 + 모델 + Parser 정의
"""
def create_qa_chain():
    # 1. 모델 초기화
    llm = ChatOpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.7, # 기본값이 0.7이고, 나중에 0.1 로 수정필요
        max_tokens=4096, # 최대 출력 토큰 수, 로컬은 토큰 압박 없기 때문에 최대한 크게 설정
    )

    # 2. 파서 설정
    parser = JsonOutputParser(pydantic_object=QAItem)

    # 3. 프롬프트 템플릿 작성 (이상하면 바꾸기)
    template_str = """You are an Expert HR Regulation Specialist and AI Dataset Generator.
Your goal is to generate **1 high-quality Korean Q&A pair** to help employees understand company regulations.

### Context (Company Regulations):
{context}

### Strict Constraints:
1. **Source of Truth**: You must generate questions and answers based **ONLY** on the provided context. Do not use outside knowledge or general labor laws.
2. **Handling Missing Info**: If the instruction asks about a topic NOT in the context, your answer must explicitly state that "Such information is not found in the regulations" or "It is not specified." Do NOT invent fake rules.
3. **Completeness**: The ANSWER must be a complete, polite Korean sentence.
4. **Clarity**: The QUESTION should be self-contained and clear without looking at the context.

### Task Requirements:
- **Type**: {type_desc}
- **Instruction**: {instruction}
- **Example Question**: {ex_q}
- **Example Answer**: {ex_a}
- **Language**: Korean (한국어)

### Output Format:
{format_instructions}

IMPORTANT: Return ONLY the JSON object properly formatted.
"""
    prompt = PromptTemplate(
        template=template_str,
        input_variables=["context", "type_desc", "instruction", "ex_q", "ex_a"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 4. 체인 연결 (LCEL 문법: 프롬프트 -> 모델 -> 파서)
    chain = prompt | llm | parser
    return chain

"""
# [실행 로직] 데이터 생성 및 저장
"""
# 유형펼 프롬프트 설정값
TYPE_CONFIG = {
    "simple": {
        "desc": "Simple Fact Retrieval",
        "instruction": "규정에 명시된 숫자, 날짜, 기간, 정의, 조건을 묻는 질문.",
        "ex_q": "신규 채용된 직원은 채용일로부터 며칠 이내에 관련 서류를 제출해야 합니까?",
        "ex_a": "신규 채용된 직원은 채용 후 10일 이내에 서류를 제출해야 합니다.",
    },
    "procedural": {
        "desc": "Procedural/Process",
        "instruction": "특정 상황에서의 절차, 제출 서류, 승인 과정 등을 묻는 질문.",
        "ex_q": "예비군 휴가를 신청하려면 어떤 절차를 거쳐야 합니까?",
        "ex_a": "예비군 휴가를 신청하려면 훈련 증빙 서류를 첨부하여 상사의 승인을 받은 후 인사담당자에게 제출해야 합니다.",
    },
    "negative": {
        "desc": "Negative/Non-existent",
        "instruction": "규정에 없는 내용을 묻거나 틀린 전제를 확인하는 질문. 답변은 '규정에 없습니다' 등으로 시작.",
        "ex_q": "회사에서 직원들에게 기숙사를 제공합니까?",
        "ex_a": "제공된 취업규칙에는 기숙사 제공에 관한 조항이 포함되어 있지 않습니다."
    }
}

# 데이터 생성 함수
def generate_dataset(context_dict):
    chain = create_qa_chain() # 체인 생성
    dataset = []
    
    total_len = sum(len(t) for t in context_dict.values())
    if total_len == 0:
        print("생성할 데이터가 없습니다.")
        return []

    print(f"\n생성 시작 (총 텍스트: {total_len}자, 목표: 300개)")
    
    for fname, text in context_dict.items():
        # 파일 길이에 비례하여 할당량 계산
        ratio = len(text) / total_len
        print(f"\nProcessing [{fname}]...")
        
        for cat, target in TOTAL_TARGETS.items():
            count = max(1, math.floor(target * ratio)) # 최소 1개 보장
            config = TYPE_CONFIG[cat]
            
            # tqdm으로 진행상황 표시
            for _ in tqdm(range(count), desc=f"   - {cat}"):
                # 체인 실행
                result = chain.invoke({
                    "context": text,
                    "type_desc": config['desc'],
                    "instruction": config['instruction'],
                    "ex_q": config['ex_q'],
                    "ex_a": config['ex_a']
                })
                    
                # 출처 파일명 추가
                result['source_file'] = fname
                dataset.append(result)

    return dataset

"""
# 메인 실행부
"""
if __name__ == "__main__":
    # 1. 파일 로드
    context_data = load_files()
    
    if context_data:
        # 2. 생성
        final_data = generate_dataset(context_data)
        
        # 3. 저장
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
            
        print(f"\n총 {len(final_data)}개의 데이터셋이 '{OUTPUT_FILE}'에 저장되었습니다.")