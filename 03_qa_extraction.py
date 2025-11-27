import os
import json
import re
import math
from tqdm import tqdm
from typing import List, Optional
import logging
import time
from datetime import datetime

# LangChain 관련 기능 임포트
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

"""
# [설정] 경로 및 vLLM 서버 정보
"""
# 컨텍스트 데이터 주소와 출력 파일명
DATA_DIR = "./data/chunking_chapters_len_preprocess_final"
OUTPUT_FILE = "./data/QA/qwen3-coder-A3B-instruct/qa_dataset.json" # 기본 파일명 (온도에 따라 이름 변경 예정)

# --- 로그 파일명 명명 규칙 적용 ---
# 1. 현재 시각을 YYYYMMDD_HHMMSS 형식으로 포맷
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# 2. 로그 파일명 설정: qa_extraction_YYYYMMDD_HHMMSS.log
LOG_FILE = f"./logs/03_qa_extraction_{timestamp}.log"
# -----------------------------------

# vLLM 서버 정보
OPENAI_BASE_URL = "http://localhost:8001/v1" # vLLM 서버 주소, 앞으로 보고 바꾸면 될듯?
OPENAI_API_KEY = "EMPTY" # 로컬에서 실행할 거라 비어있음 처리
MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct" # 사용할 모델 이름

# 생성할 데이터 종류 및 개수 총 300개(150, 100, 50) -> 일단 처음에 잘 생성되는지 6, 4, 2로 실행
TOTAL_TARGETS = {
    "simple": 150, # 단순 조회형
    "procedural": 100,# 절차형
    "negative": 50, # 부정형
}

"""
# [로그 설정] 로거 초기화
"""
# 1. 로거 초기화
logging.basicConfig(
    level=logging.INFO, # INFO 레벨 이상만 기록
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'), # 파일로 저장 (덮어쓰기), 파일 이름 수정해야 함
        logging.StreamHandler() # 콘솔에도 출력
    ]
)
logger = logging.getLogger(__name__) # 스크립트 이름으로 로거 생성
logger.info(f"로깅 시작. 로그 파일 경로: {LOG_FILE}")

"""
# [구조 정의] llm이 출력해야 할 JSON 형식에 대한 정의
"""
class QAItem(BaseModel): # 설계도
    category: str = Field(description="질문 유형 (simple, procedural, negative 중 하나)")
    question: str = Field(description="생성된 질문 텍스트")
    answer: str = Field(description="질문에 대한 답변 텍스트")
    source: str = Field(description="답변의 근거가 되는 문장 혹은 '규정 없음' 등의 근거")

# 새로운 최상위 모델: 여러 개의 QAItem을 리스트로 담는 구조
class QAList(BaseModel): # <--- 추가
    qa_pairs: List[QAItem] = Field(description="생성된 질문-답변 쌍 리스트")

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
    logger.info(f"📁 데이터 로드 시작: {DATA_DIR}에서 {len(files)}개 파일 발견") # <-- 로깅
    
    # 딕셔너리로 `파일명: 내용`` 형태로 저장
    for f_name in files:
        file_path = os.path.join(DATA_DIR, f_name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = clean_text(f.read())
            if content: # 내용이 있는 경우만 저장
                context_dict[f_name] = content
                logger.debug(f"   - {f_name}: {len(content)}자 로드 완료") # <-- 디버그 로깅
            else:
                logger.warning(f"   - {f_name}: 내용이 비어있어 스킵합니다.") # <-- 경고 로깅
    return context_dict

"""
# [Langchain 설정] 프롬프트 + 모델 + Parser 정의
"""
def create_qa_chain(temperature: float): # <--- temperature를 인수로 받도록 수정
    logger.info(f"⚙️ LangChain 체인 생성 시작 (Temperature: {temperature:.1f})") # <-- 로깅
    # 1. 모델 초기화
    llm = ChatOpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        model_name=MODEL_NAME,
        temperature=temperature, # <--- 인수로 받은 temperature 사용
        max_tokens=4096,
    )

    # 2. 파서 설정
    # parser = JsonOutputParser(pydantic_object=QAItem)
    parser = JsonOutputParser(pydantic_object=QAList)

    # 3. 프롬프트 템플릿 작성
    # 부정형 answer가 영어로 출력되는지 확인 필요함
    template_str = """You are an Expert HR Regulation Specialist and AI Dataset Generator.
Your goal is to generate **{num_to_generate} high-quality Korean Q&A pairs** to help employees understand company regulations.

### Context (Company Regulations):
{context}

### Strict Constraints:
1. **Language Consistency (CRITICAL)**: Both QUESTION and ANSWER must be **STRICTLY generated in Korean (한국어)**. Do NOT use English.
2. **Source of Truth**: You must generate questions and answers based **ONLY** on the provided context. Do not use outside knowledge or general labor laws.
3. **Handling Missing Info (Negative QA)**: If the instruction asks about a topic NOT in the context, you must logically verify its absence and answer in Korean: "제공된 규정에 관련 내용이 없습니다." Do NOT invent fake rules.
4. **No Subjective Interpretation**: Do not infer priority (e.g., "most important"), intent, or subjective value unless explicitly written in the text. Stick to the facts presented.
5. **Completeness**: The ANSWER must be a complete, polite Korean sentence (e.g., ~합니다, ~입니다).
6. **Clarity**: The QUESTION should be self-contained and clear without looking at the context.
7. **Source Format**: The `source` field must contain the **full text segment** (Article number + Title + Content) that supports the answer. Do not abbreviate.

### Task Requirements:
- **Type**: {type_desc}
- **Instruction**: {instruction} (if any)
- **Example Question**: {ex_q}
- **Example Answer**: {ex_a}
- **Language**: Korean (한국어)

### Output Format:
{format_instructions}

IMPORTANT: Return ONLY the JSON object properly formatted.
"""

    # 이전에 사용했던 프롬프트: 부정형 answer가 영어로 출력되서 변경 진행
    # "Such information is not found in the regulations" or "It is not specified." 이 부분 때문에 부정형에서 영어로 출력됨
    before_template_str = """You are an Expert HR Regulation Specialist and AI Dataset Generator.
Your goal is to generate **{num_to_generate} high-quality Korean Q&A pair** to help employees understand company regulations.

### Context (Company Regulations):
{context}

### Strict Constraints:
1. **Source of Truth**: You must generate questions and answers based **ONLY** on the provided context. Do not use outside knowledge or general labor laws.
2. **Handling Missing Info**: If the instruction asks about a topic NOT in the context, your answer must explicitly state that "Such information is not found in the regulations" or "It is not specified." Do NOT invent fake rules.
3. **Completeness**: The ANSWER must be a complete, polite Korean sentence.
4. **Clarity**: The QUESTION should be self-contained and clear without looking at the context.
5. **Language Consistency**: Both QUESTION and ANSWER must be **STRICTLY** generated in **Korean (한국어)**.

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
        input_variables=["context", "type_desc", "instruction", "ex_q", "ex_a", "num_to_generate"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 4. 체인 연결 (LCEL 문법: 프롬프트 -> 모델 -> 파서)
    chain = prompt | llm | parser
    logger.info("✅ LangChain 체인 생성 완료") # <-- 로깅
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

# [실행 로직] 데이터 생성 및 저장 (수정됨)
def generate_dataset(context_dict, chain):
    # 전체 시작 시간 기록
    start_time_total = time.time()

    dataset = []
    total_len = sum(len(t) for t in context_dict.values())
    if total_len == 0:
        logger.error("🛑 생성할 컨텍스트 데이터가 없습니다.") # <-- 에러 로깅
        return []

    target_count = sum(TOTAL_TARGETS.values())
    logger.info(f"📊 데이터 생성 시작 (총 텍스트: {total_len}자, 목표: {target_count}개)") # <-- 로깅
    
    for fname, text in context_dict.items():
        # 파일별 시작 시간 기록
        start_time_file = time.time()

        ratio = len(text) / total_len
        logger.info("=======================================================")
        logger.info(f"파일 처리 시작: [{fname}] ({len(text)}자, 비율: {ratio:.2f})") # <-- 로깅
        
        for cat, target in TOTAL_TARGETS.items():
            count = max(1, math.floor(target * ratio)) # 이만큼을 한 번에 생성
            config = TYPE_CONFIG[cat]
            
            # tqdm을 제거하고 파일/유형별로 한 번만 호출하도록 변경
            logger.info(f"   -> 유형: {cat} / 목표: {count}개 일괄 생성 요청") # <-- 로깅 
            
            try:
                result_obj = chain.invoke({
                    "context": text,
                    "type_desc": config['desc'],
                    "instruction": config['instruction'],
                    "ex_q": config['ex_q'],
                    "ex_a": config['ex_a'],
                    "num_to_generate": count # 생성할 개수 전달
                })
            except Exception as e:
                # 사용자 요청(try-except 금지)에 따라 예외를 직접 처리하지는 않지만,
                # 로그를 남겨 오류 상황을 기록
                logger.error(f"❌ LLM 호출 중 오류 발생 - 파일: {fname}, 유형: {cat}. 오류: {e}")
                continue # 다음 유형으로 넘어갑니다.


            # ---  ERROR 방지 및 데이터 추출 로직  ---
            qa_data = []
            
            # 1. Pydantic 인스턴스일 경우 (가장 이상적인 경우)
            if hasattr(result_obj, 'qa_pairs'):
                qa_data = result_obj.qa_pairs
            # 2. 단순 딕셔너리일 경우 (AttributeError의 원인)
            elif isinstance(result_obj, dict):
                qa_data = result_obj.get('qa_pairs', [])
            
            generated_count = len(qa_data)
            logger.info(f"   <- 유형: {cat} / {generated_count}개 추출됨 (목표: {count})") # <-- 로깅
            
            # 추출된 리스트를 순회하며 최종 데이터셋에 추가
            for item in qa_data:
                # item이 딕셔너리(dict)일 수도 있으므로, 키 접근을 사용
                if isinstance(item, dict):
                    item['source_file'] = fname
                    dataset.append(item)
                # item이 Pydantic 객체일 경우 dict로 변환 후 저장
                elif hasattr(item, 'dict'):
                    item_dict = item.dict()
                    item_dict['source_file'] = fname
                    dataset.append(item_dict)

            # --- 파일별 종료 시간 기록 ---
            end_time_file = time.time()
            elapsed_time_file = end_time_file - start_time_file
            logger.info(f"⏳ [{fname}, {cat}] 처리 완료. 소요 시간: {elapsed_time_file:.2f}초")
            # -----------------------------

        # --- 전체 종료 시간 기록 ---
        end_time_total = time.time()
        elapsed_time_total = end_time_total - start_time_total
        logger.info(f"✅ 데이터 생성 완료. 총 {len(dataset)}개의 QA 쌍이 준비되었습니다.")
        logger.info(f"⏱️ **전체 데이터셋 생성 총 소요 시간: {elapsed_time_total:.2f}초**")
        # ---------------------------
    return dataset

"""
# 메인 실행부
"""
if __name__ == "__main__":
    # --- 전체 스크립트 실행 시작 시간 기록 ---
    script_start_time = time.time()
    # ----------------------------------------
    
    # 1. 파일 로드
    context_data = load_files()
    
    if not context_data:
        logger.error("🛑 Context 데이터가 없어 데이터 생성을 중단합니다.")
    else:
        # 온도 설정 (0.0부터 1.0까지 0.1씩 증가)
        # round(i * 0.1, 1)을 사용하여 부동 소수점 오류 방지
        temperatures = [round(i * 0.1, 1) for i in range(2, 11)]

        logger.info(f"총 {len(temperatures)}개의 온도 설정을 반복합니다: {temperatures}")
        
        for temp in temperatures:
            logger.info("=======================================================")
            logger.info(f"       DATA GENERATION START - TEMPERATURE: {temp:.1f}")
            logger.info("=======================================================")

            # 2. 체인 생성 (변경된 온도 적용)   
            qa_chain = create_qa_chain(temp)

            # 3. 데이터 생성
            final_data = generate_dataset(context_data, qa_chain)

            # 4. 저장 파일명 변경 (온도 값 포함)
            # 예: ./data/QA/.../qa_dataset_temp_0.1.json
            base_path, ext = os.path.splitext(OUTPUT_FILE)
            temp_output_file = f"{base_path}_temp_{temp:.1f}{ext}"
            
            # 5. 저장
            with open(temp_output_file, "w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"[SUCCESS] 총 {len(final_data)}개의 데이터셋이 '{temp_output_file}'에 저장되었습니다.")


        # --- 전체 스크립트 실행 종료 시간 기록 ---
        script_end_time = time.time()
        script_elapsed_time = script_end_time - script_start_time
        logger.info("=== 모든 온도 설정에 대한 데이터 생성이 완료되었습니다. ===")
        logger.info(f"🚀 **스크립트 전체 실행 시간 (시작~종료): {script_elapsed_time:.2f}초**")
        # ------------------------------------------  
        