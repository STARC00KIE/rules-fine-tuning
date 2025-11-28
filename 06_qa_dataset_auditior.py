import os
import json
import logging
import time
import datetime
import google.generativeai as genai
from google.generativeai import caching
from tqdm import tqdm
from config import Config  # config.py에서 설정 가져오기

# ---------------------------------------------------------
# 0. 로깅 설정 (Console + File)
# ---------------------------------------------------------
# 로그 디렉토리 생성
log_dir = os.path.dirname(Config.LOG_FILE_PATH)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE_PATH, encoding='utf-8'), # 파일 저장
        logging.StreamHandler() # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. 초기화 및 파일 로드 (No Try-Except)
# ---------------------------------------------------------
genai.configure(api_key=Config.API_KEY)

logger.info(f"파일 로드 중: {Config.DOC_FILE_PATH}")
with open(Config.DOC_FILE_PATH, "r", encoding="utf-8") as f:
    full_document_text = f.read()

logger.info(f"데이터셋 로드 중: {Config.DATA_FILE_PATH}")
with open(Config.DATA_FILE_PATH, "r", encoding="utf-8") as f:
    raw_dataset = json.load(f)

logger.info(f"로드 완료! (문서: {len(full_document_text)}자, 데이터: {len(raw_dataset)}개)")

# ---------------------------------------------------------
# 2. 컨텍스트 캐싱 생성 (비용 절감 핵심)
# ---------------------------------------------------------
logger.info("컨텍스트 캐싱 생성 중...")

cache_content = f"""
# Input Data Context (전체 규정 문서)
---
{full_document_text}
---
"""

try:
    # 캐시 생성 (TTL: 60분 유지)
    cache = caching.CachedContent.create(
        model=Config.MODEL_NAME,
        display_name='hr_regulation_cache',
        system_instruction=Config.SYSTEM_PROMPT,
        contents=[cache_content],
        ttl=datetime.timedelta(minutes=Config.CACHE_TTL_MINUTES),
    )
    logger.info(f"캐싱 완료! (Cache Name: {cache.name})")
    
    # 캐시된 모델 불러오기
    model = genai.GenerativeModel.from_cached_content(
        cached_content=cache,
        generation_config=Config.GENERATION_CONFIG  # 설정 파일에서 가져옴
    )

except Exception as e:
    logger.error(f"캐싱 생성 실패: {e}")
    exit()

# ---------------------------------------------------------
# 5. 데이터 검증 루프 실행
# ---------------------------------------------------------
final_dataset = []
stats = {"pass": 0, "revised": 0, "deleted": 0, "error": 0}
token_usage = {"prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "total_tokens": 0}

logger.info(f"검증 시작 (총 {len(raw_dataset)}건)...")

# 진행률 표시바와 함께 루프 실행
for index, item in enumerate(tqdm(raw_dataset)):
    
    # 모델에게 보낼 질문 구성 (Q&A 하나씩)
    user_query = f"""
    [검증 대상 Q&A]
    - Category: {item.get('category', 'unknown')}
    - Question: {item['question']}
    - Answer: {item['answer']}
    """
    
    try:
        # API 호출
        response = model.generate_content(user_query)

        # 토큰 사용량 집계 (메타데이터 활용)
        if response.usage_metadata:
            p_tokens = response.usage_metadata.prompt_token_count
            c_tokens = response.usage_metadata.candidates_token_count
            cached_count = response.usage_metadata.cached_content_token_count or 0
            
            token_usage["prompt_tokens"] += p_tokens
            token_usage["cached_tokens"] += cached_count
            token_usage["output_tokens"] += c_tokens
            token_usage["total_tokens"] += (p_tokens + c_tokens)

        result_text = response.text.strip()
        
        # ---------------------------------------------------
        # 결과 처리 로직
        # ---------------------------------------------------
        
        # [CASE 1] PASS (수정 없음) -> null 문자열 반환
        if result_text == "null":
            item['audit_status'] = "PASS"
            final_dataset.append(item)
            stats["pass"] += 1
            logger.debug(f"[PASS] Index {index}") 
            
        # [CASE 2] REVISE / DELETE (JSON 파싱 필요)
        else:
            # 마크다운 코드블록 제거
            clean_json_str = result_text.replace("```json", "").replace("```", "").strip()
            
            try:
                result_json = json.loads(clean_json_str)
                
                # 공통 변수 추출 (로깅 및 데이터 할당용)
                status = result_json.get('status')
                rationale = result_json.get('rationale', 'No rationale provided')
                
                if status == 'REVISE':
                    # 수정된 내용으로 업데이트
                    new_item = item.copy()
                    
                    if 'better_pair' in result_json and result_json['better_pair']:
                        new_item['category'] = result_json['better_pair'].get('category', item['category'])
                        new_item['question'] = result_json['better_pair'].get('question', item['question'])
                        new_item['answer'] = result_json['better_pair'].get('answer', item['answer'])
                    
                    new_item['audit_status'] = "REVISE"
                    new_item['audit_reason'] = rationale
                    new_item['source_citation'] = result_json.get('source_citation', '')
                    
                    final_dataset.append(new_item)
                    stats["revised"] += 1
                    # 수정: 변수로 추출한 rationale 사용
                    logger.info(f"[REVISE] Index {index}: {rationale}")
                    
                elif status == 'DELETE':
                    # 삭제 데이터는 저장하지 않고 넘어감
                    stats["deleted"] += 1
                    # 수정: 변수로 추출한 rationale 사용
                    logger.warning(f"[DELETE] Index {index}: {rationale}")
                    continue
                    
                else:
                    # JSON은 정상이지만 status 키가 없거나 이상한 값인 경우
                    item['audit_status'] = "UNKNOWN_STATUS" # 명확한 상태명으로 변경 권장
                    item['raw_response'] = result_text
                    final_dataset.append(item)
                    stats["error"] += 1
                    # 수정: je 변수 제거 (여기서는 파싱 에러가 아님)
                    logger.error(f"[UNKNOWN_STATUS] Index {index}: 알 수 없는 Status 값. | Raw: {result_text}")
                    
            except json.JSONDecodeError as e: # 수정: as e 추가
                # JSON 파싱 실패 시 원본 유지
                item['audit_status'] = "JSON_ERROR" # API 에러보다는 JSON 에러가 더 명확함
                item['error_message'] = str(e)
                item['raw_response'] = result_text # 디버깅용 원본 저장 추천
                final_dataset.append(item)
                stats["error"] += 1
                logger.error(f"[JSON_ERROR] Index {index}: {e} | Raw snippet: {clean_json_str[:50]}...", exc_info=True)
                time.sleep(1)

    except Exception as e:
        # 그 외 API 호출 에러 등
        item['audit_status'] = "API_ERROR"
        item['error_message'] = str(e) # 에러 메시지 저장 추천
        final_dataset.append(item)
        stats["error"] += 1
        logger.error(f"[API_ERROR] Index {index}: {e}", exc_info=True)
        time.sleep(1)

# ---------------------------------------------------------
# 4. 결과 저장
# ---------------------------------------------------------
# 디렉토리 존재 확인 및 생성 (os.makedirs는 에러가 아니므로 사용)
output_dir = os.path.dirname(Config.OUTPUT_FILE_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(Config.OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=4)

# 캐싱 효율 계산 (전체 입력 중 캐시 비중)
total_input = token_usage['prompt_tokens']
cached_input = token_usage['cached_tokens']
real_input = total_input - cached_input # 실제로 전송한 텍스트 (질문 등)
cache_rate = (cached_input / total_input * 100) if total_input > 0 else 0

# 최종 리포트 출력
report = f"""
===================================================
[데이터 검증 리포트]
===================================================
1. 실행 일시 : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
2. 검증 모델 : {Config.MODEL_NAME}
===================================================
[검증 결과 통계]
---------------------------------------------------
1. PASS (통과)    : {stats['pass']:>5} 건
2. REVISED (수정) : {stats['revised']:>5} 건
3. DELETED (삭제) : {stats['deleted']:>5} 건
4. ERROR (오류)   : {stats['error']:>5} 건
---------------------------------------------------
5. Total Processed : {len(raw_dataset):>5} 건
===================================================
[파일 경로 정보]
---------------------------------------------------
1. 입력 데이터 : {Config.DATA_FILE_PATH}
2. 결과 데이터 : {Config.OUTPUT_FILE_PATH}
3. 상세 로그   : {Config.LOG_FILE_PATH}
===================================================
[토큰 사용량 분석 (비용 예측)]
---------------------------------------------------
1. 총 입력 토큰   : {total_input:,}
   ├─ 캐시된 입력 : {cached_input:,}
   └─ 신규 입력   : {real_input:,}
---------------------------------------------------
2. 총 출력 토큰   : {token_usage['output_tokens']:,}
---------------------------------------------------
3. 합계 (Input+Output) : {token_usage['total_tokens']:,}
   └─ 캐싱 효율(절감률) : {cache_rate:.1f}%
===================================================
"""
print(report)

report_file_path = Config.OUTPUT_FILE_PATH.replace('.json', '_report.txt')
with open(report_file_path, "w", encoding="utf-8") as f:
    f.write(report)
logger.info(f"리포트 파일 저장 완료: {report_file_path}")

print(report)
logger.info("검증 프로세스 종료.")
