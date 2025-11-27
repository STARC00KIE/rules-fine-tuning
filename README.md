# LLM 파인튜닝 계획

# 1. 목표
- 전문 LLM 파인튜닝 및 경량 모델(SLM) 효율성 검증

---

# 2. 개요
- 학습 대상: 사내 규정 PDF
- 활용 인프라: .02(QA data 생성), .48(파인튜닝) 서버 활용
- 수행 내용
  - PDF 텍스트 추출 및 오류 전처리
  - 학습용 QA(질문-답변) 데이터셋 구축
  - QLoRA 기반 Qwen3-8B 파인튜닝 및 벤치마킹

---

# 3. 실행 방안
# 3.1 학습 데이터셋 구축
- 비정형 데이터 구조화: PDF 파싱 툴을 활용하여 텍스트 추출 후 오류 전처리
- 계층적 청킹: 문서의 목차, 장, 절과 같은 계층적 구조를 파악 후 청킹 진행

# 3.2 QA 데이터셋 생성	
- 합성 데이터 생성: Qwen3-Coder-30B-A3B-Instruct를 활용, 텍스트 데이터를 입력받아 예상 질문 생성
- 질문 유형 다변화:
  - 단순 조회형(150): 숫자, 기간 등 명시적 정보 (예: "경조사 휴가 일수는?")
  - 절차형(100): 신청 방법, 제출 서류 등 프로세스 (예: "육아휴직 신청 절차는?")
  - 부정형(50): 규정에 없는 내용에 대해 "없음" 답변 유도

# 3.3 학습 전략
- 데이터셋 분할: 과적합 방지 위해 데이터셋을 8:1:1로 분할
- 알고리즘: QLoRA (Quantized Low-Rank Adaptation) 적용 (GPU 메모리 효율화)
- 라이브러리: Unsloth 활용 (학습 최적화)
- 프롬프트 엔지니어링:
  - 시스템 프롬프트: "인사 규정 전문가" 페르소나 부여
  - Loss Masking: 모델의 사용자 답변(Output) 영역만 학습하도록 설정
- 벤치마크: 더 좋은 LLM으로 평가 진행(LLM as a Judge)

---

# 4. 추진 일정
- 데이터셋 구축 및 생성: 2025. 11. 24. ~ 2025. 11. 28.
  - **PDF 텍스트 추출 및 오류 전처리**: 2025. 11. 24. ~ 2025. 11. 24.
  - **계층적 청킹**: 2025. 11. 25. ~ 2025. 11. 25.
  - **합성 데이터 생성**: 2025. 11. 25. ~ 2025. 11. 28.
- LLM 파인튜닝: 2025. 12. 01. ~ 2025. 12. 05.
  - **학습**: 2025. 12. 01. ~ 2025. 12. 02.
  - **벤치마킹**: 2025. 12. 02. ~ 2025. 12. 05.

---

# 5. 파일 설명
## 01_pdf_parse.ipynb
- PDF 텍스트 추출 및 오류 전처리

## 02_txt_chunking.ipynb
- pdf에서 추출한 텍스트를 계층적 청킹 및 토큰 수 균등하게 분할

## 03_qa_extraction.py
- 합성 데이터 생성: Langchain과 OpenAI, vLLM, Qwen3-Coder-30B-A3B-Instruct를 활용하여 텍스트 데이터를 입력받아 예상 질문 생성

## 04_qa_validation.ipynb
- 합성 데이터 검증: 합성 데이터의 질문-답변 쌍이 정확한지 확인

---

# 6. 실행 방법
## 실행 환경
- Linux
- Python 3.10
- Miniconda
- jpype 1.5.2

## 01_pdf_parse.ipynb
- 라이브러리 설치
  - pip install pymupdf
- 실행
  - jupyter notebook 01_pdf_parse.ipynb

## 02_txt_chunking.ipynb
- 라이브러리 설치
  - 없음
- 실행
  - jupyter notebook 02_txt_chunking.ipynb

## 03_qa_extraction.py
- 라이브러리 설치
  - pip install langchain-openai
  - pip install langchain-core
  - pip install pydantic
  - pip install tqdm
  - pip install openai
- 실행
  - python 03_qa_extraction.py

## 04_qa_validation.ipynb
- 라이브러리 설치
  - pip install konlpy
  - pip install scikit-learn
  - pip install numpy
  - pip install pandas
- 실행
  - jupyter notebook 04_qa_validation.ipynb