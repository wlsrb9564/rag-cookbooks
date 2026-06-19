# RAG Cookbooks - 학습 체크리스트

## 환경 세팅
- [x] uv venv 생성 (Python 3.11)
- [x] 의존성 설치 (265개 패키지)
- [x] Jupyter 커널 등록 (`RAG Cookbooks (Python 3.11)`)
- [ ] `.env` 파일 생성 (`.env.example` 참고)
  - `ANTHROPIC_API_KEY` (Claude 사용 시)
  - `OPENAI_API_KEY` (노트북 기본값)
  - `TAVILY_API_KEY` (agentic 노트북 필수)
  - `ATHINA_API_KEY` (평가 셀 사용 시)
  - `LANGCHAIN_API_KEY` (fusion_rag LangSmith 트레이싱)

---

## 스킵 목록 (실행 불가 / 불필요)

| 노트북 | 이유 |
|---|---|
| `advanced_rag/naive_rag` | Pinecone 클라우드 계정 필요 |
| `advanced_rag/hyde_rag` | Weaviate 클라우드 계정 필요 + `langchain-weaviate` 미설치 |
| `advanced_rag/basic_unstructured_rag` | Tesseract OCR 시스템 설치 필요, 매우 무거움 |
| `agentic/basic_agentic_rag` | Google Gemini API 키 + HuggingFace 모델 다운로드 필요 |
| `agentic/deepseek_qdrant` | DeepSeek API + HuggingFace 임베딩 모델 필요 |

---

## Advanced RAG (핵심 3개)

### 공통 패치 (모든 노트북)
```python
# 제거
from google.colab import userdata

# 추가 (맨 위 셀)
from dotenv import load_dotenv
load_dotenv()
```

### 1. `hybrid_rag.ipynb` ★★★
> BM25 키워드 검색 + 벡터 검색 앙상블 → 실무 가장 많이 쓰이는 기법

- [x] 공통 패치 적용
- [x] `pip install` 셀 삭제 (이미 설치됨)
- [x] 학습 완료

### 2. `fusion_rag.ipynb` ★★★
> 쿼리 1개 → 여러 서브쿼리 생성 → Reciprocal Rank Fusion으로 순위 합산

- [x] 공통 패치 적용
- [x] `pip install` 셀 삭제
- [x] 학습 완료

### 3. `contextual_rag.ipynb` ★★★
> 검색된 문서를 LLM으로 압축 → 관련 정보만 남겨 정밀도 향상

- [x] 공통 패치 적용
- [x] `pip install` 셀 삭제
- [x] 학습 완료

### 4. `rewrite_retrieve_read.ipynb` ★★ (여유될 때)
- [ ] 공통 패치 적용
- [ ] `pip install` 셀 삭제
- [ ] 학습 완료

### 5. `parent_document_retriever.ipynb` ★★ (여유될 때)
- [ ] 공통 패치 적용
- [ ] `pip install` 셀 삭제
- [ ] 학습 완료

---

## Agent Techniques

### 6. `react.ipynb` ★★★ (필수)
> ReAct = Reasoning + Acting. LangChain + CrewAI로 구현

- [x] 공통 패치 적용
- [x] `pip install` 셀 삭제
- [ ] crewai 설치: `uv sync --extra agent`
- [x] 학습 완료

### 7. `reflexion.ipynb` ★★ (여유될 때)
> 자기 반성 루프로 응답 개선

- [ ] 공통 패치 적용
- [ ] `pip install` 셀 삭제
- [ ] 학습 완료

### 8. `rewoo.ipynb` ★★ (여유될 때)
> 계획 수립 → 실행 분리 (Plan-then-Execute)

- [ ] 공통 패치 적용
- [ ] `pip install` 셀 삭제
- [ ] 학습 완료

---

## Agentic RAG

### 공통 패치 (adaptive / self / corrective)
```python
# 제거
from langchain_core.pydantic_v1 import BaseModel, Field

# 수정
from pydantic import BaseModel, Field
```

### 9. `adaptive_rag.ipynb` ★★★ (필수)
> 쿼리 유형에 따라 벡터DB 검색 vs 웹 검색을 자동 선택

- [ ] 공통 패치 적용
- [ ] `pip install --q athina faiss-gpu langgraph` 셀 삭제
- [ ] `pydantic_v1` import 수정
- [ ] 학습 완료

### 10. `self_rag.ipynb` ★★★
> 검색 필요 여부 판단 → 생성 후 자기 검증 루프

- [ ] 공통 패치 적용
- [ ] `pip install --q athina faiss-gpu langgraph` 셀 삭제
- [ ] `pydantic_v1` import 수정
- [ ] 학습 완료

### 11. `corrective_rag.ipynb` ★★★
> 검색 결과 품질 판단 → 낮으면 웹 검색으로 fallback

- [ ] 공통 패치 적용
- [ ] `pip install` 셀 삭제
- [ ] `pydantic_v1` import 수정
- [ ] 학습 완료

---

## 노트북 실행 방법

```bash
cd C:/Users/김진규/rag-cookbooks

# PYTHONUTF8=1 필수 (한국어 Windows 인코딩 이슈)
PYTHONUTF8=1 uv run jupyter notebook

# 커널 선택: RAG Cookbooks (Python 3.11)
```

---

## 알려진 이슈

| 이슈 | 원인 | 해결 |
|---|---|---|
| `athina` import 오류 | setuptools 80+에서 `pkg_resources` 제거 | setuptools 69.x로 고정 (pyproject.toml) |
| UnicodeDecodeError | 한국어 Windows cp949 기본 인코딩 | `PYTHONUTF8=1` 환경변수 필수 |
| FAISS AVX2 경고 | Windows에서 AVX2 미지원 | 무시해도 됨, 자동으로 기본 FAISS 로드 |
| `langchain_core.pydantic_v1` | langchain 0.3+에서 제거 | `from pydantic import BaseModel, Field` 로 교체 |
