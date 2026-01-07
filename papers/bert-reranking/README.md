# Passage Re-Ranking with BERT (2019)

Created: 2025년 10월 2일 오후 4:55

# 한줄요약

query랑 passage를 sentence A, B로 BERT에 동시에 넣었더니 re-ranking 성능이 우수했음.

(BERT의 [CLS] 벡터를 하나의 single-layer 신경망에 넣어 passage가 관련 있는지 여부의 확률을 구함.)

# Introduction

이전까지 IR 분야에서 neural ranking 모델들은 passage ranking용 데이터셋 부족으로 크게 발전하지 못함.

→ 이 논문에서는 MS MARCO passage ranking 데이터셋 + BERT로 우수한 성과 달성을 보여줌.

# Passage Re-ranking with BERT

### Task

QA 파이프라인은 크게 세 단계:

1. 주어진 질문과 관련 있을 법한 대량의 문서를 corpus에서 검색. (여기서는 BM25를 사용한다고 하는데, DPR 논문 이전에 나온 거라 그런 것 같음.)
2. passage re-ranking - 이 문서들 각각을 더 정확하게 점수화하고 순위를 다시 매김.
3. 이 중 상위 k개가 answer generation 모듈에 의해 최종 후보 답변 출처가 됨.

이 논문에서는 2단계 : passage re-ranking에 BERT를 적용하는 것에 집중.

### Method

re-ranker의 역할 : 후보 passage d_i가 query q와 얼마나 관련있는지를 나타내는 점수 s_i를 추정하는 것.

re-ranker로 BERT 사용.

**query를 sentence A, passage를 sentence B로 BERT에 입력.**

(query는 최대 64토큰, query + passage + [SEP] 토큰의 전체 길이는 512토큰으로 truncate.)

BERT-Large를 binary classification model로 사용.

**즉, [CLS] 벡터를 하나의 single-layer 신경망에 넣어 passage가 관련 있는지 여부의 확률을 구함.**

각 passage에 대해 독립적으로 확률 계산 후 이를 기준으로 passage를 ranking하여 최종 리스트를 얻음.

사전학습된 BERT에서 시작하여 이를 re-ranking 태스크에 맞게 cross-entropy loss로 파인튜닝.

**Loss :**

$$
L = - \sum_{j \in J_{\text{pos}}} \log(s_j)     - \sum_{j \in J_{\text{neg}}} \log(1 - s_j)
$$

J_pos : 관련 passage들의 인덱스 집합

J_neg : Retriever가 가져온 상위 1000개 문서 중 비관련 passage들의 인덱스 집합.