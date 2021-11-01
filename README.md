# Technology forecasting using GNN 

## 1. 프로젝트 소개

### 진행기간

2020.11 - 2021.06<br></br>

### 필요 기술

`Graph Neural Network` `Doc2vec` 

`web crawling` `visualization` <br></br>

### 참여 인원

- 김인조<br><br>

### 사용 툴 및 라이브러리

`pytorch` `pytorch geometric` `python` 

`scikit learn` `gephi`
<br></br>

### 연구 목적

깃허브 데이터를 활용하여 자율주행 산업군의 요소 기술 분석 및 향후 유망기술을 예측
<br></br>
<br></br>

## 2. GNN 기반 자율 주행 유망 기술 예측 

본 프로젝트에서는 Graph Neural Network 알고리즘 중 Graph convolutional network와 Variational graph autoencoder를 활용하여 링크 예측을 진행하였습니다. 실험에 사용된 네트워크는 GitHub repository를 노드로 하고 저장소 간의 co-contribution 관계를 엣지로 하고 있습니다. 
<br></br>

네트워크 구축과 자율 주행 동향 분석 관련된 내용은 [여기](https://github.com/Kiminjo/Technology-forecasting-using-GNN/files/7453594/2021._.pdf)를 참고해주세요.
<br></br>

GNN 기반 링크 예측 알고리즘과 자율 주행 유망 기술 예측에 관련된 내용은 향후 업로드 예정입니다.
<br></br>

실험에 사용된 라이브러리와 베이스 코드는 pytorch-geometric reference를 참고하였습니다. 더 자세한 내용은 [여기](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/link_pred.py)를 참고해주세요.
<br></br>

모델의 feature vector는 각 노드의 node2vec 임베딩 벡터와 Readme 문서의 doc2vec embedding vector의 concat으로 구성하였습니다. 
<br></br>
<br></br>

## 3. 데이터셋 
실험에는 GitHub repository 데이터가 사용되었습니다. Readme와 contributor 수, stargazer수, forker수는 노드의 feature vector를 구성하며, contributor 정보는 네트워크를 구축하는데 사용되었습니다. 

**데이터 통계**
- 23,017개의 데이터가 `Autonomous vehicle`, `self-driving car`등 5개의 자율 주행 관련 토픽으로부터 수집되었습니다. 

- 385개의 데이터를 User type과 contributor 수를 기준으로 필터링하여 최종  확정하였습니다.

- contributor counts, stargazer counts, forker counts 등 social 통계량은 멱함수의 형태를 띄고 있어 Normalizer로 정규화 과정을 거쳤습니다.

- 3.4%의 데이터가 readme를 가지고 있지 않아, Doc2vec 시 'None' 으로 대체하였습니다. 
<br></br>
<br></br>

## 4. Key file 

- `link_prediction_GCN.py` : Graph convolutional network를 이용하여 실제 링크 예측 과정을 수행합니다.

- `link_prediction_GAE.py` : Variational graph auto-encoder를 이용하여 실제 링크 예측 과정을 수행합니다. 


