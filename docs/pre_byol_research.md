# PRE-BYOL 연구 기록

작성일: 2026-09-07 (Asia/Seoul). 기준은 이 작업 시작 시점의 실제 Hydra config와
미커밋 변경을 포함한 소스이다. 기존 `runs/`, `archive/`, checkpoint는 보존했다.
새 증거는 `runs/pre-research-20260907/`에 저장한다.

## 연구 질문과 현재 판단

**질문:** sentence BYOL에서 teacher target의 분산을 고르게 만들 때,
augmentation에 의해 달라지는 방향과 재현되는 방향을 구별하면 더 나은 학습 신호를 만들 수 있는가?

제안은 **Paired Reliability Equalization BYOL (PRE-BYOL)**이다.
teacher의 두 view에서 전체 공분산과 augmentation 차이 공분산을 추정하고,
신뢰도 행렬을 이용해 제한된 분산 보정을 teacher projector 앞에 적용한다.
기존 scalar center를 포함하므로 강도 0 ablation이 원래 구현으로 환원된다.

합성 데이터는 메커니즘을 지지하지만, 고정 BERT 표현의 STS-B 결과는
성능 향상 가설에 반하는 증거다. 3 seed × 3 방법 × 200 step의 실제 학습에서도
PRE의 우월성은 확인되지 않았다. 따라서 이 버전을 성능 개선 방법으로 채택하거나
논문 수준의 신규성·효과를 확정할 근거는 없다. 구현과 반증 가능한 가설, 부정적 결과를
후속 연구의 출발점으로 남긴다. 학습 결과와 계산 제약은 아래에 구분하여 기록한다.

## 저장소에서 확인한 사실

현재 기준 설정: `bert-base-uncased` revision
`86b5e0934494bd15c9632b12f734a8a67f723594`, mean pooling, word+dropout,
word strength 0.25, online/teacher dropout 0.35/0.02, projection 256,
projector/predictor hidden 4096, teacher EMA 0.992 → 1,
center momentum 0.96, α=0.05, batch 64, 길이 제한 256, 최대 3000 step.
encoder LR 3.05e-6, head LR 2.97e-4, warmup 비율 0.1,
**처음 150 step encoder 고정**, STS-B validation 전체 1500 pair를 300 step마다 평가한다.
평가 표현은 projector 이전의 online encoder embedding이며 추론에는 보정을 적용하지 않는다.

`scripts/research_pre.py audit`가 STS 기록을 가진 186개 과거 run을 수집했다.
가능하면 `.hydra/config.yaml`, 없으면 `config.json`을 사용하고 commit 소스,
`git_state.json`, `git.diff` SHA256 및 dropout 관련 patch를 함께 기록했다.
요약은 `audit/historical_runs.json`, 현재 설정은 `audit/current_config.yaml`에 있다.
일부 오래된 artifact writer는 staged 변경을 patch에 포함하지 않았다.
따라서 설정은 복원해도 실행 소스를 완벽히 복원할 수 없는 run이 있다.

| 과거 run | 실제 word/dropout 조건 | 마지막 STS ρ | effective rank | uniformity |
|---|---|---:|---:|---:|
| pretrained | 동일 초기 가중치 | 0.59312 | 511.57 | −1.63483 |
| word_aug_test | word 0.15, dropout 꺼짐 | 0.70918 | 485.48 | −2.04091 |
| word_aug_test_much_test | word 0.20, dropout 꺼짐 | 0.71640 | 480.41 | −2.01812 |
| word_aug_test_very_test | word 0.25, dropout 꺼짐 | 0.72229 | 477.32 | −1.98244 |
| word_aug_test_both | word 0.25, online/teacher dropout 켜짐 | 0.70404 | 483.74 | −2.04692 |

마지막 run의 resolved config는 `augmentation.name=word`였지만 saved patch가 dropout을
켜고 있었다. 앞의 세 run은 commit의 word branch에서 dropout을 명시적으로 끈다.
따라서 run 이름이나 config의 dropout 확률만으로 augmentation을 판단하면 안 된다.
또한 첫 세 run에서는 STS가 높을수록 rank는 낮고 uniformity 값은 덜 음수였다.
이것은 상관적 관찰이며, rank 저하가 성능 향상의 원인이라는 증거가 아니다.
과거 DINO/InfoNCE의 실패 기록 역시 현재 BYOL과 objective·소스가 달라 대조군으로 쓰지 않는다.

## Scalar center의 한계

고정된 batch 통계 c와 모든 문장에 동일한 α에 대해

\[
(h_i-\alpha c)-(h_j-\alpha c)=h_i-h_j,\qquad
\operatorname{Cov}(h-\alpha c)=\operatorname{Cov}(h).
\]

따라서 이 연산 자체는 centered covariance의 고유값, rank,
Euclidean 문장 간 거리를 바꾸지 못한다. 동일한 평균을 가진 두 분포에서
어떤 방향은 유용한 변이인지 augmentation 잡음인지도 식별하지 못한다.
단, 원점이 바뀌면 cosine은 바뀌고, 뒤의 비선형 projector/LayerNorm과 학습 dynamics도
달라질 수 있다. 따라서 “centering은 아무 효과가 없다”거나 “학습된 covariance를 절대
바꾸지 못한다”는 주장은 하지 않는다.

모든 방향을 whitening하는 대안은 이미 잘 알려져 있고,
작은 분산에 포함된 잡음까지 증폭할 수 있다.
여러 centroid로 centering하는 대안은 topic 성분을 제거할 위험이 있고,
token별 anchor는 별도의 token 의미 보존 가정을 요구한다.
여기서는 이미 계산하는 두 teacher view로 검증 가능한 통계적 가설을 선택했다.

## 제안 방법

같은 문장에 대해 teacher embedding을 \(h_1,h_2\)라 하자.
두 augmentation이 조건부 독립이고 동일한 분포라면
\(h_v=s+\epsilon_v\), \(E[\epsilon_v\mid s]=0\)로 쓸 수 있다.
여기서 s는 augmentation 평균 표현이며, 반드시 원문 의미 전체와 같지는 않다.

\[
C=\operatorname{Cov}(h_v),\quad
N=\tfrac12 E[(h_1-h_2)(h_1-h_2)^\top],\quad S=C-N.
\]

이 모델에서 N은 augmentation noise covariance, S는 view 사이에 공유되는 covariance다.
N 추정에는 STS label, negative pair, 새 encoder forward가 필요 없다.

현재 batch의 loss를 계산한 **뒤에** teacher의 raw embedding으로 통계를 업데이트한다.
mean μ와 C는 batch 간 평균 이동 항까지 포함하는 EMA mixture covariance로 유지한다.
첫 batch 관측값으로 통계를 초기화하여 가상의 0 벡터 관측을 섞지 않는다.
기존 center c의 0 초기화와 EMA 규칙은 그대로 유지한다.
실행 순서는 target 생성 → loss/backward → optimizer/teacher EMA → 통계 업데이트이며,
통계에 들어가는 embedding은 teacher EMA 직전 forward의 값이다.

\(C=U\Lambda U^\top\), \(\tau=\mathrm{tr}(C)/d\),
\(\delta=0.01\max(\tau,10^{-12})\)라 두고 다음을 계산한다.

\[
W=(C+\delta I)^{-1/2},\qquad
R=\operatorname{clip}_{[0,1]}\!\left(W(C-N)W\right),
\]

\[
D=U\operatorname{diag}\!\left[
\operatorname{clip}_{[0.5,2]}\sqrt{\frac{\max(\tau,10^{-12})}{\lambda_j+\delta}}-1
\right]U^\top,\qquad
K=\beta R^{1/2}DR^{1/2},
\]

\[
\boxed{\ h'_{target}=h_{target}-\alpha c+(h_{target}-\mu)K\ },\qquad \beta=1.
\]

clip은 R의 고유값에 적용한다. 수식 마지막 줄은 row-vector 표기다.
R의 nullspace에는 추가 보정이 없고, 안정적인 방향에만 분산 equalization이 작용한다.
ridge를 S에 더하지 않으므로 N=C인 순수 잡음을 신뢰할 이유가 생기지 않는다.
R과 C가 commute할 필요가 없다는 점이 PCA 방향별 scalar gate와 다르다.

초기 후보는 C의 PCA 방향마다 noise/total variance 비율을 계산했다.
전체 분산이 같거나 가까운 안정 신호와 잡음의 방향이 섞이면 이 방식은 구분에 실패한다.
full R은 그 off-diagonal 구조를 보존한다. 초기 후보의 ridge 처리도 순수 잡음에
양의 신뢰도를 부여했으므로 수정했다. 최종 `diagonal` ablation은 **동일한 ridge와
signal numerator**를 사용하여 off-diagonal 효과만 제거한다.
초기 후보의 결과는 `synthetic/`, 수정된 결과는 `synthetic-matrix/`에 각각 보존한다.
`none`/R=I는 같은 mean 처리·ridge·gain bound를 쓰는 **내부 bounded-whitening ablation**이다.
W-MSE나 WhitenedCSE 논문 전체를 재현한 baseline은 아니며, 그 논문의 결과와 비교하지 않는다.

R의 고유값은 [0,1], D의 고유값은 [−0.5,1]이므로,
Rayleigh quotient로 \(I+K\)의 고유값이 [0.5,2]임을 얻는다 (β∈[0,1]).
따라서 한 번의 보정은 차원을 제거하거나 무제한으로 증폭하지 않는다.
가역 연산이므로 이미 분산이 정확히 0인 방향을 복구하지도 않는다.
operator probe에서 effective rank가 늘어나는 것은 존재하는 분산의 재분배이지 algebraic rank의 증가가 아니다.
FP32에서는 경계에 수 e−6 정도의 오차가 관찰될 수 있다.
이는 **전체 BYOL 학습의 collapse 방지 정리나 STS 향상 보장**이 아니다.

### 구현과 비용

`src/minimal_dino/geometry.py`에 통계와 연산자를 분리했다.
`objective=paired_byol`만 이를 활성화하고 기본 `objective=byol`은 그대로다.
target에만 적용하며 online 평가 embedding과 inference 비용은 바뀌지 않는다.
FP32 통계·분해, stop-gradient, 50-step 통계 warmup, 이후 10 step마다 분해를 사용한다.
EMA mean과 covariance는 매 step 갱신된다. 실행 때의 config, 전체 buffer 및 갱신 수는
checkpoint에 저장되고, 다른 geometry 설정으로 resume하면 오류를 낸다.

추가 비용은 batch마다 O(Bd²), 분해 시 O(d³), persistent buffer O(d²)다.
768차원에서 네 개의 768×768 FP32 행렬이 약 9 MiB를 차지한다
(분해 작업공간과 transient 행렬 제외). 새로운 teacher forward는 없다.
현재 단일-process training 구현이며 DDP 통계 동기화는 구현하지 않았다.

## 선행 연구와 신규성의 범위

문장 BYOL 자체는 새롭지 않다. [BSL (ACL 2021)](https://aclanthology.org/2021.acl-long.402/)은
두 augmented sentence view와 EMA target을 이용하는 bootstrapping을 이미 제안했다.
[UNSEE (2024)](https://arxiv.org/abs/2401.15316)는 EMA target과
non-contrastive sentence objectives를 연구했다.

[W-MSE](https://arxiv.org/abs/2007.06346)는 whitening과 positive-view regression을,
[WhitenedCSE](https://aclanthology.org/2023.acl-long.677/)는 grouped whitening과
contrastive sentence learning을 결합했다.
[VICReg](https://arxiv.org/abs/2105.04906)는 variance/covariance loss를 명시적으로 사용한다.
따라서 “covariance를 추가했다”, “negative가 없다”, “whitening을 쓴다”만으로 기여라 할 수 없다.

[DirectPred 분석](https://arxiv.org/abs/2102.06810)의 부록 D.2는
augmentation signal/noise covariance와 BYOL의 학습 방향 관계를 이미 분석한다.
새로운 signal/noise 분해 이론을 발견했다고 주장하지 않는다.
또한 조건부 독립 view의 population에서 S는 cross-view covariance이므로,
R은 이를 whitening한 correlation 계열 통계다. 통계량 그 자체의 최초성도 주장하지 않는다.
[SiNGER](https://arxiv.org/abs/2509.20986)는 vision teacher feature refinement를 다룬다.
따라서 teacher 정제 자체도 최초성의 근거가 아니다.

이 저장소에서 검증하는 구체적 기여 후보는 **causal paired teacher statistics로 얻은
full reliability matrix를 bounded covariance equalization의 양쪽에 곱하여,
noncommuting signal/noise 구조까지 반영하는 target-only sentence BYOL 학습법**이다.
이는 공통 translation을 바꾸는 하이퍼파라미터 조정으로 표현할 수 없다.
위 primary sources에 대해 차이를 확인했지만 체계적인 전 문헌 신규성 검증을 마친 것은 아니다.
논문 수준 기여가 되려면 아래 ablation을 넘는 full-training 및 외부 benchmark 근거가 필요하다.

## 검증 1: 합성 signal/noise 분리

세 seed(11,22,33), seed마다 8192개 12차원 관측을 사용했다.
4개 강한 stable 방향(signal variance 9), 4개 약한 stable 방향(0.09),
4개 약한 noisy 방향(signal 0.0001, noise 0.09)이 있다.
약한 stable/noisy 방향의 total variance가 비슷하므로 C만으로 구별하기 어렵다.
모든 방법에 동일한 관측을 제공하고, label이나 STS에 맞추어 parameter를 고르지 않았다.
결과는 알려진 생성 signal/noise covariance에 연산자를 적용한 에너지 비율이다.
이는 새 random holdout 관측의 기대 에너지에 대응하지만, 학습된 encoder의 성능은 아니다.

| 방법 | 잡음 에너지 배율 | 약한 안정 신호 에너지 배율 |
|---|---:|---:|
| scalar (이론값) | 1 | 1 |
| PRE matrix | 약 0.965 | 약 3.021 |
| 신뢰도 제거 (`none`) | 약 3.638 | 약 4.000 |
| diagonal R | 약 1.559 | 약 2.116 |
| 잘못된 문장 짝 (`shuffled`) | 약 1.01 | 약 1.02 |

분산을 무조건 키우는 것과 pair의 정보를 사용하는 것이 다름을 보여준다.
하지만 augmentation이 의미를 파괴하여 N=C가 되면 R=0이다.
이 경우 유용하더라도 view 사이에 재현되지 않는 의미는 보정 대상으로 선택되지 않는다.
`synthetic-matrix/results.json`에 이 반례도 저장했다.

## 검증 2: 고정 pretrained BERT 표현의 실제 STS

Wikipedia 학습 문장 2048개를 seed 42로 추출하고 word strength 0.25 및 teacher dropout
0.02의 두 view를 계산했다. 계산 제약으로 이 Wiki 입력만 max length 64로 제한했다.
encoder/head 가중치는 고정했고, Wiki에서 population 통계를 한 번 fit했다.
STS-B validation 1500 pair는 augmentation·truncation 없이 평가했다.
STS의 embedding이나 label은 통계 fit에 들어가지 않았다. label은 보고에만 썼다.
Wiki 및 validation 파일 SHA256, 선택한 Wiki index, feature cache를 함께 보존했다.

**이 결과는 post-processing operator의 probe이며, BYOL로 학습한 encoder의 결과가 아니다.**
stationary center 추정을 위해 center momentum=0, geometry warmup=1을 사용했다.
온라인 EMA 학습과 구분하고, 과거 0.704–0.722와 성능 비교하지 않는다.

| 방법 | STS ρ | scalar 대비 Δρ | effective rank | participation ratio |
|---|---:|---:|---:|---:|
| raw pretrained | 0.59312 | +0.00104 | 511.57 | 32.58 |
| scalar α=0.05 | 0.59208 | 0 | 511.57 | 32.58 |
| PRE matrix | 0.58723 | −0.00484 | 573.07 | 38.89 |
| whitening, R=I | 0.59905 | +0.00697 | 607.39 | 47.68 |
| diagonal R | 0.59992 | +0.00785 | 582.88 | 42.36 |
| shuffled pairs | 0.59092 | −0.00116 | 520.46 | 33.17 |
| β=0 | 0.59208 | 0 | 511.57 | 32.58 |

PRE의 scalar 대비 paired bootstrap 95% 구간은 [−0.00875, −0.00130]이다.
1000회 validation pair 재표집, 동일 bootstrap index로 두 방법을 비교했다.
이는 고정한 모델·fit set에서의 pair 표본 불확실성이며, seed 간 학습 분산을 나타내지 않는다.
STS pair 사이의 공통 문장/출처 의존성 및 여러 비교에 대한 보정도 반영하지 않았다.
`probe/results.json`과 `probe/predictions.npz`에서 재계산할 수 있다.

PRE의 rank와 participation ratio는 높아졌지만 STS는 낮아졌다.
alignment 역시 scalar 0.19236 → PRE 0.22153으로 나빠졌다.
따라서 여기서 “dimensional collapse를 줄였으므로 semantic quality가 향상됐다”는
결론은 성립하지 않는다. full matrix의 자유도가 실제 텍스트에서 도움이 되는지도 아직 지지되지 않는다.

### 음성 결과에 대한 추가 진단

위 결과를 본 뒤 두 진단을 수행했으며, 독립 확인 실험으로 간주하지 않는다.
첫째, α=0.05와 full center α=1이라는 두 고정 조건만 비교했다.
α=1에서 scalar는 0.58182, PRE는 0.57188로 모두 낮아졌다.
따라서 이 데이터에서 부족한 평균 제거만이 PRE의 약점이라는 설명은 지지되지 않았다.
이는 α 탐색을 통해 방법을 선택한 결과가 아니다.

둘째, Wiki 2048개를 반으로 나누어 한쪽에서 통계를 fit하고 다른 쪽의 noise energy를 측정했다.
두 half의 역할을 바꾸어 반복했다. 비율은 같은 partition에서 보정 전 대비 보정 후다.

| 방법 | fit noise energy 배율 (2 fold) | heldout noise energy 배율 (2 fold) |
|---|---|---|
| PRE matrix | 0.960 / 0.958 | 1.048 / 1.058 |
| R=I | 1.206 / 1.192 | 1.299 / 1.307 |
| diagonal R | 0.856 / 0.849 | 0.869 / 0.873 |
| shuffled | 1.178 / 1.180 | 1.186 / 1.189 |

PRE가 추정한 reliability 순위와 heldout empirical reliability의 Spearman은
0.885/0.876으로 양의 관계를 보인다. 그러나 가장 낮은 1/4 방향의 empirical reliability는
fit에서 −0.38 정도, heldout에서는 +0.61–0.63이다. 순위가 맞더라도
고차원 covariance와 그 inverse를 추정하는 과정의 크기 오차가 매우 크다는 뜻이다.
clip 자체도 finite-sample 양의 부분에 선택 편향을 줄 수 있다.
이는 표본 수를 늘리거나 off-diagonal shrinkage를 검증해야 할 근거이며,
**STS 저하 원인을 covariance 오차 하나로 확정하는 증거는 아니다.**
augmentation-stability와 semantic usefulness의 불일치도 아직 가능한 설명이다.
전체 기록은 `diagnostics/results.json`에 있다.

표본 수만 늘리면 해결되는지도 추가로 검사했다. 고정한 Wiki 512개를 holdout으로 두고
fit 문장 수 256/512/1024/1536에서 PRE의 heldout noise 배율은 각각
0.953/1.054/1.046/1.036이었다 (`sample-size/results.json`). 512 이후 일반화 gap은
작아졌지만, 전체 구간에서 단조 개선은 아니다. 작은 표본에서 생기는 rank 제약도
연산자를 바꾸므로, 이 결과를 “표본이 많을수록 의미 표현이 좋아진다”로 해석할 수 없다.
이 검사는 STS label을 쓰지 않았다.

### 학습된 기존 checkpoint에서의 반례

초기 BERT에만 해당하는 문제인지 확인하기 위해 `runs/word_aug_test_both/checkpoint.pt`의
online encoder를 고정한 별도 probe를 수행했다. 가장 높은 과거 점수를 고른 것이 아니라,
현재 기본 word+dropout 조건에 가까운 run을 선택했다. 기존 checkpoint는 읽기만 했으며
원본 SHA256은 `cache-trained/manifest.json`에 저장했다.
동일한 Wiki index와 word view, 같은 α·ridge·gain bound를 사용했다.
모델 초기화 경로가 달라 dropout mask까지 encoder 간 동일하게 고정한 비교는 아니며,
각 checkpoint **내부**에서는 모든 보정 방법이 동일한 cached feature를 공유한다.

| 기존 학습 encoder의 보정 | STS ρ | effective rank | participation ratio |
|---|---:|---:|---:|
| raw | 0.70405 | 483.74 | 42.47 |
| scalar | 0.70377 | 483.74 | 42.47 |
| PRE matrix | 0.68409 | 569.74 | 60.41 |
| R=I | 0.68234 | 593.30 | 67.06 |
| diagonal R | 0.68694 | 573.01 | 61.21 |
| shuffled | 0.70177 | 493.89 | 44.25 |

PRE−scalar Δρ는 −0.01968, 같은 방식의 pair bootstrap 95% 구간은
[−0.02428, −0.01469]다. 신뢰도를 제거한 equalization도 악화되었다.
PRE의 alignment도 0.15377 → 0.19538로 나빠졌다.
학습된 표현에서는 분산을 더 고르게 만들수록 성능이 나빠지는 반례를 얻었다.
따라서 nuisance anisotropy와 의미를 담는 covariance를 augmentation stability만으로
충분히 구별할 수 있다는 가설은 지지되지 않는다.
이는 **PRE를 처음부터 학습한 모델의 결과가 아니며**, 신규 학습법의 최종 성능과 혼동하면 안 된다.
데이터와 결과는 `cache-trained/`, `probe-trained/`에 있다.

## 검증 3: 동일 조건의 end-to-end pilot

실행 조건을 미리 고정했다: 200 step, batch 8, max train length 64,
CPU FP32, worker 0, STS-B validation 전체 1500 pair를 step 0과 200에 평가.
encoder freeze 150, LR, dropout, α, momentum, head dimension 등은 기준 config 그대로다.
총 1600개의 training sentence presentation이며 encoder가 바뀌는 것은 마지막 50 step뿐이다.
학습률과 teacher momentum의 schedule horizon도 200 step이므로 3000-step 실험의 prefix가 아니다.
baseline, PRE, R=I에 동일 seed/데이터 순서/augmentation/dropout 난수열을 사용한다.
geometry 계산은 난수를 소비하지 않는다. 최종 step으로 평가하며 최고 validation step을 고르지 않는다.
seed 42가 진행 중일 때, 조건은 그대로 두고 seed 43/44 반복을 추가했다.

아홉 run 모두 완료했다. 결과는 `pilot-matrix/`, `pilot-seed43/`, `pilot-seed44/`의
`*/metrics.jsonl`, resolved configs, checkpoints 및 validation predictions에 기록했다.
최종 집계는 `aggregate/results.json`, 그림은 `aggregate/pilot_comparison.png`와 `.pdf`다.

완료된 scalar/PRE 비교의 seed별 최종 ρ는 다음과 같다.

| seed | scalar | PRE matrix | PRE − scalar |
|---|---:|---:|---:|
| 42 | 0.593236 | 0.593115 | −0.000121 |
| 43 | 0.594265 | 0.593927 | −0.000338 |
| 44 | 0.593058 | 0.593352 | +0.000294 |

세 seed 평균은 scalar 0.593520, PRE 0.593465이다.
seed 간 sample SD는 각각 0.000651/0.000417이다. 차이의 부호가 일관되지 않으며,
평균 차이 −0.000055로 이 pilot에서 우월성을 주장할 근거가 없다.
전체 3000-step 효과가 없음을 입증한 결과도 아니다.

| 방법 | STS ρ, 평균 ± seed SD | effective rank | participation ratio | alignment | uniformity |
|---|---:|---:|---:|---:|---:|
| scalar | 0.593520 ± 0.000651 | 511.227 | 32.594 | 0.184316 | −1.644719 |
| PRE matrix | 0.593465 ± 0.000417 | 511.355 | 32.650 | 0.185047 | −1.652203 |
| bounded whitening (R=I) | 0.593645 ± 0.000410 | 511.374 | 32.656 | 0.185130 | −1.653263 |

동일한 validation pair index를 세 seed에 공유하여 1000회 bootstrap했다.
PRE−scalar 평균 차이의 95% 구간은 [−0.000290, +0.000213],
R=I−scalar는 [−0.000109, +0.000376]이다. 둘 다 0을 포함한다.
이는 seed를 고정한 pair 표본 불확실성이며, seed 선택의 불확실성까지 합친 구간이 아니다.
R=I는 PRE보다 세 seed 모두 높았지만 차이는 작았다. 이 결과는 full reliability
행렬을 추가해야 할 필요성도 지지하지 않는다.
export된 cosine과 기존 평가의 추가 normalize 및 재-encoding으로 생기는 ρ 차이는
최대 4.61e−6이었고 별도로 기록했다. 이 수치 차이로 결론이 바뀌지는 않는다.

모든 run에서 공통 config와 source patch, 1500개 STS score 순서가 일치했다.
모든 seed의 첫 50-step loss는 방법 간 bitwise 동일했고, encoder를 고정한
첫 150-step raw std/student view cosine/teacher view cosine도 정확히 같았다.
이에 대한 자동 검사는 `aggregate/results.json`의 `trajectory_checks`에 있다.

평가 embedding std는 평균 scalar 0.22790, PRE 0.22867, R=I 0.22856으로,
완전한 collapse의 증거는 없다. effective rank와 covariance top1 mass 변화도 작다
(top1 mass scalar 0.14330, PRE/R=I 약 0.14309).
따라서 이 짧은 pilot에서는 STS와 geometry가 함께 큰 개선을 뒷받침하지 않는다.
초기 encoder의 고정 구간과 짧은 schedule이 효과를 제한하므로,
장기 학습 결과에 대한 결론으로 확대해서는 안 된다.

NVIDIA device가 이 환경에 노출되지 않았고 `torch.cuda.is_available()`은 False였다.
따라서 3000-step, batch 64의 full GPU 실험은 수행하지 않았다.
새 결과는 약 16 GiB이며 대부분 아홉 개의 full training checkpoint다.

## 후속 실험과 반증 기준

현재 기본값을 유지한 3000-step, batch 64, max length 256 실험을 seed 42/43/44로 수행한다.
필수 비교는 scalar, PRE matrix, R=I, diagonal R, shuffled pairing, β=0이다.
마지막 두 대조군은 구현 환원과 paired 통계의 인과적 역할을 확인한다.
대상 방법을 고정한 후 heldout STS-B test를 한 번 평가하고,
STS12–16/SICK-R 또는 관련 MTEB subset으로 일반화 범위를 확인한다.
기존 이력에서 반복 관찰한 validation은 이미 연구 선택에 사용된 개발 데이터다.

평가할 주장은 다음과 같다.

1. PRE가 scalar보다 seed 평균 STS에서 개선되고, seed 간 변동에 비해 효과가 유지되는가?
2. 동일한 whitening의 R=I 및 diagonal R보다 좋은가? 그렇지 않으면 full reliability의 필요성이 없다.
3. pairing을 깨면 효과가 감소하는가? 그렇지 않으면 paired noise 해석은 지지되지 않는다.
4. effective rank만이 아니라 covariance top1/top10 mass, participation ratio,
   std, cosine 분포, alignment/uniformity, target projection std가 함께 설명되는가?
5. 보정 강도 0의 전체 loss/평가 trajectory가 baseline과 같고 checkpoint resume도 일치하는가?
6. word-only/dropout-only에서도 성립하는가? 의미 파괴 augmentation에서는 실패하는가?
7. 통계 표본 수를 늘리거나 covariance off-diagonal shrinkage를 적용하면
   full matrix와 diagonal의 차이가 바뀌는가? parameter 탐색이 아닌 추정오차 가설의 검증으로 실시한다.
8. projector/predictor가 보정을 흡수하여 encoder 변화가 작아지는가?
   raw target와 projected target의 geometry를 함께 측정하고, 동일한 대조군 구성에서
   head 앞의 보조 regression을 비교하면 이 가설을 구분할 수 있다.

N이 작은 방향은 반드시 semantic하지 않고, 다른 문장과 공유되는 topic/style도 안정적일 수 있다.
teacher 파라미터 이동이 C에 포함되지만 N에는 같은 방식으로 들어가지 않아 temporal drift를
signal로 오인할 가능성도 있다. 작은 batch/큰 d에서는 covariance의 rank와 표본오차가 문제다.
잘못된 pair의 finite-sample C−N도 양의 고유값을 가질 수 있어 shuffled R이 완전히 0일 필요는 없다.
현재 보정은 global affine operator이며 teacher projector의 첫 linear layer로 대수적으로
합성할 수 있다. EMA로 묶인 두 head와 predictor가 이 변화를 얼마나 흡수하는지는 별개의
학습 dynamics 문제다. target 분산을 바꿨다는 사실만으로 raw encoder geometry가 바뀐다고 볼 수 없다.
이 모두가 미래 ablation에서 확인할 제한이다.

## 실행 방법

모든 명령은 project root에서 수행하고 Python 전에 venv를 활성화한다.
현재 sandbox에서는 Hub model cache를 읽되 datasets cache 쓰기는 `/tmp`로 지정했다.
아래 output 이름은 예시이며 이미 존재하면 새 이름을 사용한다. script는 기존 디렉터리를 덮어쓰지 않는다.

```bash
source .venv/bin/activate
export UV_CACHE_DIR=.uv-cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/tmp/minimal-dino-hf-datasets

uv run --offline python scripts/research_pre.py audit --output runs/pre-new/audit
uv run --offline python scripts/research_pre.py synthetic --output runs/pre-new/synthetic
uv run --offline python scripts/research_pre.py cache --output runs/pre-new/cache \
  --samples 2048 --batch-size 16 --max-length 64
uv run --offline python scripts/research_pre.py probe --cache runs/pre-new/cache \
  --output runs/pre-new/probe
uv run --offline python scripts/research_pre_diagnostics.py --cache runs/pre-new/cache \
  --output runs/pre-new/diagnostics
uv run --offline python scripts/research_pre_diagnostics.py --cache runs/pre-new/cache \
  --output runs/pre-new/sample-size --sample-curve
uv run --offline python scripts/research_pre_checkpoint_cache.py \
  --checkpoint runs/word_aug_test_both/checkpoint.pt --reference-cache runs/pre-new/cache \
  --output runs/pre-new/cache-trained
uv run --offline python scripts/research_pre.py probe --cache runs/pre-new/cache-trained \
  --output runs/pre-new/probe-trained
uv run --offline python scripts/research_pre.py pilot --output runs/pre-new/pilot \
  --batch-size 8 --max-length 64 --steps 200 --seeds 42 --methods scalar paired none
uv run --offline python scripts/research_pre.py summarize --pilot runs/pre-new/pilot \
  --output runs/pre-new/summary
# 여러 pilot 폴더를 합쳐 common config, source, STS pair 일치를 검사한다.
uv run --offline python scripts/research_pre_summary.py \
  --pilots runs/pre-research-20260907/pilot-matrix \
    runs/pre-research-20260907/pilot-seed43 runs/pre-research-20260907/pilot-seed44 \
  --output runs/pre-new/aggregate
MPLCONFIGDIR=/tmp/minimal-dino-mpl uv run --offline python scripts/research_pre_figures.py \
  --root runs/pre-research-20260907 --output runs/pre-new/figures

# GPU에서 full default comparison. 이 작업에서는 실행하지 않았다.
uv run --offline python scripts/research_pre.py pilot --full --device cuda \
  --output runs/pre-full-new --seeds 42 43 44 \
  --methods scalar paired none diagonal shuffled zero

uv run --offline python -m pytest -q
uv run --offline ruff check src tests scripts/research_pre*.py
```

### 측정 정의와 보존 기록

기존 `effective_rank`는 **centered embedding의 singular value 분포**의 entropy rank다.
covariance eigenvalue entropy rank와 다르다. 새로운 `participation_ratio`는
\((\sum\lambda)^2/\sum\lambda^2\), top1/top10 mass는 covariance eigenvalue 질량이다.
상수 embedding의 rank는 0으로 명시했다.
alignment의 positive는 로컬 STS score >0.8이며, 실제 parquet가 [0,1] 범위임을 확인했다
(validation에서 208 pair). raw [0,5] score에 이 threshold를 그대로 적용하면 안 된다.

기존 테스트에서 config 기본값을 예전 값으로 검사하던 부분을 현재 Hydra 값으로 수정했다.
초기 2개 테스트 실패 중 하나는 이 stale assertion, 다른 하나는 쓰기 불가능한
global datasets cache였다. 후자는 실행 환경의 cache 경로를 바꾸어 해결했다.
최종 단위 테스트 74개가 통과했다. Ruff 전체 검사도 통과했다.
테스트에는 translation 한계, mixture covariance, 동일 분산 signal/noise 구별,
회전 equivariance, degenerate/singleton batch, causal update, teacher gradient 차단,
β=0 환원, checkpoint 복원 및 잘못된 설정의 resume 거부가 포함된다.

그림은 `figures/mechanism_vs_semantics.png`와 `.pdf`에 저장했다.

`pilot/`은 datasets cache 권한 문제로 학습 전 실패했다.
`pilot-v2/`는 방법을 수정하면서 import된 코드와 디스크 snapshot이 달라지는 것을 방지하기 위해
scalar의 freeze 단계에서 중단했다. 이 두 폴더는 완료된 비교에서 제외한다.
모든 기존 실험 파일은 삭제하지 않았다.
