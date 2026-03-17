# Flipchip_surrogate
# 🚀 AI 기반 반도체 패키지 신뢰성 예측 프로젝트
## 반도체가 열변형에 의해 휘어지고 깨지고 뜯어지는 것을 ai로 예측합니다.
<br>

## 0. 키워드 설명

### 1) CAE (Computer-Aided Engineering)란 무엇인지?

![01  CAE란](https://github.com/user-attachments/assets/166da2d9-df44-4706-9bdf-a40879853e84)
   컴퓨터 속 가상 세계에서 그 물건이 **튼튼한지, 열에 잘 견디는지 등을 미리 시뮬레이션해 보는 '디지털 모의실험'입니다.
   
   CG나 3D 모델링에서 배우의 얼굴이나 몸에 점을 찍어 동작을 읽어내는 '트래커(Tracker)'를 떠올려 보세요.
   
   CAE의 메쉬는 그 점 하나하나마다 물리 법칙(힘, 열, 압력)을 계산하기 위한 계산 포인트입니다.

   
### 2) flipchip이 무엇인지?
 
![02  플립 칩이란](https://github.com/user-attachments/assets/cd8d6bb6-e3c8-49c0-a294-5bbf95eb10b2)

   "플립칩(Flip-Chip)"은 반도체 칩을 기판에 연결할 때 칩을 뒤집어(Flip) 표면의 수많은 돌기(Bump)를 기판에 직접 맞닿게 붙이는 연결 기술입니다.
   
   이번 프로젝트는 제 pc의 cpu인 i9-13900k를 모델로 했고 이 cpu가 플립칩 타입입니다.
        
### 3) 왜 변형되면 안되는지

![03  왜 변형이 되면 안되는지](https://github.com/user-attachments/assets/35582c67-d848-4b13-8bb9-0a90480f75c0)

   
   **Warpage(뒤틀림)**은 칩이 프링글스 과자처럼 굽어 기판의 수만 개 접점이 제대로 맞물리지 못하게 만드는 현상이고, **Delamination(박리)**은 층 사이가 스티커처럼 벌어져 신호가 끊기거나 내부 열이 빠져나가지 못하게 가두는 현상입니다.
   
   i9-13900K 같은 고성능 CPU가 전기가 통하지 않는 '먹통'이 되거나, 내부 열을 못 견디고 타버리게 만들기 때문에 설계 단계에서 반드시 막아야 할 치명적인 결함입니다.

## 1. 프로젝트 레퍼런스

최근 반도체 패키징 분야에서는 유한요소해석(FEA)의 막대한 연산 비용을 극복하기 위해 심층신경망(DNN)을 활용한 휨(Warpage) 예측 연구[1]와 트리 기반 머신러닝을 적용한 열응력/방열 최적화 연구[2]가 활발히 진행되고 있으며,     
복잡한 물리적 비선형성을 정밀하게 학습하기 위해 CNN 기반의 데이터 주도형 모델링 기법도 도입되고 있습니다[3].      
이와 더불어 솔더 조인트 수명 예측, 설비 잔존 수명(RUL)을 위한 시계열 데이터 생성, 배터리의 기계적 파손 예측 등 공학적 신뢰성 평가 전반에 AI를 적용하는 최신 학계의 융합 연구 흐름[4,5,6]은 본 프로젝트의 방법론적 배경을 넓히는 기반이 되었습니다.     
본 프로젝트는 이러한 기존의 단순 순방향 예측(Forward Prediction)을 넘어, 가상의 완벽한 응력 상태를 정의한 '유토피아 타겟 텐서(Utopia Tensor)'와 오토인코더의 잠재 공간(Latent Space)을 활용하여 최적 설계의 초안을 즉시 도출하는 역설계(Inverse Design) 파이프라인을 제안합니다.    
특히 시계열 파동에서 가장 파괴적인 'Max Peak'만을 추출해 학습 효율을 극대화하고, GPR(Gaussian Process Regression)이 산출하는 예측 불확실성($\sigma$)을 유전 알고리즘(NSGA-II)의 강제 제약 조건(Hard Constraint)으로 부여함으로써,     
AI 특유의 꼼수(Reward Hacking)나 메쉬 붕괴 에러를 원천 차단하는 '제조 가능한 강건 최적화(Robust Optimization)'를 달성한 것이 본 연구의 가장 큰 차별점입니다.    
      
References[1] Panigrahy, S. K., Che, F. X., Ong, Y. C., Ng, H. W., & Kumar, G. (2025). Deep Learning Study on Memory IC Package Warpage Using Deep Neural Network and Finite Element Simulation. Chips, 4, 35.    
[2] Acharya, P. V., Lokanathan, M., Ouroua, A., Hebner, R., Strank, S., & Bahadur, V. (2018). Machine Learning-Based Predictions of Benefits of High Thermal Conductivity Encapsulation Materials for Power Electronics Packaging. Journal of Electronic Packaging, 140(4), 041109.    
[3] Yang, J., Wu, Y., & Liu, X. (2023). Proton Exchange Membrane Fuel Cell Power Prediction Based on Ridge Regression and Convolutional Neural Network Data-Driven Model. Sustainability, 15(14), 11010.    
[4] Akhtar, M. Z., Schmid, M., Zippelius, A., & Elger, G. (2025). Solder joint lifetime model using AI framework operating on FEA data. Engineering Failure Analysis, 167, 109032.    
[5] Ahn, G., Yun, H., Hur, S., & Lim, S. (2021). A Time-Series Data Generation Method to Predict Remaining Useful Life. Processes, 9(7), 1115.     
[6] Zhang, X. C., Yin, X. D., Huang, Z. X., Zhang, T., Ci, T. J., Li, C. Y., Wang, Q. L., & El-Rich, M. (2024). Mechanical Behavior and Failure Prediction of Cylindrical Lithium-Ion Batteries under Mechanical Abuse Using Data-Driven Machine Learning. SSRN Electronic Journal.     
 
   
## 2. 프로젝트 개요

![04  프로젝트 개요](https://github.com/user-attachments/assets/6feae6dd-7266-4372-bea2-7047a4df75c9)

반도체 패키지는 서로 다른 성질의 재료들이 겹겹이 쌓인 '초정밀 샌드위치'입니다. 열을 받으면 각 재료의 팽창 속도가 달라 두 가지 치명적인 문제가 발생합니다.

문제점: 이러한 불량을 확인하기 위한 기존 시뮬레이션은 한 번 수행에 너무 많은 시간과 비용이 소모됩니다.

해결: 이 뒤틀림을 예측하려면 복잡한 물리 계산(시뮬레이션)이 필요한데, 한 번 계산에 시간이 너무 오래 걸립니다. 그래서 우리는 결과를 순식간에 맞추는 AI를 가르치기로 했습니다.    
<br>                     

![04-1  프로젝트 개요](https://github.com/user-attachments/assets/6e8b63bd-4be3-4fb1-9f0d-f549877fae6c)

## 3. 대리 모델의 도입 (The "Surrogate Model" Concept)

![05  대리 모델의 도입](https://github.com/user-attachments/assets/215cd1b9-e564-494a-a393-93f6b7a6efd3)


본 프로젝트의 핵심은 복잡하고 오래걸리는 무거운 컴퓨터 시뮬레이션 작업를 대신할 **'대리 모델(Surrogate Model)'**을 구축하는 것입니다.

개념: 실제 물리 시뮬레이션은 정확하지만 너무 느립니다. 이를 대신해 입력값과 결과값 사이의 상관관계만 빠르게 계산하는 AI 모델을 만듭니다.

비유: 수학 문제를 풀 때마다 정석대로 모든 풀이 과정을 적는 대신, 수만 개의 문제 데이터를 학습해 **문제를 보자마자 답을 내놓는 '암산 천재'**를 옆에 두는 것과 같습니다.

장점: 기존 시뮬레이션으로 수 시간이 걸리던 검증을 0.1초 이내로 단축하여 실시간 설계 최적화를 가능하게 합니다.



## 4. 반도체 '샌드위치' 구조 (The Subject)
우리가 분석하는 모델은 아래와 같이 6가지 핵심 요소로 구성된 Lidded Package (뚜껑이 있는 패키지,Lidded FCBGA (Flip-Chip Ball Grid Array))입니다.   

![images](https://github.com/user-attachments/assets/20b24c1f-3e37-430e-bfe3-985b92d31987)    

<Fig.1 플립칩 모형 단면>     

![06  플립칩 명칭](https://github.com/user-attachments/assets/7010b81c-e037-4cd4-9653-319bcd8c90de)


<Fig.2 플립칩 단순화 모델 이미지>    
> 딥러닝에 유의미한 데이터를 모으기 위해 최대한 조건과 형상을 단순화 합니다. 실제 논문의 경우 3d 형상과 복잡한 조건으로 슈퍼컴퓨터를 사용합니다.

## 5. 📊 6-Dimension Geometric Parameter 정의서

본 데이터셋은 기하학적 파탄(Mesh Error)을 방지하고 딥러닝 모델이 유의미한 변수 경향성을 학습하기 위해 설정된 상/하한선(Bounds) 규격입니다.

| 분류 | 파라미터명 | 탐색 범위 (Min ~ Max) | 물리적 의미 및 설정 사유 |
| :--- | :--- | :--- | :--- |
| **기판/접착** | `P1 (Substrate)` | **0.80 ~ 1.20 mm** (기준 1.01) | 기판 층수(Layer) 변경에 따른 베이스 구조 강성 변화 범위 |
| **(언더필)** | `P2 (Underfill)` | **0.05 ~ 0.09 mm** (기준 0.07) | 솔더 범프(Solder Bump) 높이에 종속되므로 타이트한 공차 부여 |
| **(열/응력원)**| `P3 (Die)` | **0.60 ~ 0.85 mm** (기준 0.74) | 웨이퍼 백그라인딩(Back-grinding) 한계 및 표준 두께 반영 |
| **(지지부)** | `P4 (Adhesive)` | **0.10 ~ 0.30 mm** (기준 0.19) | 실런트 도포량 한계 (접착력 확보 및 다리 들뜸/파고듬 방지) |
| **상단 덮개** | `P5 (IHS roof)` | **1.20 ~ 1.80 mm** (기준 1.50) | 뚜껑 강성 확보 및 전체 패키지 규격(Height) 준수 마지노선 |
| **내부 코어** | `P6 (TIM)` | **0.03 ~ 0.08 mm** (기준 0.05) | 공정상 최소 도포 두께 한계 및 열 저항 급증 방지 |

## 6. 📊 22-Column Simulation Dataset 정의서

본 데이터셋은 패키지 신뢰성 예측 모델(대리 모델) 학습을 위해 설계 치수(원인)와 시계열 해석 결과(결과)를 병합한 최종 마스터 데이터셋입니다. (데이터 형태: `727,200 × 22`)

| 분류 | 컬럼명 | 설명 | 물리적 의미 |
| :--- | :--- | :--- | :--- |
| **설계 변수** | `P1` | IHS roof 두께 (mm) | 상단 덮개 강성 및 패키지 전체 규격 |
| (입력 특성) | `P2` | TIM 두께 (mm) | 열 전도체 도포량 (열 저항 제어) |
| | `P3` | Die 두께 (mm) | 반도체 칩 두께 (발열 및 응력원) |
| | `P4` | Underfill 두께 (mm) | 칩 하단 충격 흡수 및 응력 분산재 도포량 |
| | `P5` | Substrate 두께 (mm) | 베이스 기판 구조 강성 |
| | `P6` | Adhesive 두께 (mm) | 뚜껑 다리 고정용 실런트 도포량 |
| **기본 환경** | `Time` | 해석 시간 (0 ~ 300s) | 공정 시퀀스 타임라인 |
| (조건) | `TempBase` | 기준 온도 (℃) | 공정 온도 프로파일 |
| **전체 거동** | `WarpMax` | 패키지 최대 휨량 (mm) | Y축 방향 최대 변위 (Warpage) |
| **상단 계면** | `T_Tip_Peel` | 상단 끝점 수직 응력 (MPa) | Die-UF 계면 박리 응력 ($\sigma_y$) |
| (Die-UF) | `T_Tip_Shear` | 상단 끝점 전단 응력 (MPa) | Die-UF 계면 전단 응력 ($\tau_{xy}$) |
| | `T_Tip_SEQV` | 상단 끝점 폰-미세스 응력 | 해당 지점의 종합 응력 수준 |
| | `T_Tip_Strain` | 상단 끝점 총 변형률 | 누적 탄성+소성+크립 변형 |
| | `T_Avg_Peel` | 상단 계면 평균 수직 응력 | 계면 전체 박리 하중 수준 |
| | `T_Avg_Shear` | 상단 계면 평균 전단 응력 | 계면 전체 전단 하중 수준 |
| **하단 계면** | `B_Tip_Peel` | 하단 끝점 수직 응력 (MPa) | UF-Sub 계면 박리 응력 ($\sigma_y$) |
| (UF-Sub) | `B_Tip_Shear` | 하단 끝점 전단 응력 (MPa) | UF-Sub 계면 전단 응력 ($\tau_{xy}$) |
| | `B_Tip_SEQV` | 하단 끝점 폰-미세스 응력 | 해당 지점의 종합 응력 수준 |
| | `B_Tip_Strain` | 하단 끝점 총 변형률 | 하단 계면 누적 변형량 |
| | `B_Avg_Peel` | 하단 계면 평균 수직 응력 | 하단 전체 박리 하중 수준 |
| | `B_Avg_Shear` | 하단 계면 평균 전단 응력 | 하단 전체 전단 하중 수준 |
| **부품 파손** | `Die_SX` | 다이 중심 굽힘 응력 (MPa) | Die Crack 방지용 굽힘 응력 ($\sigma_x$) |

## 7. CAE 데이터 구현 과정

a. 3D 혹은 Fan에 의한 유동 해석 등이 포함되면 한개의 케이스 당 몇 시간씩 걸리게 됩니다.    
이 경우, 딥러닝을 위해 몇 천 개의 케이스를 해석하려면 가정용 PC 1대로는 몇 달씩 걸립니다.    
따라서 2D로 시뮬레이션을 작업하고, 대칭 형상이기 때문에 절반 자른 단면으로 해석을 진행합니다.    
또한, Fan에 의한 냉각이 반영되지 않으면 과열 상태가 되어버려 온도가 지나치게 높아지기 때문에, 윗면에 Fan에 의한 효과를 반영해 줍니다.    

b. flip-chip에 사용된 에폭시 접착제의 경우 고온경화 (높은 온도에서 굳음)이기 때문에 식으면서 발생하는 내부 응력을 반영하기 위해    
처음에 120도로 시작해 상온으로 냉각되고 100s부터 칩의 핵심인 die 에서 253W (해석의 기준 데이터인 intel i9-13900k 의 최대 출력)을 발생. 최대 100도로 유지되게 출력 제어     
200s가 되면 다시 출력을 끄고 상온으로 냉각. 냉각 이후의 영구 변형을 확인 한다.    

c. 뒤틀림(Warpage)을 확인하기 위해 y축 방향 변형(deformation)과 박리(Delamination)를 확인하기 위해 shear stress (수평방향), normal stress (수직방향)을 확인한다.    

d. 해석 결과를 바탕으로 주요 위치에 가상 센싱(하드웨어를 직접 설치하지 않고, 소프트웨어와 알고리즘을 사용)을 하여 데이터를 출력. 아래 22개 컬럼을 생성한다.    

e. 파트 별 두께를 패러미터로 하여 랜덤 생성해 해석된 데이터를 수집한다. 이때, 300s 동안, 케이스마다 변형되는 시계열 데이터로 저장된다.    

f. 50개의 케이스를 1개의 배치로 총 24개의 배치를 해석한다. 1개의 케이스에 23열 x 606열. 따라서 606 X 50 x 24 = 727,200개 

## 8. 📈 플립칩(Flip-Chip) 패키지 열변형 3-Step 시나리오 프로파일

본 해석은 딥러닝 대리모델 학습을 위해 '제조 공정 $\rightarrow$ 기기 작동 $\rightarrow$ 작동 종료'로 이어지는 패키지의 생애 주기(Life-Cycle)를 3단계로 압축하여 모사한 표준 프로파일입니다.

| 분류 | 해석 시간 | 열 하중 및 제어 방식 | 물리적 의미와 해석 목적 (왜 이렇게 설계했는가?) |
| :--- | :--- | :--- | :--- |
| **Step 1** (제조 냉각) | `0 - 100s` | **온도:** 120℃ $\rightarrow$ 22℃ 자연 냉각<br>**발열:** 없음 (Off) | **[초기 잔류응력 형성]** 언더필이 120℃에서 고온 경화되어 형태가 고정되었으므로, 120℃를 응력이 없는 상태(Stress-Free)로 기준 잡습니다. 이후 상온으로 냉각되면서 열팽창계수(CTE) 차이에 의해 발생하는 수축 응력과 초기 잔류응력을 모사합니다. |
| **Step 2** (작동 발열) | `100 - 200s` | **온도:** 다이 발열로 인한 승온<br>**발열:** APDL 쓰로틀링 (Step 2에서만 활성화) | **[열응력 발생 및 크립 이완]** 기기 전원이 켜지는 상태입니다. APDL을 통해 95℃-105℃ 구간에서 발열량(2745mW~500mW)이 오르내리며 온도를 제어합니다. 이때 고온 노출로 인해 시간에 따라 변형이 누적되는 크립(Creep)이 발생하며 응력이 점성 이완됩니다. |
| **Step 3** (작동 종료) | `200 - 300s` | **온도:** 22℃로 다시 냉각<br>**발열:** 강제 0으로 초기화 (Kill) | **[영구 변형 및 피로 누적 확인]** 기기 전원이 꺼지고 다시 냉각되는 상태입니다. Step 2에서 겪은 크립 거동 때문에 구조물은 Step 1 직후의 원래 형태론 돌아오지 못하고 영구 변형을 남깁니다. 딥러닝 모델은 이 사이클 전후의 변형량 차이(Delta)를 통해 패키지 수명을 학습합니다. |

<img width="1164" height="1600" alt="Sim 01 결과 그래프" src="https://github.com/user-attachments/assets/f46c6022-bece-4189-a28b-2f35ff3e8896" />

<Fig.3 케이스 1 - 임의 설계 치수 기준 stpe 별 수치 그래프>

## 9. 📊 22-Column Simulation Dataset 정의서

본 데이터셋은 패키지 신뢰성 예측 모델(대리 모델) 학습을 위해 설계 치수(원인)와 시계열 해석 결과(결과)를 병합한 최종 마스터 데이터셋입니다. (데이터 형태: `727,200 × 22`)

| 분류 | 컬럼명 | 설명 | 물리적 의미 |
| :--- | :--- | :--- | :--- |
| **설계 변수** | `P1` | IHS roof 두께 (mm) | 상단 덮개 강성 및 패키지 전체 규격 |
| (입력 특성) | `P2` | TIM 두께 (mm) | 열 전도체 도포량 (열 저항 제어) |
| | `P3` | Die 두께 (mm) | 반도체 칩 두께 (발열 및 응력원) |
| | `P4` | Underfill 두께 (mm) | 칩 하단 충격 흡수 및 응력 분산재 도포량 |
| | `P5` | Substrate 두께 (mm) | 베이스 기판 구조 강성 |
| | `P6` | Adhesive 두께 (mm) | 뚜껑 다리 고정용 실런트 도포량 |
| **기본 환경** | `Time` | 해석 시간 (0 ~ 300s) | 공정 시퀀스 타임라인 |
| (조건) | `TempBase` | 기준 온도 (℃) | 공정 온도 프로파일 |
| **전체 거동** | `WarpMax` | 패키지 최대 휨량 (mm) | Y축 방향 최대 변위 (Warpage) |
| **상단 계면** | `T_Tip_Peel` | 상단 끝점 수직 응력 (MPa) | Die-UF 계면 박리 응력 ($\sigma_y$) |
| (Die-UF) | `T_Tip_Shear` | 상단 끝점 전단 응력 (MPa) | Die-UF 계면 전단 응력 ($\tau_{xy}$) |
| | `T_Tip_SEQV` | 상단 끝점 폰-미세스 응력 | 해당 지점의 종합 응력 수준 |
| | `T_Tip_Strain` | 상단 끝점 총 변형률 | 누적 탄성+소성+크립 변형 |
| | `T_Avg_Peel` | 상단 계면 평균 수직 응력 | 계면 전체 박리 하중 수준 |
| | `T_Avg_Shear` | 상단 계면 평균 전단 응력 | 계면 전체 전단 하중 수준 |
| **하단 계면** | `B_Tip_Peel` | 하단 끝점 수직 응력 (MPa) | UF-Sub 계면 박리 응력 ($\sigma_y$) |
| (UF-Sub) | `B_Tip_Shear` | 하단 끝점 전단 응력 (MPa) | UF-Sub 계면 전단 응력 ($\tau_{xy}$) |
| | `B_Tip_SEQV` | 하단 끝점 폰-미세스 응력 | 해당 지점의 종합 응력 수준 |
| | `B_Tip_Strain` | 하단 끝점 총 변형률 | 하단 계면 누적 변형량 |
| | `B_Avg_Peel` | 하단 계면 평균 수직 응력 | 하단 전체 박리 하중 수준 |
| | `B_Avg_Shear` | 하단 계면 평균 전단 응력 | 하단 전체 전단 하중 수준 |
| **부품 파손** | `Die_SX` | 다이 중심 굽힘 응력 (MPa) | Die Crack 방지용 굽힘 응력 ($\sigma_x$) |

# 🚀 [Master Guide] 2D 열기계 해석 기반 패키지 최적 설계 파이프라인


## 📌 10. 프로젝트 사전 정보 (Project Context & Meta-Data)

### 10.1. 도메인 및 물리적 배경
* **분야:** 반도체 패키징 / 복합재 구조의 2D 열기계 유한요소해석(Thermo-mechanical FEA)
* **현상:** 열팽창계수(CTE) 불일치로 인한 휨(Warpage) 및 층간 박리(Delamination), 다이 크랙(Die Crack, 다이(Die)가 깨거나 약간 깨어지면 발생하는 일반적인 동전 오류 유형) 발생.
* **조건:** 300초에 걸친 극렬한 온도 사이클링(가열-유지-냉각 3 Steps).

### 10.2. 데이터셋 구조 (X와 Y)
* **입력 변수 (X):** `P1 ~ P6` (6개). 각 층의 기하학적 두께(Thickness)를 의미. 범위는 대략 0.1 ~ 1.0.
* **출력 변수 (Y):** 300초 동안 기록된 16개의 시계열(Time-series) 응력/변형 데이터.
* **핵심 Y 변수의 물리적 의미 (Named Selection 위치 기준):**
  1. `WarpMax`: 패키지 전체의 최대 열변형량 (최소화 메인 타겟)
  2. `T_Tip_Peel`: 계면 끝단의 수직 응력. **박리(Delamination)**의 직접적 원인 (최소화 메인 타겟)
  3. `T_Tip_Shear`: 계면 엇갈림 응력. **계면 피로(Fatigue, 복합재료나 이종 재료의 접합 계면이 반복적인 하중(응력)을 받아 균열이 발생하고 성장하여, 최종적으로 접합부가 파손되는 현상)** 유발
  4. `T_Tip_SEQV (Von Mises)`: 끝단 등가 응력. **소성 변형(Plasticity, 고체 재료가 탄성 한계 이상의 하중을 받아, 힘을 제거해도 원래 모양으로 돌아가지 않고 영구적으로 모양이 바뀌는 현상)** 유발
  5. `T_Avg_Peel` / `T_Avg_Shear`: 접합면 전체 평균 응력. **중앙부 거품(Void)** 유발
  6. `Die_SX` (Die Bending Stress): 실리콘 칩 본체의 휨 응력. **다이 크랙(Die Crack)** 유발
  7. `Die_Corner_Stress`: 칩 모서리 응력 집중도
 
* 응력: 물체에 외부의 힘(외력)이 작용할 때, 그 내부에서 변형에 저항하여 구조를 유지하려는 단위 면적당 내력

### 10.3. 데이터 결측치 (Missing Data / Infeasibility)
* 계획된 1200개의 DP(Design Point) 중 약 **29%는 시뮬레이션 중 메쉬 꼬임이나 심각한 박리로 인해 해석이 터져버림 (결측치 발생).**
* 살아남은 71%의 데이터만 시계열 CSV 파일(`ML_DATA_Extract_Row_N.csv`)로 존재함.

---

## 📌 11. 6단계 최적화 파이프라인 (Action Plan)

본 프로젝트는 단순히 대리 모델을 만드는 것을 넘어, 역설계(Inverse Design)와 유전 알고리즘(GA)을 결합하여 최적의 P1~P6를 도출하는 End-to-End 프레임워크입니다.

### Step 1: 대리 모델(Surrogate)을 통한 데이터 증강 (Data Augmentation)
* **목표:** 800여 개의 생존 데이터 한계를 극복하기 위해 가상 데이터를 생성.
* **방법:** 생존한 데이터의 '절댓값 Max Peak' 지표들을 추출하여 머신러닝(XGBoost, GPR, Tabular ResNet) 학습. 이후 난수 생성기(LHS, Bayesian Optimization)로 10만 개의 가상 P1~P6 조합을 만들고 Y값들을 예측함.
* **사용된 기법 및 알고리즘:** XGBOOST, GPR + ARD 커널, Tabular ResNet + Permutation Feature Importance, 난수화 기법 (LHS, Bayesian Optimization), MinMaxScaler

### Step 2: 은닉 제약조건 분류기(Gatekeeper)를 통한 필터링
* **목표:** 물리적으로 파괴되는(해석이 터지는) 치수 조합을 사전에 걸러냄.
* **방법:** 원본 1200개 DP의 CSV 파일 존재 여부를 1(Safe)과 0(Fail)으로 라벨링하여 Random Forest 분류기 학습. Step 1에서 만든 10만 개의 가상 조합을 이 분류기에 통과시켜, **생존 확률 95% 이상인 안전한 가상 데이터**만 필터링함.
* **사용된 기법 및 알고리즘:** Random Forest

### Step 3: 파레토 프론티어 (Pareto Frontier) 타겟 곡선 추출
* **목표:** 역설계 AI에 입력할 '물리적으로 도달 가능하면서도 완벽에 가까운 타겟 시계열 텐서' 생성.
* **[🚨 경고 - 치명적 오류 주의]:** * 300초 '평균값'이 아닌 **반드시 시계열 전체의 '절댓값 최대 피크(Max Peak)'**를 기준으로 우수 데이터를 선별할 것. [Image of thermal cycling stress hysteresis loop]
  * Von Mises 등가 응력이 아닌 **`WarpMax`와 `T_Tip_Peel` 단 2가지만을 기준**으로 파레토 최적 DP(상위 5~10%)를 선별할 것.
* **추출 및 스케일링 방법:** 선정된 파레토 DP의 300초 Raw 시계열 곡선을 8~9개 주요 채널 모두 통째로 가져옴. 이후 모든 채널에 **동일한 스칼라(예: x0.9)**를 곱하여 진폭만 10% 낮춘 '유토피아 타겟 다채널 텐서'를 생성. (물리적 위상차 보존)
* **사용된 기법 및 알고리즘:** 파레토 비지배 정렬

### Step 4: 딥러닝 기반 역설계 (Inverse Design) 초안 출력
* **목표:** 타겟 곡선을 입력하면 이를 구현할 수 있는 최적의 P1~P6 초안(Draft) 도출.
* **방법:** 1D-CNN 역방향 모델 또는 오토인코더(Autoencoder) 잠재 매핑 모델을 사용. Step 3에서 만든 '유토피아 타겟 텐서'를 입력하면 AI가 1차 P1~P6 설계안을 한 번의 연산으로 출력함. (8채널 동시 입력 구조로 전체 응력 밸런스 학습)
* **사용된 기법 및 알고리즘:** 시계열 리샘플링, 사비츠키-골레이 필터, 1D-CNN 오토인코더, Residual Block (ResNet),  U-Net Skip Connection,  Upsample + Conv1d, Smooth L1 Loss, Total Variation Loss, 지도형 오토인코더, 다층 퍼셉트론 (MLP) 역매핑, StandardScaler

### Step 5: 머신러닝 미세 튜닝 (Fine-tuning via GA & Penalty Limits)
* **목표:** 도출된 초안을 바탕으로 유전 알고리즘(NSGA-II)을 돌려 최종 최적화 및 물리적 한계치(Limit) 강제 적용.
* **방법:** 1. Step 4의 초안 치수 기준 $\pm 10\%$ 내외로 좁은 탐색 바운더리 설정.
  2. 목적 함수(Loss)는 `WarpMax`와 `T_Tip_Peel` 가중치 합산으로 최소화.
  3. **[🚨 필수 - Hard Constraints]:** 최적화 과정에서 나머지 응력들이 재료의 물리적 한계치(Limit)를 넘을 경우, Loss에 +999,999점의 페널티를 부여하여 즉시 도태시킴.
     * Limit 예시: `Die_SX` < 실리콘 파괴 인성, `T_Tip_Shear` < 계면 피로 한계 등. [Image of Pareto front in multi-objective optimization]
* **사용된 기법 및 알고리즘:** NSGA-II (다목적 유전 알고리즘), 강건 최적화, Feasibility Rule (`pymoo` `G` Matrix),  Knee Point (최적 밸런스 점 추출)

### Step 6: 디지털 트윈 (Digital Twin) 최종 시뮬레이션 검증
* **목표:** AI가 제시한 최종 P1~P6 조합을 실제 Ansys 2D 해석에 입력.
* **방법:** 도출된 시계열 곡선 결과를 베이스라인(초기 설계) 및 Step 3의 유토피아 타겟과 중첩 플롯(Overlay Plot)하여 휨 및 박리 응력의 저감률을 검증.


## 12. 결과 분석

### 12.1 

### 12.2 최적 설계 전후 비교

### 12.3 대리모델 성능평가


## 13. 개선 방안

* 13.1 패러미터의 변경
  
  이번 실험에서 선정한 두께 패러미터는 실제로는 파운드리나 칩 설계사가 이미 고정해놓은 상수로 변경이 어렵습니다.
두께의 경우 소숫점 단위의 최적 수치를 도출하더라도 절대 이와 정확한 수치를 적용하기 어렵습니다. 현실적으로 가공오차는 필연적이며 정밀가공을 하더라도 단가나 납기의 문제가 발생합니다.
warpage의 경우 핵심 요인이 CTE(열팽창계수 차이), 온도변화량, 탄성계수, 두께 이기 때문에 EMC의 배합(CTE), 공정온도 (이번 실험에서는 120 -> 22도로 고정), si-bridge 두께나 substrtate의 내부 구조 등이 더 좋은 패러미터 입니다.
(reference. Artificial Intelligence-Based Warpage Prediction Model for Accelerating Thermo-Mechanical Simulation in Advanced Packaging,  2025 IEEE 75th Electronic Components and Technology Conference (ECTC).)

* 13.2 CAE 시뮬레이션 고도화
  
  본 프로젝트에서는 2D 모델링을 사용하여 기판 등 파트의 형상의 복잡성을 단순화 하였습니다.
또한, 팬에 의한 냉각 같은 유동 해석 문제도 간략화를 통해 생략했습니다.
실제 현실에 가까운 결과를 원할 경우 실제 3d 형상에 유동해석도 포함해야 합니다만
이 경우 실제 논문들에서 사용하는 슈퍼 컴퓨터를 사용하지 않으면 몇 개월을 24시간 풀로 돌려도 충분한 데이터를 얻을수 없습니다

* 13.3 대리모델의 방향성

   본래 기존의 대리모델의 개념에 맞게 구현하려면 모든 time step에 대응하는 시계열 데이터를 예측하고 구현하는 대리모델을 만들어야 하지만 이는 논문 수준의 매우 어려운 작업이기에 이번에는 절대값의 최대치만 예측하는 대리모델을 구현했습니다.
하지만 높은 정확도의 시계열 데이터를 예측하는 대리모델을 구현할 경우 step4 에서 학습시킬때 더 많은 데이터를 기반으로 더 정확한 결과를 도출할 수 있습니다.















