# Flipchip_surrogate
# 🚀 AI 기반 반도체 패키지 신뢰성 예측 프로젝트
## 반도체가 열변형에 의해 휘어지고 깨지고 뜯어지는 것을 ai로 예측합니다.
<br>

## 0. 키워드 설명

### 1) CAE (Computer-Aided Engineering)란 무엇인지?

![01  CAE란_2](https://github.com/user-attachments/assets/a42065d8-5f65-43a8-8534-10a495136957)

   컴퓨터 속 가상 세계에서 그 물건이 **튼튼한지, 열에 잘 견디는지 등을 미리 시뮬레이션해 보는 '디지털 모의실험'입니다.
   
   CG나 3D 모델링에서 배우의 얼굴이나 몸에 점을 찍어 동작을 읽어내는 '트래커(Tracker)'를 떠올려 보세요.
   
   CAE의 메쉬는 그 점 하나하나마다 물리 법칙(힘, 열, 압력)을 계산하기 위한 계산 포인트입니다.

   
### 2) flipchip이 무엇인지?
 
![02  플립 칩이란](https://github.com/user-attachments/assets/cd8d6bb6-e3c8-49c0-a294-5bbf95eb10b2)

   "플립칩(Flip-Chip)"은 반도체 칩을 기판에 연결할 때 칩을 뒤집어(Flip) 표면의 수많은 돌기(Bump)를 기판에 직접 맞닿게 붙이는 연결 기술입니다.
   
   이번 프로젝트는 제 pc의 cpu인 i9-13900k를 모델로 했고 이 cpu가 플립칩 타입입니다.
        
### 3) 왜 변형되면 안되는지

![03  왜 변형이 되면 안되는지_2](https://github.com/user-attachments/assets/99de2347-4fb1-48fa-b21c-aa4fc2f468a3)

   
   **Warpage(뒤틀림)**은 칩이 프링글스 과자처럼 굽어 기판의 수만 개 접점이 제대로 맞물리지 못하게 만드는 현상이고, **Delamination(박리)**은 층 사이가 스티커처럼 벌어져 신호가 끊기거나 내부 열이 빠져나가지 못하게 가두는 현상입니다.
   
   i9-13900K 같은 고성능 CPU가 전기가 통하지 않는 '먹통'이 되거나, 내부 열을 못 견디고 타버리게 만들기 때문에 설계 단계에서 반드시 막아야 할 치명적인 결함입니다.

### 4) 오토 인코더

<img width="682" height="372" alt="오토인코더" src="https://github.com/user-attachments/assets/4b7c65b7-aa15-475b-ae84-4592de3dbcac" />

1. 인코더 (Encoder) : "1,000쪽짜리 책을 1장으로 요약하기"데이터가 너무 크고 복잡하면 AI도 헷갈립니다. 그래서 복잡한 시뮬레이션 파도 그래프(1,000쪽짜리 책)를 집어넣으면, 쓸데없는 내용은 다 버리고 가장 중요한 핵심 내용만 딱 1장으로 압축 요약해 주는 역할을 합니다.    
2. 잠재 공간과 잠재 변수 $z$ : "핵심 다이얼이 모여있는 조종판"1장으로 요약된 핵심 내용들이 바로 **잠재 변수($z$)**입니다. 비유하자면, 복잡한 데이터를 마음대로 조절할 수 있는 **'핵심 조종판의 다이얼들'**이라고 생각하시면 됩니다. 이 다이얼들이 모여있는 곳이 잠재 공간입니다.    
3. 보간과 유토피아 : "다이얼을 돌려 상상하기 & 꿈의 숫자 입력하기"우리는 이 조종판의 다이얼을 이리저리 돌려보면서(보간), 기존에 없던 새로운 상태를 상상해 볼 수 있습니다. 특히 우리 프로젝트에서는 과감하게 "응력 0, 휨 0"이라는 현실엔 없는 완벽한 꿈의 숫자(유토피아)로 다이얼을 강제로 휙 돌려버렸습니다.    
4. 디코더 (Decoder) : "요약본을 보고 완전히 새로운 책 써내기"디코더는 반대로 작동합니다. 우리가 방금 다이얼을 '유토피아(완벽한 0)'로 맞춰놓고 작동 버튼을 누르면, 디코더는 그 1장짜리 요약 힌트만 보고 "아하, 이런 완벽한 결과를 원해? 그럼 설계도를 이렇게 그려야 해!"라며 완전히 새로운 1,000쪽짜리 정답 책(설계도 초안)을 써내는 역할을 합니다.     

## 1. 프로젝트 레퍼런스

최근 반도체 패키징 분야에서는 **유한요소해석(FEA)의 막대한 연산 비용을 극복하기 위해 심층신경망(DNN)을 활용**한 휨(Warpage) 예측 연구[1]와 **트리 기반 머신러닝을 적용한 열응력/방열 최적화** 연구[2]가 활발히 진행되고 있으며,     
복잡한 물리적 비선형성을 정밀하게 학습하기 위해 **CNN 기반의 데이터 주도형 모델링 기법**도 도입되고 있습니다[3].      
이와 더불어 솔더 조인트 수명 예측, 설비 잔존 수명(RUL)을 위한 시계열 데이터 생성, 배터리의 기계적 파손 예측 등 공학적 신뢰성 평가 전반에 AI를 적용하는 최신 학계의 융합 연구 흐름[4,5,6]은 본 프로젝트의 방법론적 배경을 넓히는 기반이 되었습니다.     
본 프로젝트는 이러한 기존의 단순 순방향 예측(Forward Prediction)을 넘어, 가상의 완벽한 응력 상태를 정의한 **'유토피아 타겟 텐서(Utopia Tensor)'** 와 오토인코더의 **잠재 공간(Latent Space)** 을 활용하여 최적 설계의 초안을 즉시 도출하는 **역설계(Inverse Design)** 파이프라인을 제안합니다.    
특히 시계열 파동에서 가장 파괴적인 'Max Peak'만을 추출해 학습 효율을 극대화하고, GPR(Gaussian Process Regression)이 산출하는 예측 불확실성($\sigma$)을 **유전 알고리즘(NSGA-II)의 강제 제약 조건(Hard Constraint)** 으로 부여함으로써,     
AI 특유의 꼼수(Reward Hacking)나 메쉬 붕괴 에러를 원천 차단하는 **제조 가능한 강건 최적화(Robust Optimization)** 를 달성한 것이 본 연구의 가장 큰 차별점입니다.    
      
References[1] Panigrahy, S. K., Che, F. X., Ong, Y. C., Ng, H. W., & Kumar, G. (2025). Deep Learning Study on Memory IC Package Warpage Using Deep Neural Network and Finite Element Simulation. Chips, 4, 35.    
[2] Acharya, P. V., Lokanathan, M., Ouroua, A., Hebner, R., Strank, S., & Bahadur, V. (2018). Machine Learning-Based Predictions of Benefits of High Thermal Conductivity Encapsulation Materials for Power Electronics Packaging. Journal of Electronic Packaging, 140(4), 041109.    
[3] Yang, J., Wu, Y., & Liu, X. (2023). Proton Exchange Membrane Fuel Cell Power Prediction Based on Ridge Regression and Convolutional Neural Network Data-Driven Model. Sustainability, 15(14), 11010.    
[4] Akhtar, M. Z., Schmid, M., Zippelius, A., & Elger, G. (2025). Solder joint lifetime model using AI framework operating on FEA data. Engineering Failure Analysis, 167, 109032.    
[5] Ahn, G., Yun, H., Hur, S., & Lim, S. (2021). A Time-Series Data Generation Method to Predict Remaining Useful Life. Processes, 9(7), 1115.     
[6] Zhang, X. C., Yin, X. D., Huang, Z. X., Zhang, T., Ci, T. J., Li, C. Y., Wang, Q. L., & El-Rich, M. (2024). Mechanical Behavior and Failure Prediction of Cylindrical Lithium-Ion Batteries under Mechanical Abuse Using Data-Driven Machine Learning. SSRN Electronic Journal.     
 
   
## 2. 프로젝트 개요

<img width="1380" height="752" alt="04  프로젝트 개요_2" src="https://github.com/user-attachments/assets/f8d7c348-78d1-481c-a2c4-ac434cec3663" />


반도체 패키지는 서로 다른 성질의 재료들이 겹겹이 쌓인 '초정밀 샌드위치'입니다. 열을 받으면 각 재료의 팽창 속도가 달라 두 가지 치명적인 문제가 발생합니다.

문제점: 이러한 불량을 확인하기 위한 기존 시뮬레이션은 한 번 수행에 너무 많은 시간과 비용이 소모됩니다.

해결: 이 뒤틀림을 예측하려면 복잡한 물리 계산(시뮬레이션)이 필요한데, 한 번 계산에 시간이 너무 오래 걸립니다. 그래서 우리는 결과를 순식간에 맞추는 AI를 가르치기로 했습니다.    
<br>                     

![04-1  프로젝트 개요_2](https://github.com/user-attachments/assets/f9c069b0-ecab-4e7f-96d2-541d1f5b8834)


## 3. 대리 모델의 도입 (The "Surrogate Model" Concept)

![05  대리 모델의 도입_2](https://github.com/user-attachments/assets/96a14396-a356-4238-b592-41d250841162)



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

<img width="1800" height="900" alt="normal stress 300s(1)" src="https://github.com/user-attachments/assets/1da6fae9-a988-4299-b1bc-0e88e8af8034" />
<img width="1800" height="900" alt="shear stress 300s(1)" src="https://github.com/user-attachments/assets/37fea198-0b02-4556-88dc-95ff05c07477" />
<img width="1800" height="900" alt="directional deformation 300s(1)" src="https://github.com/user-attachments/assets/f10880a2-2c85-4b9e-a789-6857952e4d76" />

<Fig.3 CAE 해석 결과 예시>

## 8. 📈 플립칩(Flip-Chip) 패키지 열변형 3-Step 시나리오 프로파일

본 해석은 딥러닝 대리모델 학습을 위해 '제조 공정 $\rightarrow$ 기기 작동 $\rightarrow$ 작동 종료'로 이어지는 패키지의 생애 주기(Life-Cycle)를 3단계로 압축하여 모사한 표준 프로파일입니다.

| 분류 | 해석 시간 | 열 하중 및 제어 방식 | 물리적 의미와 해석 목적 (왜 이렇게 설계했는가?) |
| :--- | :--- | :--- | :--- |
| **Step 1** (제조 냉각) | `0 - 100s` | **온도:** 120℃ $\rightarrow$ 22℃ 자연 냉각<br>**발열:** 없음 (Off) | **[초기 잔류응력 형성]** 언더필이 120℃에서 고온 경화되어 형태가 고정되었으므로, 120℃를 응력이 없는 상태(Stress-Free)로 기준 잡습니다. 이후 상온으로 냉각되면서 열팽창계수(CTE) 차이에 의해 발생하는 수축 응력과 초기 잔류응력을 모사합니다. |
| **Step 2** (작동 발열) | `100 - 200s` | **온도:** 다이 발열로 인한 승온<br>**발열:** APDL 쓰로틀링 (Step 2에서만 활성화) | **[열응력 발생 및 크립 이완]** 기기 전원이 켜지는 상태입니다. APDL을 통해 95℃-105℃ 구간에서 발열량(2745mW~500mW)이 오르내리며 온도를 제어합니다. 이때 고온 노출로 인해 시간에 따라 변형이 누적되는 크립(Creep)이 발생하며 응력이 점성 이완됩니다. |
| **Step 3** (작동 종료) | `200 - 300s` | **온도:** 22℃로 다시 냉각<br>**발열:** 강제 0으로 초기화 (Kill) | **[영구 변형 및 피로 누적 확인]** 기기 전원이 꺼지고 다시 냉각되는 상태입니다. Step 2에서 겪은 크립 거동 때문에 구조물은 Step 1 직후의 원래 형태론 돌아오지 못하고 영구 변형을 남깁니다. 딥러닝 모델은 이 사이클 전후의 변형량 차이(Delta)를 통해 패키지 수명을 학습합니다. |

<img width="1164" height="1600" alt="Sim 01 결과 그래프" src="https://github.com/user-attachments/assets/f46c6022-bece-4189-a28b-2f35ff3e8896" />

<Fig.4 케이스 1 - 임의 설계 치수 기준 stpe 별 수치 그래프>

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
* **이유:** 가정용 pc로 생성하는 데이터양은 제한적이다. 2d로 최대한 간단한 시뮬레이션을 했지만 24시간 pc를 끄지않고 6일동안 884개의 데이터만 얻을수 있었다. 따라서 step 1에서 대리모델을 위한 ai 모델 학습에 필요한 충분한 데이터양을 위해 데이터 증강을 한다.
* **사용된 기법 및 알고리즘:** XGBOOST, GPR + ARD 커널, Tabular ResNet + Permutation Feature Importance, 난수화 기법 (LHS, Bayesian Optimization), MinMaxScaler, 5-Fold CV

### Step 2: 은닉 제약조건 분류기(Gatekeeper)를 통한 필터링
* **목표:** 물리적으로 파괴되는(해석이 터지는) 치수 조합을 사전에 걸러냄.
* **방법:** 원본 1200개 DP의 CSV 파일 존재 여부를 1(Safe)과 0(Fail)으로 라벨링하여 Random Forest 분류기 학습. Step 1에서 만든 10만 개의 가상 조합을 이 분류기에 통과시켜, **결측치가 발생하지 않는 정상적인 형상의 데이터**만 필터링함.
* **이유:** step 1에서 데이터 증강을 하였지만 적은 데이터를 바탕으로 많은 데이터를 생성했지 때문에 신뢰도가 높지 않다. 실제 사용가능한 데이터를 분류하여 결측치가 생성되는 것을 방지한다.
* **사용된 기법 및 알고리즘:** Random Forest

### Step 3: 파레토 프론티어 (Pareto Frontier) 타겟 곡선 추출
* **목표:** 역설계 AI에 입력할 '물리적으로 도달 가능하면서도 완벽에 가까운 타겟 시계열 텐서' 생성.
* **[🚨 경고 - 치명적 오류 주의]:** * 300초 '평균값'이 아닌 **반드시 시계열 전체의 '절댓값 최대 피크(Max Peak)'**를 기준으로 우수 데이터를 선별할 것. [Image of thermal cycling stress hysteresis loop]
  * Von Mises 등가 응력이 아닌 **`WarpMax`와 `T_Tip_Peel` 단 2가지만을 기준**으로 파레토 최적 DP(상위 5~10%)를 선별할 것.
* **추출 및 스케일링 방법:** 선정된 파레토 DP의 300초 Raw 시계열 곡선을 8~9개 주요 채널 모두 통째로 가져옴. 이후 모든 채널에 **동일한 스칼라(예: x0.9)**를 곱하여 진폭만 10% 낮춘 '유토피아 타겟 다채널 텐서'를 생성. (물리적 위상차 보존)
* **이유:** 본 프로젝트는 휘어짐, 박리 및 다양한 복수의 타겟이 존재하는 케이스로 한가지 정답으로 수렴되지 않는다. 따라서 파레토 프론티어라는 복수의 타겟 곡선을 목표로 한다.
* **사용된 기법 및 알고리즘:** 파레토 비지배 정렬

### Step 4: 딥러닝 기반 역설계 (Inverse Design) 초안 출력
* **목표:** 타겟 곡선을 입력하면 이를 구현할 수 있는 최적의 P1~P6 초안(Draft) 도출.
* **방법:** 1D-CNN 역방향 모델 또는 오토인코더(Autoencoder) 잠재 매핑 모델을 사용. Step 3에서 만든 '유토피아 타겟 텐서'를 입력하면 AI가 1차 P1-P6 설계안을 한 번의 연산으로 출력함. (8채널 동시 입력 구조로 전체 응력 밸런스 학습)
* **사용된 기법 및 알고리즘:** 시계열 리샘플링, 사비츠키-골레이 필터, 1D-CNN 오토인코더, Residual Block (ResNet),  U-Net Skip Connection,  Upsample + Conv1d, Smooth L1 Loss, Total Variation Loss, 지도형 오토인코더, 다층 퍼셉트론 (MLP) 역매핑, StandardScaler
* **이유:** 15개의 물리적 채널(WarpMax, Peel 등)이 300초 동안 617개의 시간대별로 기록된 방대한 시계열 데이터로 방대한 날것의 데이터를 그대로  역추적하라고 하면 연산량이 폭발하고 과적합이 발생합니다. 오토인코더는 압축하여 노이즈는 버리고 "온도가 떨어질 때 휨이 어떻게 변하는지"에 대한 물리 법칙의 핵심만 남기는 완벽한 전처리 역할을 수행합니다.

### Step 5: 머신러닝 미세 튜닝 (Fine-tuning via GA & Penalty Limits)
* **목표:** 도출된 초안을 바탕으로 유전 알고리즘(NSGA-II)을 돌려 최종 최적화 및 물리적 한계치(Limit) 강제 적용.
* **방법:** 1. Step 4의 초안 치수 기준 $\pm 10\%$ 내외로 좁은 탐색 바운더리 설정.
  2. 목적 함수(Loss)는 `WarpMax`와 `T_Tip_Peel` 가중치 합산으로 최소화.
  3. **[🚨 필수 - Hard Constraints]:** 최적화 과정에서 나머지 응력들이 재료의 물리적 한계치(Limit)를 넘을 경우, Loss에 +999,999점의 페널티를 부여하여 즉시 도태시킴.
     * Limit 예시: `Die_SX` < 실리콘 파괴 인성, `T_Tip_Shear` < 계면 피로 한계 등. [Image of Pareto front in multi-objective optimization]
* **이유:** Step 4에서 도출된 역설계 초안의 물리적 모순을 교정하고, GPR 불확실성 제약 조건을 통해 제조 불가능한 형상(외삽)을 배제하며 상충하는 지표 간의 최적 타협점(Knee Point)을 도출하기 위함입니다
* **사용된 기법 및 알고리즘:** NSGA-II (다목적 유전 알고리즘), 강건 최적화, Feasibility Rule (`pymoo` `G` Matrix),  Knee Point (최적 밸런스 점 추출)

### Step 6: 디지털 트윈 (Digital Twin) 최종 시뮬레이션 검증
* **목표:** AI가 제시한 최종 P1~P6 조합을 실제 Ansys 2D 해석에 입력.
* **방법:** 도출된 시계열 곡선 결과를 베이스라인(초기 설계) 및 Step 3의 유토피아 타겟과 중첩 플롯(Overlay Plot)하여 휨 및 박리 응력의 저감률을 검증.


## 12. 결과 분석

### 12.1 step 별 시각화 그래프 및 분석

### Step 1: 대리 모델(Surrogate)을 통한 데이터 증강 (Data Augmentation)
Case A : XGboost + LHS, Case B: GPR + ARD 커널 + LHS, Case C: Tabular ResNet + Bayesian Optimization 로 나누어 진행
<img width="2151" height="1183" alt="image" src="https://github.com/user-attachments/assets/dc245223-a80f-4032-9c10-9664ab29fd98" />

 <Fig 5. Y 변수 피크값 분포 히스토그램>
 
각 응력/변형 채널의 피크 분포를 확인하여 편향(skew)이나 이상치 진단

<img width="1481" height="471" alt="image" src="https://github.com/user-attachments/assets/8a7445b0-93b0-4885-8f15-ebb9b120f678" />

<Fig 6. 상관계수 히트맵>

어떤 두께 변수(P)가 어떤 응력(Y)에 강하게 영향을 미치는지 파악

<img width="2391" height="591" alt="image" src="https://github.com/user-attachments/assets/e6281223-8201-4bb3-b980-8f7a4dc5a2e8" />

<Fig 7. GPR의 모델 성능 평가>

[Graph A] 변수별 학습 성취도 (R²)
거시적인 휨(Warpage) 물리 법칙은 AI가 완벽히 파악(R² $\approx$ 1.0), 일부 변수는 미시적 비선형성을 가져 AI가 경향성은 알겠는데 완벽한 수식은 못 찾았다.   
이를 통해 R2 score가 높은 핵심 7개 변수 WarpMax, T_Tip_Peel, Die_SY_Max, B_Avg_Peel, B_Tip_SEQV, T_Tip_Strain, T_Tip_SEQV을 중심으로 해석을 진행합니다.  

[Graph B] 실제 vs 예측 산점도 (예측 해상도)
휨 예측은 100%에 가까운 디지털 트윈 수준으로 일치하지만, 박리 응력은 올바른 방향성(트렌드)만 잡을 뿐 핀포인트 수치 예측에는 다소 넓게 흩뿌려져(Scatter) 있습니다.

[Graph C] GPR 불확실성(σ) 분포 (강건 설계의 핵심
AI가 박리 응력처럼 자신이 예측하기 어려운 구간(높은 σ)을 스스로 인지하고 수치화한 결과입니다. WarpMax (파란색 좁은 탑): 불확실성($\sigma$)이 0 근처에 뾰족하게 몰려 있습니다.
T_Tip_Peel (주황색 넓은 산): 불확실성($\sigma$)이 우측으로 넓게 퍼져 있습니다.
이 주황색 넓은 산(높은 $\sigma$)의 존재를 확인했기 때문에, Step 5 유전 알고리즘에서 목적함수 + 2$\sigma$라는 강건 최적화를 적용하게 되었습니다.

<img width="1911" height="948" alt="image" src="https://github.com/user-attachments/assets/c37d49ae-934f-497d-ab4b-6036d4d17fd9" />

<Fig 8. GPR + ARD 커널 + LHS의 증강 데이터 품질 검증>

<img width="1911" height="948" alt="image" src="https://github.com/user-attachments/assets/7b8faaa1-8cd5-4dda-93c1-e3d488072dc2" />

<Fig 9. XGboost + LHS의 증강 데이터 품질 검증>

<img width="2151" height="1012" alt="image" src="https://github.com/user-attachments/assets/69e8cc62-8f35-43f0-8378-e0653f425217" />

<Fig 10. Tabular Resnet + Bayesian Optimization의 증강 데이터 품질 검증>

x축은 예측된 물리량, Y축 데이터 빈도수
LHS의 경우 골고루 데이터를 난수화 하다보니 극단 적인 형상도 포함 되어 양 끝단 뿔 형상이 warpmax 그래프에서 확인이 가능하다. 
XGBoost경우 트리 기반 모델의 특성으로 뿔 형상이 강화된다.
Bayesian Optimization은 최적화 과정이 포함되기 대문에 최적 조건에 몰리는  형상을 확인 가능하다.

### Step 2: 은닉 제약조건 분류기(Gatekeeper)를 통한 필터링

**Random Forest** (n_estimators=300, max_depth=7, class_weight='balanced')를 통해 분류된 결측치 생성 예측 데이터는 제거한다. -> 결과 20% 제거

<img width="1911" height="948" alt="image" src="https://github.com/user-attachments/assets/75bec92a-f98c-488a-b001-12bc2c67c2dc" />

<Fig 11. gpr의 분류기 이후 증강 데이터 성능 비교 (max peak)>

분류기 이후 warpmax의 양 뿔 형상이 제거된 것을 확인 할수 있다. 
XGBoost의 경우 뿔 형상의 비중이 높아 분류기로 완전히 제거되지 않았다.

### Step 3: 파레토 프론티어 (Pareto Frontier) 타겟 곡선 추출

<img width="1071" height="711" alt="image" src="https://github.com/user-attachments/assets/32b5ef4c-f8f1-4652-9824-258697fccffa" />

<Fig 12. 파레토 프론티어 warpmax(휘어짐) vs T_tip_peel(박리)>

위에서 선택한 R2 score가 높은 핵심 7개 변수만 추출하여 유토피아 텐서를 생성한다.
유토피아 텐서란 **"현실에는 존재할 수 없는 완벽한 이상향의 물리적 상태"**를 수학적으로 정의한 텐서이고
Step 4 역설계 모델(1D-CNN)의 입력 텐서로써 사용이 된다. 
그러나 이 과정에서 물리적으로 모순된 치수 조합이 나올수 있기 때문에 step 5에서 미세튜닝 작업을 거쳐야합니다.

### Step 4: 딥러닝 기반 역설계 (Inverse Design) 초안 출력
Step 3에서 생성된 유토피아 타겟 텐서(7채널 × 617 timestep)를 입력하면,
이를 구현할 수 있는 **최적의 P1~P6 설계변수 초안**을 출력하는 과정입니다. 

원본 시계열에 **사비츠키-골레이 필터**로 메쉬 노이즈를 제거한 뒤, **ResNet 잔차 연결 + U-Net Skip Connection** 구조의 오토인코더가 7채널×600 timestep을 32차원 잠재 벡터로 압축한다.
학습 시 **Smooth L1 + TV Loss**로 계단 경계의 깁스 현상과 평탄 구간 잔물결을 동시에 억제하며, **지도형 오토인코더(Supervised AE)**가 잠재 벡터에 P1-P6 정보를 강제 주입하여 U-Net의 정보 우회 현상을 방지한다.
최종적으로 유토피아 타겟 텐서를 학습된 Encoder로 압축한 뒤, MLP 역매핑이 잠재 벡터로부터 P1-P6 설계변수 초안을 도출하여 Step 5 NSGA-II의 시작점으로 전달한다.
pymoo는 NSGA-II의 진화 루프 + 제약조건 처리 + 비지배 정렬을 패키지화한 라이브러리입니다.

위 기법들은 원본 데이터의 노이즈를 제거하고 오토인코더 복원의 깁스 현상 (없던 노이즈를 생성), 언더피팅등을 방지 하기위해 적용된 것들이다.

<img width="2149" height="471" alt="image" src="https://github.com/user-attachments/assets/314594d4-3785-4698-ba67-6f9be368964b" />

<Fig 13. 오토인코더 학습 곡선, 복원 품질, 잠재 공간 시각화>

오토인코더의 학습 곡선을 보면 train loss는 0.02 가까이 수렴하는 것을 확인할수 있다. val loss는 복원 오차 + 평탄화 + P예측 오차(Sup_Loss)를 모두 포함하기 때문에     
잠재 변수 z(오토인코더로 압축된 단순 벡터)에서 P1~P6 6개의 물리 변수를 정확히 역추적하는 것 때문에 수치가 크지만 복원 오차 자체는 0.02에 근접하다.     
아래 Tabular ResNet + Bayesian Optimization의 경우 복원 오차만 그래프에 반영해 train loss 보다 더 작은 수치로 수렴하는 것을 볼 수 있다.

<img width="2151" height="471" alt="image" src="https://github.com/user-attachments/assets/9f9c675c-5095-4d7a-bd74-8706e07030df" />

<Fig 14. Tabular + Bayesian 오토인코더 학습 곡선, 복원 품질, 잠재 공간 시각화>

복원 품질은 매우 우수한 결과가 나온 것을 볼수 있다. 

세번째 그래프는 오토인코더의 잠재공간 시각화이다.    

<img width="512" height="339" alt="복원 3" src="https://github.com/user-attachments/assets/12c30326-5918-4907-a117-798de7bf2aad" />
<img width="512" height="339" alt="복원 2" src="https://github.com/user-attachments/assets/8723f966-fc7d-4a3d-b146-da41512d2bd6" />
<img width="512" height="339" alt="복원 1" src="https://github.com/user-attachments/assets/b52c4437-9b7a-40d3-94d5-1dfcc585ec85" />

<Fig 15. 오토인코더 복원 품질 그래프에서 원본 그래프의 3가지 타입>

열팽창계수의 차이에 의해 두께 패러미터 조합별로 위로 휘거나 별로 휘지 않거나 아래로 휘는 3가지 패턴이 발생하고 스텝이 넘어갈때 CAE프로그램 해석에서 극단적인 조건 변화에 의한 노이즈가 발생하는것을 확인할수 있다.

<img width="1673" height="948" alt="image" src="https://github.com/user-attachments/assets/099c9d4d-6b96-47c7-ac72-c0647ee6d445" />

<Fig 16. 역매핑 성능 평가>

역매핑(결과를 입력하면 원인을 도출해 내는 수학적 역추적 과정) 결과 우수한 선형 형태를 확인할수 있다. P6의 경우 스케일의 차이이고 수치를 보면 소수점 단위로 매우 밀착했음을 알수 있다.
$y=x$ 그래프(대각선 점선)에 수렴한다는 것은 주문한 목표 성능(Target)"과 "AI가 역산출한 설계도로 확인한 실제 성능(Achieved)"이 오차 없이 일치함을 의미합니다.

### Step 5: 머신러닝 미세 튜닝 (Fine-tuning via GA & Penalty Limits)

Step 4에서 도출된 P1~P6 초안을 바탕으로 **NSGA-II 유전 알고리즘**을 실행하여
최종 최적 설계변수를 도출한다. 물리적 한계치(Limit)를 초과하는 설계는
페널티로 즉시 도태시켜 안전한 최적해만 생존시킨다.

1. 패러미터 별로 바운더리를 초과하지 않도록 클리핑 적용
2. GPR 대리 모델로 나머지 응력을 예측하여 재료 한계치 초과 여부를 판정한다.
한계치를 넘으면 Loss에 대형 페널티를 부여하여 해당 개체를 즉시 도태시킨다.
3. WarpMax(휘어짐)과 T_Tip_Peel(박리)의 가중합을 최소화하는것이 목표

<img width="2151" height="831" alt="image" src="https://github.com/user-attachments/assets/8235d7ea-a091-4cf4-872a-b6b56e016b60" />

<Fig 17. top 5 최적 설계안>

 Knee Point (추천, 최적 밸런스), WarpMax 최소 우선, T_Tip_Peel 최소 우선, 파레토 중간 트레이드오프, σ 총합 최소 (최고 신뢰도) 등 우선도에 따라 다양한 최적 설계 조건이 도출.

 이중 knee point(두 가지 목적 함수(예: X축=WarpMax, Y축=T_Tip_Peel)를 그래프로 그렸을 때, 곡선이 마치 사람의 '구부린 무릎'처럼 급격하게 꺾이는 지점) 수치를 최종 선택하였다.

 
### 12.2 최적 설계 전후 비교 

<img width="719" height="236" alt="결과 교차 검증" src="https://github.com/user-attachments/assets/48fe6de2-326e-4f30-b3da-770ec55c66e8" />

<img width="719" height="236" alt="JMS결과 교차 검증" src="https://github.com/user-attachments/assets/aabb03d4-462e-4ca0-b6cc-118256385734" />

<fig 18. Case B: GPR + ARD 커널 + LHS와 Case C: Tabular ResNet + Bayesian Optimization의 개선 전후 개선율>

비교용으로 패러미터 범위 이내의 임의의 수치로 만든 비교군과 case B, C의 휘어짐(warpmax), 박리(T_Tip_peel), 다이 깨짐(Die_SY_Max)를 비교하면 둘다 모두 우수하게 개선 된것을 확인 할수 있다.

임의의 수치 결과는 휘어짐, 박리가 발생하는 수치였지만 case B, C는 모두 방지되는 안전 영역대에 도달하였다.

<img width="1600" height="444" alt="개선 결과 시계열 그래프" src="https://github.com/user-attachments/assets/200b3f6b-910f-4945-9430-c92dff2f5a2b" />

<fig 19. Case B: GPR + ARD 커널 + LHS의 개선 전후 시계열 그래프 비교>

시계열 전체 그래프를 보아도 크게 개선 된 것을 확인할 수 있다.

### 12.3 대리모델 성능평가

step 1에서 학습 시킨 대리모델의 성능이 어느정도 인지 실제 cae 프로그램 결과와 일치율을 비교해보았다.
이는 대리모델 논문들에서도 자주 사용되는 성능 확인 방식이다.

<img width="600" height="1200" alt="대리 모델 성능 결과" src="https://github.com/user-attachments/assets/1a63aaf5-7c4b-4f0c-8653-17f4c026a9f4" />

<fig 20. 대리모델 성능평가>

결과 휘어짐과 박리를 보여주는 핵심 컬럼인 warpmax와 t_tip_peel은 90% 이상의 우수한 일치율을 보여주었다.
이외에도 좋은 일치율이지만 낮은 신뢰도인 컬럼도 있었고 T_avg_peel과 같이 노이즈에 의해 엉망이 결과가 나온 컬럼도 있었다.

### 12.4 ai모델 학습시 공학적 제한의 필요성 (미세 튜닝의 필요성)

<img width="707" height="255" alt="결과 교차 검증" src="https://github.com/user-attachments/assets/7abdc6a5-a379-4bc9-a298-0a9c66aedb2f" />

<Fig 21. Case A : XGboost + LHS의 압도적으로 뛰어난 개선 품질>

<img width="2561" height="1356" alt="directional deformation 300s" src="https://github.com/user-attachments/assets/6a82f511-8bd9-4321-964a-931ea8f12682" />

<Fig 22. Case A : XGboost + LHS의 물리적 비현실적인 형상>

미세 튜닝에서 바운더리에 대한 클리핑이 적용 되지 않는다면 위 결과의 기판 두께 0.01mm같은 비현실적인 형상이 도출될수 있다.    
단순 수치적으로는 최적의 결과를 도출했기 때문에 우수한 개선 결과를 보였다.
현실적인 결과를 출력하도록 구체적인 제한이 필요하다.

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

* 13.4 플립칩(Flip-Chip) 패키지 열변형 3-Step 시나리오의 오류

   각 파트들이 본래 상온에서 가공되기 때문에 상온에서 잔류응력이 없고 고온경화 조건인 현재의 step 1구간에서 열변형이 일어나도록 반영이 되어야 합니다.
따라서 현재 step 1앞에 상온 상태의 조건을 한개 더 추가하여 총 step 4개의 시나리오가 되어야 정확합니다.













