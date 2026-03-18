# Flipchip_surrogate
# 🚀 AI 기반 반도체 패키지 신뢰성 예측 프로젝트
## 반도체가 열변형에 의해 휘어지고 깨지고 뜯어지는 것을 ai로 예측합니다.
<br>

## 1. 키워드 설명

### 1) CAE (Computer-Aided Engineering)란 무엇인지?

![01  CAE란_2](https://cdn.discordapp.com/attachments/1480386986840953003/1483416644498886676/AOI_d_-Z-yEFpxJf0VFEK67ntE3JZvsO5JRVQZM6Tc2AxlOl1HvUA35RFxid-tM-yuc7cUxECSs4_QaWi9SNhXmF5HF0P5JJaF1H1q5R8jh_Xf-2cmzmh7jV1U-VzYqz0MXR33CtG3pfq55FkrOMRnhLr5V2XLUtDmJSQLs9mR6mNApkeQlKs1600-rj.png?ex=69ba82e8&is=69b93168&hm=b824ab711a42f3ef2c7567923acb24954bbcff67fd4142c6f6bd732b7dc567dc&)

### 2) flipchip이 무엇인지?
 
![02  플립 칩이란](https://cdn.discordapp.com/attachments/1480386986840953003/1483417230808186985/image.png?ex=69ba8374&is=69b931f4&hm=c9276ced1f498e1a7283c719ab310a15f11e7a609486d72819a7eecb98ac02e9&)
   플립칩(Flip-Chip)은 반도체 칩을 기판에 연결할 때 칩을 뒤집어(Flip) 표면의 수많은 돌기(Bump)를 기판에 직접 맞닿게 붙이는 연결 기술입니다.
   
   이번 프로젝트는 i9-13900k를 모델로 했고 이 cpu가 플립칩 타입입니다.
        
### 3) 왜 변형되면 안되는지

![03  왜 변형이 되면 안되는지_2](https://cdn.discordapp.com/attachments/1480386986840953003/1483419919818428416/image.png?ex=69ba85f5&is=69b93475&hm=f9dd1a0394925482192745514158c532e3296b8d641045d1cb32e76b48ace7be&)
   Warpage(뒤틀림)은 칩이 프링글스 과자처럼 굽어서, 기판의 수만 개 접점이 제대로 맞물리지 못하게 만드는 현상이고, Delamination(박리)은 층 사이가 스티커처럼 벌어져 신호가 끊기거나, 내부 열이 빠져나가지 못하게 가두는 현상입니다.
   
   고성능 CPU가 전기가 통하지 않는 '먹통'이 되거나, 내부 열을 못 견디고 타버리게 만들기 때문에, 설계 단계에서 반드시 막아야 할 치명적인 결함입니다. 

### 4) 오토 인코더

<img width="682" height="372" alt="오토인코더" src="https://github.com/user-attachments/assets/4b7c65b7-aa15-475b-ae84-4592de3dbcac" />

1. 인코더 (Encoder) : "1,000쪽짜리 책을 1장으로 요약하기" 데이터가 너무 크고 복잡하면 AI도 헷갈립니다. 그래서 복잡한 시뮬레이션 파도 그래프(1,000쪽짜리 책)를 집어넣으면, 쓸데없는 내용은 다 버리고 가장 중요한 핵심 내용만 딱 1장으로 **압축 요약** 해 주는 역할을 합니다.    
2. 잠재 공간과 잠재 변수 $z$ : "핵심 다이얼이 모여있는 조종판" 1장으로 요약된 핵심 내용들이 바로 **잠재 변수($z$)**입니다. 비유하자면, 복잡한 데이터를 마음대로 조절할 수 있는 **핵심 조종판의 다이얼들**이라고 생각하시면 됩니다. 이 다이얼들이 모여있는 곳이 잠재 공간입니다.    
3. 보간과 유토피아 : "다이얼을 돌려 상상하기 & 꿈의 숫자 입력하기" 우리는 이 조종판의 다이얼을 이리저리 돌려보면서(보간), 기존에 없던 새로운 상태를 상상해 볼 수 있습니다. 특히, 우리 프로젝트에서는 과감하게 "응력 0, 휨 0"이라는 현실엔 없는 완벽한 꿈의 숫자(유토피아)로 다이얼을 강제로 휙 돌려버렸습니다.    
4. 디코더 (Decoder) : "요약본을 보고 완전히 새로운 책 써내기" 디코더는 **반대로 작동**합니다. 우리가 방금 다이얼을 '유토피아(완벽한 0)'로 맞춰놓고 작동 버튼을 누르면, 디코더는 그 1장짜리 요약 힌트만 보고 "아하, 이런 완벽한 결과를 원해? 그럼 설계도를 이렇게 그려야 해!" 라며 완전히 새로운 1,000쪽짜리 정답 책(설계도 초안)을 써내는 역할을 합니다.     
   
## 2. 프로젝트 개요

<img width="1380" height="752" alt="04  프로젝트 개요_2" src="https://github.com/user-attachments/assets/f8d7c348-78d1-481c-a2c4-ac434cec3663" />


반도체 패키지는 성질이 서로 다른 재료들을 겹겹이 쌓아 만든 '초정밀 샌드위치'와 같습니다. 하지만 한 가지 까다로운 점이 있습니다. 바로 열을 받으면 재료마다 부풀어 오르는 속도가 제각각이라는 것입니다.

문제점: 이런 불량을 미리 확인하기 위해 예전에는 컴퓨터로 복잡한 가상 실험(시뮬레이션)을 했습니다. 하지만 이 계산을 한 번 확인하는 데 엄청난 시간과 비용이 들어갔습니다.

해결: 그래서 우리는 수만 번의 실험 데이터를 학습하여 결과를 순식간에 예측하는 AI를 도입하기로 했습니다. AI는 복잡한 계산 과정을 거치지 않고도, 설계도만 보고 "이 위치가 휘어질 것 같다"고 0.1초 만에 정답을 알려줍니다.
<br>                     

![04-1  프로젝트 개요_2](https://github.com/user-attachments/assets/f9c069b0-ecab-4e7f-96d2-541d1f5b8834)


## 3. 대리 모델의 도입 (The "Surrogate Model" Concept)

![05  대리 모델의 도입_2](https://cdn.discordapp.com/attachments/1480386986840953003/1483430010982502451/AOI_d_-SPkxBDBGGHUHhE-IvK0oHbBARAhYWQTRGYDPbLpR3r3btAoqcJh83oQapI01idqMJYN8jKGarVqS3tpIVMrwEmWw0HX75NORkcUEW5XQw0tldt_L5XuxoJLXRyHFQ-2GosB6inUEAUil_f0j2SFPEBOdZX-nnMu1Dxu9QoQ5ZnLnl8As1600-rj.png?ex=69ba8f5b&is=69b93ddb&hm=1b1f7c3b3543b9c944f76744854ca765d89ca8982398a84daa31dfd632378769&)
이번 프로젝트의 핵심은 복잡하고 오래 걸리는 컴퓨터 시뮬레이션을 대신할 '대리 모델(Surrogate Model)'을 구축하는 것입니다.

개념: 원래 설계가 잘 되었는지 확인하려면 컴퓨터가 아주 복잡한 물리 법칙을 하나하나 계산해야 합니다. 결과는 정확하지만 시간이 너무 오래 걸린다는 게 문제입니다. '대리 모델'은 이 복잡한 과정을 거치지 않고, 그동안의 데이터를 바탕으로 **"이런 설계라면 결과는 이렇게 나올 거야"**라고 결과를 즉시 예측하는 인공지능입니다.

비유: 수학 문제를 풀 때마다 정석대로 모든 풀이 과정을 적는 대신, 수만 개의 문제 데이터를 학습해 **문제를 보자마자 답을 내놓는 암산 천재**를 만드는 것과 같습니다.

장점: 기존 시뮬레이션으로는 몇 시간이 걸리던 검증을, 0.1초 이내로 단축 가능하게 합니다.



## 4. 반도체 '샌드위치' 구조 (The Subject)
우리가 분석하는 모델은 아래와 같이 6가지 핵심 요소로 구성된 Lidded Package (뚜껑이 있는 패키지, Lidded FCBGA (Flip-Chip Ball Grid Array))입니다.   

![images](https://github.com/user-attachments/assets/20b24c1f-3e37-430e-bfe3-985b92d31987)

<Fig.1 플립칩 모형 단면>     

![06  플립칩 명칭](https://cdn.discordapp.com/attachments/1480386986840953003/1483427913738223697/AOI_d_96C3Edfqp7PVvExr_RW74EtvfmI4UX_5SJop7CyxYJi-SEfL-4A3a99BgS7VlCbTBFH3D84Rt0qjDilCNtf0Uy2uDmykjY70yK3jUGp2TjxBbQVM8quFw-SOAZ8lvFdiTbBXu_fhP-IPzwcS1SfSSKRM7t8gqxjEPlNrlKs_aN0FDiJgs1600-rj.png?ex=69ba8d67&is=69b93be7&hm=a5eeb0dcabed210ef7d837e1ba0467ca44b890db7e1abfd634a975778a6f5675&)

<Fig.2 플립칩 단순화 모델 이미지>    
> 딥러닝에 유의미한 데이터를 모으기 위해서 최대한 조건과 형상을 단순화 합니다. 실제 논문의 경우 3D 형상과 복잡한 조건으로 슈퍼컴퓨터를 사용합니다.

## 5. 📊 6-Dimension Geometric Parameter 정의서

AI에게 "이 설계가 괜찮은지" 물어보려면, 먼저 설계를 숫자로 표현해야 합니다. 이 프로젝트에서는 **6개 부품의 두께**를 설계 변수로 사용합니다.

| 부품 이름 | 역할 (쉬운 설명) | 두께 범위 |
| :--- | :--- | :--- |
| **P1 (Substrate, 기판)** | 전체 구조물의 바닥판. 모든 부품이 여기 위에 올라감 | 0.80 ~ 1.20 mm |
| **P2 (Underfill, 언더필)** | 칩과 기판 사이를 채워 충격을 흡수하는 접착제 | 0.05 ~ 0.09 mm |
| **P3 (Die, 다이)** | 실제 계산을 수행하는 반도체 칩 본체. 열이 여기서 발생 | 0.60 ~ 0.85 mm |
| **P4 (Adhesive, 접착제)** | 뚜껑 다리를 기판에 고정시키는 실런트(풀) | 0.10 ~ 0.30 mm |
| **P5 (IHS roof, 뚜껑)** | 칩을 보호하고 열을 분산시키는 금속 덮개 | 1.20 ~ 1.80 mm |
| **P6 (TIM, 열전도재)** | 칩과 뚜껑 사이에 바르는 열전도 페이스트 | 0.03 ~ 0.08 mm |

## 6. 시뮬레이션 결과 데이터 22개 컬럼 (출력값)

6개의 두께 값(입력)을 정하고 시뮬레이션을 돌리면, 300초 동안 변화하는 **16개의 물리량**(출력)이 기록됩니다. 입력 6개 + 시간/온도 2개 + 출력 16개 = 총 22개 컬럼, 72만 7천여 행의 데이터가 만들어집니다.

각 출력 값이 무엇을 의미하는지, 일상적인 비유와 함께 설명합니다:

| 컬럼 | 쉬운 설명 | 왜 중요한가? |
| :--- | :--- | :--- |
| **WarpMax** | 패키지가 얼마나 휘었는지 (최대 휨량) | 가장 중요한 지표. 많이 휘면 연결이 끊어짐 |
| **T_Tip_Peel** | 칩과 접착제 사이가 벌어지려는 힘 (수직 방향) | 박리(층 분리)의 직접적 원인 |
| **T_Tip_Shear** | 칩과 접착제 사이가 엇갈리려는 힘 (수평 방향) | 반복되면 피로 파괴 발생 |
| **T_Tip_SEQV** | 한 지점에 걸리는 종합적인 힘의 크기 | 이 값이 크면 영구 변형 발생 |
| **T_Tip_Strain** | 해당 지점이 실제로 변형된 양 | 탄성+소성+시간에 따른 변형 모두 포함 |
| **T_Avg_Peel/Shear** | 접합면 전체의 평균적인 벌어짐/엇갈림 힘 | 넓은 영역의 접합 상태 판단 |
| **B_Tip/B_Avg** | 아래쪽 접합면(기판-접착제)의 동일한 측정값들 | 위쪽뿐 아니라 아래쪽도 확인 필요 |
| **Die_SX** | 칩 자체가 휘면서 받는 힘 | 이 값이 크면 칩이 깨질 수 있음 |

## 7. CAE 데이터 구현 과정

**a. 왜 2D로 했나?**
3D로 하면 한 번 계산에 몇 시간씩 걸립니다. AI 학습에는 수천 번의 계산이 필요하므로, 가정용 PC로는 몇 달이 걸릴 수 있습니다. 그래서 계산량을 줄이기 위해 2D 단면으로 해석하고, 좌우 대칭인 구조이므로 절반만 계산했습니다.

**b. 온도 시나리오**
실제 CPU의 생애 주기를 300초로 압축했습니다:
1. 120°C에서 시작 → 상온(22°C)으로 식힘 (제조 직후)
2. 칩이 작동하며 다시 약 100°C까지 가열 (사용 중)
3. 전원을 끄고 다시 상온으로 냉각 (종료 후)

**c. 무엇을 측정했나?**
각 시점에서 패키지가 얼마나 휘는지(변형), 층 사이가 얼마나 벌어지려 하는지(응력)를 측정했습니다.

**d~f. 데이터 수집**
6개 부품의 두께를 무작위로 조합하여 총 1,200가지 설계를 만들고, 각각에 대해 300초 동안의 변화를 기록했습니다. 50개씩 24묶음 = 1,200개 케이스, 케이스당 606개 시점, 총 **727,200행의 데이터**가 생성되었습니다. 

<img width="1800" height="900" alt="normal stress 300s(1)" src="https://github.com/user-attachments/assets/1da6fae9-a988-4299-b1bc-0e88e8af8034" />
<img width="1800" height="900" alt="shear stress 300s(1)" src="https://github.com/user-attachments/assets/37fea198-0b02-4556-88dc-95ff05c07477" />
<img width="1800" height="900" alt="directional deformation 300s(1)" src="https://github.com/user-attachments/assets/f10880a2-2c85-4b9e-a789-6857952e4d76" />

<Fig.3 CAE 해석 결과 예시>

## 8. 300초 열변형 시나리오 상세

시뮬레이션은 CPU의 "탄생 $\rightarrow$ 사용 $\rightarrow$ 종료"를 3단계로 압축한 것입니다:

| 단계 | 시간 | 무슨 일이 일어나는가? | 왜 이 단계가 필요한가? |
| :--- | :--- | :--- | :--- |
| **Step 1** (제조 냉각) | 0~100초 | 120°C → 22°C로 식힘. 아직 칩은 작동 안 함 | 접착제가 고온에서 굳었기 때문에, 식으면서 재료마다 줄어드는 정도가 달라 **초기 뒤틀림**이 생깁니다. 새 신발이 마르면서 살짝 뒤틀리는 것과 비슷합니다. |
| **Step 2** (사용 중) | 100~200초 | 칩에 전원 ON, 253W 발열로 ~100°C 상승 | CPU가 전력을 소모하며 뜨거워지는 상태입니다. 고온에서 재료가 서서히 늘어나는 **크립(Creep)** 현상이 발생합니다. 무거운 짐을 오래 들고 있으면 팔이 서서히 처지는 것과 같습니다. |
| **Step 3** (종료 냉각) | 200~300초 | 전원 OFF, 다시 22°C로 냉각 | 전원을 꺼도 Step 2의 크립 때문에 **원래 모양으로 돌아오지 못합니다.** 남은 변형이 '영구 변형'이며, 이것이 축적되면 제품 수명이 줄어듭니다. |

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

### 10.1. 데이터셋 구조 (X와 Y)
- **입력(X)**: P1-P6 — 6개 부품의 두께 (0.03-1.80mm 범위)
- **출력(Y)**: 300초 동안 기록된 16개의 물리량 (응력, 변형)
- **가장 중요한 출력값**:
  - `WarpMax`: 전체 휨량 → 최소화해야 할 **1순위 목표**
  - `T_Tip_Peel`: 층간 벌어짐 힘 → 최소화해야 할 **2순위 목표**
  - `Die_SX`: 칩이 받는 휨 힘 → 이 값이 한계를 넘으면 **칩이 깨짐**

### 10.2. 데이터 결측치 (Missing Data / Infeasibility)
계획한 1,200개의 설계 조합 중 약 29%는 너무 극단적인 형상이라 시뮬레이션 도중 실패했습니다. 살아남은 71%의 데이터만 사용합니다.
> 이것은 결함이 아니라 오히려 중요한 정보입니다. "이 조합은 물리적으로 불가능하다"는 사실 자체가 AI 학습에 활용됩니다.

---

## 📌 11. 6단계 최적화 파이프라인 (Action Plan)

이 프로젝트는 단순히 "고장 여부를 예측"하는 것에 그치지 않고, **"고장이 안 나는 최적의 설계를 자동으로 찾아주는"** 전체 시스템입니다. 6단계로 이루어져 있습니다.

### Step 1: 데이터 늘리기 (Data Augmentation)

**문제**: 시뮬레이션으로 모은 데이터가 약 800개 밖에 안 됩니다. AI를 잘 훈련시키려면 더 많은 데이터가 필요합니다.

**해결**: 800개의 실제 데이터로 AI(XGBoost, GPR, Tabular ResNet)를 먼저 훈련시킨 뒤, 이 AI로 **10만 개의 가상 데이터**를 만들어냅니다. 마치 800개의 시험 문제를 풀어본 학생이 비슷한 유형의 문제 답을 추측할 수 있는 것과 같습니다.

**사용한 기법 및 알고리즘**: XGBOOST, GPR + ARD 커널, Tabular ResNet + Permutation Feature Importance, 난수화 기법 (LHS, Bayesian Optimization), MinMaxScaler, 5-Fold CV

### Step 2: 불량 설계 걸러내기 (Gatekeeper)

**문제**: Step 1에서 만든 10만 개의 가상 설계 중에는 실제로 만들면 부서지는 조합이 섞여 있습니다.

**해결**: 원본 1,200개 데이터에서 "시뮬레이션이 성공한 것=1, 실패한 것=0"으로 분류 모델(Random Forest)을 훈련시켜, 물리적으로 불가능한 설계를 미리 걸러냅니다. 결과적으로 약 20%가 제거됩니다.

**사용한 기법 및 알고리즘**: Random Forest

### Step 3: '이상적인 목표'를 수학적으로 정의하기 (Pareto Frontier)

**문제**: AI에게 "이런 결과를 만들어줘"라고 주문하려면, 먼저 "이상적인 결과"가 무엇인지 수치로 정의해야 합니다.

**해결**: 기존 데이터 중 가장 우수한 상위 5~10%를 골라내고, 그 결과에서 응력 값을 10% 더 낮춘 **유토피아 타겟**을 만듭니다. "현실에서 가장 좋았던 것보다 살짝 더 좋은 목표"를 설정하는 것입니다. 이때 휨(WarpMax)과 박리(T_Tip_Peel) 두 가지만을 기준으로 최적 데이터를 선별합니다.

**사용한 기법 및 알고리즘**: 파레토 비지배 정렬

### Step 4: AI로 역설계하기 (Inverse Design)

**문제**: "이런 결과를 원한다" → "그러면 어떤 설계를 해야 하는가?"를 알아내야 합니다.

**해결**: 오토인코더(Autoencoder)라는 딥러닝 모델을 사용합니다. Step 3의 '유토피아 타겟'을 입력하면, 이를 구현할 수 있는 **P1~P6 두께의 초안**을 한 번의 계산으로 출력합니다.

**사용한 기법 및 알고리즘**: 시계열 리샘플링, 사비츠키-골레이 필터, 1D-CNN 오토인코더, Residual Block (ResNet), U-Net Skip Connection, Upsample + Conv1d, Smooth L1 Loss, Total Variation Loss, 지도형 오토인코더, 다층 퍼셉트론 (MLP) 역매핑, StandardScaler

### Step 5: 유전 알고리즘으로 미세 조정 (Fine-tuning)

**문제**: Step 4의 초안은 대략적인 방향은 맞지만, 물리적으로 불가능한 수치가 포함될 수 있습니다.

**해결**: 유전 알고리즘(NSGA-II)을 사용하여 초안을 기준으로 ±10% 범위 내에서 미세 조정합니다. 핵심은 **물리적 한계를 넘는 설계에 거대한 벌점(+999,999점)을 부여**하여 즉시 탈락시키는 것입니다.

**사용한 기법 및 알고리즘**: NSGA-II (다목적 유전 알고리즘), 강건 최적화, Feasibility Rule (pymoo G Matrix), Knee Point (최적 밸런스 점 추출)

### Step 6: 실제 시뮬레이션으로 최종 검증 (Digital Twin)

**마지막 단계**: AI가 찾아낸 최적 설계(P1~P6)를 실제 시뮬레이션 프로그램(Ansys)에 다시 넣어서, 정말로 좋은 결과가 나오는지 확인합니다. 최초 설계와 비교하여 휨과 박리가 얼마나 줄었는지를 검증합니다.


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
AI가 박리 응력처럼 자신이 예측하기 어려운 구간(높은 σ)을 스스로 인지하고 수치화한 결과입니다. WarpMax (파란색 좁은 탑): 불확실성( $\sigma$)이 0 근처에 뾰족하게 몰려 있습니다.
T_Tip_Peel (주황색 넓은 산): 불확실성( $\sigma$)이 우측으로 넓게 퍼져 있습니다.
이 주황색 넓은 산(높은 $\sigma$)의 존재를 확인했기 때문에, Step 5 유전 알고리즘에서 목적함수 + 2 $\sigma$라는 강건 최적화를 적용하게 되었습니다.

<img width="1911" height="948" alt="image" src="https://github.com/user-attachments/assets/c37d49ae-934f-497d-ab4b-6036d4d17fd9" />

<Fig 8. GPR + ARD 커널 + LHS의 증강 데이터 품질 검증>

<img width="1911" height="948" alt="image" src="https://github.com/user-attachments/assets/7b8faaa1-8cd5-4dda-93c1-e3d488072dc2" />

<Fig 9. XGboost + LHS의 증강 데이터 품질 검증>

<img width="2151" height="1012" alt="image" src="https://github.com/user-attachments/assets/69e8cc62-8f35-43f0-8378-e0653f425217" />

<Fig 10. Tabular Resnet + Bayesian Optimization의 증강 데이터 품질 검증>

x축은 예측된 물리량, Y축 데이터 빈도수
LHS의 경우 골고루 데이터를 난수화 하다보니 극단 적인 형상도 포함 되어 양 끝단 뿔 형상이 warpmax 그래프에서 확인이 가능하다. 
XGBoost경우 트리 기반 모델의 특성으로 뿔 형상이 강화된다.
Bayesian Optimization은 최적화 과정이 포함되기 대문에 최적 조건에 몰리는 형상을 확인 가능하다.

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
유토피아 텐서란 **현실에는 존재할 수 없는 완벽한 이상향의 물리적 상태**를 수학적으로 정의한 텐서이고
Step 4 역설계 모델(1D-CNN)의 입력 텐서로써 사용이 된다. 
그러나 이 과정에서 물리적으로 모순된 치수 조합이 나올수 있기 때문에 step 5에서 미세튜닝 작업을 거쳐야합니다.

### Step 4: 딥러닝 기반 역설계 (Inverse Design) 초안 출력
Step 3에서 생성된 유토피아 타겟 텐서(7채널 × 617 timestep)를 입력하면,
이를 구현할 수 있는 **최적의 P1~P6 설계변수 초안**을 출력하는 과정입니다. 

원본 시계열에 **사비츠키-골레이 필터**로 메쉬 노이즈를 제거한 뒤, **ResNet 잔차 연결 + U-Net Skip Connection** 구조의 오토인코더가 7채널×600 timestep을 32차원 잠재 벡터로 압축한다.
학습 시 **Smooth L1 + TV Loss**로 계단 경계의 깁스 현상과 평탄 구간 잔물결을 동시에 억제하며, 지도형 오토인코더(Supervised AE)가 잠재 벡터에 P1-P6 정보를 강제 주입하여 U-Net의 정보 우회 현상을 방지한다.
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

step 1에서 학습 시킨 대리모델의 성능이 어느정도 인지 실제 CAE 프로그램 결과와 일치율을 비교해보았다.
이는 대리모델 논문들에서도 자주 사용되는 성능 확인 방식이다.

<img width="980" height="1489" alt="대리 모델 성능 결과" src="https://github.com/user-attachments/assets/1a63aaf5-7c4b-4f0c-8653-17f4c026a9f4" />

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













