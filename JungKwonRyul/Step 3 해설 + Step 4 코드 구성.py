먼저 **Step 3에서 진행한 '유토피아 타겟 추출' 로직에 대한 심층 해설**을 드리고, 이어서 **Step 4: 역설계 AI(1D-CNN)를 실제로 학습시키기 위한 전체 코드**를 구성해 드리겠습니다.

---

## 1. Step 3 로직 해설: 왜 이렇게 했는가?

우리가 작성한 Step 3 코드는 단순히 숫자를 바꾸는 것이 아니라 **물리적 개연성**을 유지하는 데 초점이 맞춰져 있습니다.

1. **파레토 프론티어(Pareto Frontier) 선별**:
* 시뮬레이션 결과 중 `WarpMax`(전체 휘어짐)와 `T_Tip_Peel`(박리 응력)은 서로 상충(Trade-off) 관계일 때가 많습니다.
* 하나만 최소화하는 것이 아니라, 두 위험 지표가 **동시에 낮은 최전선(Frontier)**의 설계 ID를 찾아냄으로써 AI에게 '현실적으로 가장 안전한 목표'를 제시한 것입니다.


2. **가상(Virtual) ID의 실제 파일 매칭**:
* XGBoost가 만든 10만 개의 가상 데이터는 '결과값'만 있고 '300초 시계열 파형'은 없습니다.
* 따라서 가상 데이터의 결과값과 가장 유사한 성능을 냈던 **실제 시뮬레이션 파일(ID: 324 등)**을 찾아 그 파형을 베이스라인으로 삼았습니다.


3. **유토피아 스케일링 (x0.9)**:
* `README` 지침에 따라 전체 채널에 0.9를 곱했습니다. 이는 **"패키지의 거동 패턴(Phase)은 유지하되, 전체적인 응력 수준만 10% 개선된 이상적인 상태"**를 시뮬레이션 없이 생성해낸 것입니다.



---

## 2. Step 4: 1D-CNN 역설계 모델 학습 코드

이제 `Data` 폴더에 있는 모든 실제 시뮬레이션 데이터(~900개)를 읽어와서 AI를 학습시켜야 합니다. 이 모델은 **[시계열 텐서] → [P1~P6 설계 변수]**를 예측합니다.

### [Step 4-1] 전체 데이터 로딩 및 텐서 구축

```python
import os
import pandas as pd
import numpy as np
from tqdm import tqdm  # 진행률 표시

def load_all_training_data(data_dir, df_peaks, channels):
    """
    Data 폴더의 모든 CSV를 읽어 (N, 300, 채널수) 텐서와 (N, 6) 설계변수를 반환합니다.
    """
    X_list = []
    y_list = []
    
    # 실제 존재하는 파일들만 루프 (df_peaks에 기록된 Survived ID 기준)
    for _, row in tqdm(df_peaks.iterrows(), total=len(df_peaks), desc="데이터 로딩 중"):
        row_id = int(row['Row_ID'])
        file_path = os.path.join(data_dir, f'ML_DATA_Extract_Row_{row_id}.csv')
        
        if os.path.exists(file_path):
            df_raw = pd.read_csv(file_path)
            
            # 1. 입력 데이터 (X): 시계열 텐서 (300, len(channels))
            X_list.append(df_raw[channels].values)
            
            # 2. 정답 데이터 (y): 설계 변수 P1~P6
            y_list.append(row[['P1', 'P2', 'P3', 'P4', 'P5', 'P6']].values)
            
    return np.array(X_list), np.array(y_list)

# 실행 (numeric_channels는 이전 단계에서 정의된 8~9개 물리 채널)
data_dir = r"D:\AI_Study\GitHub\Battery\Data"
X_train_full, y_train_full = load_all_training_data(data_dir, df_peaks, numeric_channels)

print(f"\n✅ 학습 데이터 구축 완료: X{X_train_full.shape}, y{y_train_full.shape}")

```

### [Step 4-2] 1D-CNN 모델 정의 및 학습

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_inverse_model(input_shape):
    model = models.Sequential([
        # 시계열 특징 추출 (Temporal Feature Extraction)
        layers.Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(), # 시계열 정보를 하나의 벡터로 압축
        
        # 회귀 분석 헤드 (P1~P6 예측)
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(6) # 최종 출력: P1, P2, P3, P4, P5, P6
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 모델 생성 및 학습
input_shape = (X_train_full.shape[1], X_train_full.shape[2]) # (300, 채널수)
model_inv = build_inverse_model(input_shape)

history = model_inv.fit(
    X_train_full, y_train_full,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

```

---

## 3. 유토피아 타겟을 이용한 최종 설계 도출

학습이 끝나면, 우리가 Step 3에서 만든 **유토피아 타겟(0.9배 스케일링된 곡선)**을 모델에 넣어 최적의 설계 변수 초안을 얻습니다.

```python
# 1. 유토피아 타겟 텐서 준비 (1, 300, 채널수)
utopia_input = df_utopia_target[numeric_channels].values.reshape(1, 300, len(numeric_channels))

# 2. 최적 설계 변수 예측
predicted_design = model_inv.predict(utopia_input)

print("\n🚀 [최종 결과] 유토피아 곡선을 구현하기 위한 최적 설계 초안 (P1~P6):")
print(predicted_design)

# 결과를 데이터프레임으로 정리
df_final_draft = pd.DataFrame(predicted_design, columns=['P1', 'P2', 'P3', 'P4', 'P5', 'P6'])
display(df_final_draft)

```

---

### 이 단계의 핵심 포인트

* **입력 데이터의 일관성**: 학습 시 사용한 `numeric_channels`의 순서와 `utopia_input`에 들어가는 채널의 순서가 반드시 일치해야 합니다.
* **1D-CNN의 역할**: 이 모델은 시계열 곡선의 '기울기'와 '피크 시점' 등을 보고, "이런 응력 분포가 나오려면 각 재료의 두께(P)는 이래야 한다"라는 물리적 역관계를 학습합니다.

이제 이 코드를 실행하면, `Data` 폴더의 모든 시뮬레이션을 학습한 뒤 **"당신이 원하는 완벽한 곡선을 만들려면 P1~P6를 이렇게 조합하세요"**라는 답을 얻게 됩니다.

학습을 진행해 보시고, 결과값(P1~P6)이 물리적으로 타당한 범위 내에 있는지 같이 확인해 볼까요?