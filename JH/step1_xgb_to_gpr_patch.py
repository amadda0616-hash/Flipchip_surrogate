"""
==========================================================================
 [Step 1 GPR 패치] XGBoost → Gaussian Process Regression 변경 사항
==========================================================================

변경 대상 셀과 위치 안내:
  - 셀 [2]  : import 수정 (xgboost 제거, GPR 추가)
  - 셀 [12] : 마크다운 대체 (XGBoost 설명 → GPR 설명)
  - 셀 [13] : 코드 대체 (Val split 제거, StandardScaler 추가)
  - 셀 [14] : 코드 대체 (XGBoost 학습 → GPR 학습)
  - 셀 [15] : 코드 수정 (시각화에서 XGBoost 참조 → GPR 참조)
  - 셀 [16] : 코드 대체 (Feature Importance → ARD 커널 Length Scale 기반 중요도)
  - 셀 [19] : 코드 대체 (predict → GPR predict + 불확실성 출력)

  후속 영향:
  - 셀 [23] 분포 비교 그래프: 코드 변경 불필요 (df_augmented 변수명 동일)
  - Step 2 gatekeeper: 변경 불필요 (Augmented_100k_Data.csv 포맷 동일)
==========================================================================
"""

# =====================================================================
# 셀 [2] 대체: import 및 환경 설정
# 위치: 기존 셀 마지막 줄 "print(f'추출 대상 Y 변수: {len(Y_COLUMNS)}개')"
# =====================================================================

CELL_2_REPLACEMENT = """
import os
import re
import glob
import time
import warnings
import platform
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 9

# ====================================================================
# [경로 자동 설정] Windows와 WSL(Linux) 환경을 자동 감지하여 경로 할당
# ====================================================================
if platform.system() == 'Linux':
    CSV_FOLDER = '/mnt/i/ai_model_dev/cfd/SIM_CSV_DATA'
    MASTER_CSV = '/mnt/i/ai_model_dev/cfd/Master_DOE_1200.csv'
    BASE_DIR   = '/mnt/i/ai_model_dev/cfd'
else:
    CSV_FOLDER = r'I:\\ai_model_dev\\cfd\\SIM_CSV_DATA'
    MASTER_CSV = r'I:\\ai_model_dev\\cfd\\Master_DOE_1200.csv'
    BASE_DIR   = r'I:\\ai_model_dev\\cfd'

Y_COLUMNS = [
    'WarpMax',          # 패키지 전체 최대 열변형량 (최소화 메인 타겟 #1)
    'T_Tip_Peel',       # Top 계면 끝단 수직응력 - 박리 원인 (최소화 메인 타겟 #2)
    'T_Tip_Shear',      # Top 계면 끝단 전단응력 - 계면 피로 유발
    'T_Tip_SEQV',       # Top 끝단 Von Mises 등가응력 - 소성 변형 유발
    'T_Tip_Strain',     # Top 끝단 변형률
    'T_Avg_Peel',       # Top 접합면 평균 수직응력 - 중앙부 Void 유발
    'T_Avg_Shear',      # Top 접합면 평균 전단응력
    'B_Tip_Peel',       # Bottom 끝단 수직응력
    'B_Tip_Shear',      # Bottom 끝단 전단응력
    'B_Tip_SEQV',       # Bottom 끝단 Von Mises 등가응력
    'B_Tip_Strain',     # Bottom 끝단 변형률
    'B_Avg_Peel',       # Bottom 평균 수직응력
    'B_Avg_Shear',      # Bottom 평균 전단응력
    'Die_SX',           # 다이(실리콘 칩) 휨 응력 - Die Crack 유발
    'Die_SY_Max'        # 다이 최대 Y방향 응력 - 모서리 응력 집중
]

# 물리적으로 항상 양수(>=0)인 변수 목록 (외삽 시 음수 clipping 대상)
# Von Mises 등가응력과 변형률은 정의상 음수가 될 수 없음
POSITIVE_ONLY_COLS = ['T_Tip_SEQV', 'T_Tip_Strain', 'B_Tip_SEQV', 'B_Tip_Strain']

SEED = 42
np.random.seed(SEED)

print('=== 환경 설정 완료 (GPR 모드) ===')
print(f'현재 감지된 OS: {platform.system()}')
print(f'시계열 CSV 폴더 : {CSV_FOLDER}')
print(f'마스터 DOE 파일 : {MASTER_CSV}')
print(f'추출 대상 Y 변수: {len(Y_COLUMNS)}개')
print(f'양수 전용 변수  : {POSITIVE_ONLY_COLS}')
"""


# =====================================================================
# 셀 [12] 대체: 마크다운 (XGBoost 설명 → GPR 설명)
# 위치: 기존 셀 마지막 줄 "- `reg_alpha=0.1, reg_lambda=1.0`: L1/L2 정규화로 복잡도 억제"
# =====================================================================

CELL_12_REPLACEMENT = """
---
## 4. Gaussian Process Regression (GPR) 대리 모델 학습

### XGBoost에서 GPR로 변경한 이유
- XGBoost/LightGBM은 트리 기반 모델로, 학습 데이터 범위 경계에서 **리프 노드 평균값으로 수렴**하여
  분포 양 끝에 비정상적인 뿔(Spike)이 발생함 (구조적 한계, Optuna 튜닝으로도 해결 불가)
- GPR은 **연속 함수**로 예측하므로 경계에서 flat 수렴 없이 부드러운 분포 생성
- 예측값과 함께 **불확실성(σ)**을 출력하여 외삽 영역을 자동 감지 가능
- 데이터가 ~900개인 상황은 GPR의 최적 구간 (수천 개 이상이면 느려짐)

### 커널 선정
- `Matern(nu=2.5)`: 물리 시뮬레이션에 적합 (2차 미분 가능, 매끄러운 응력 곡면)
- `ConstantKernel`: 출력 스케일 자동 조정
- `WhiteKernel`: 관측 노이즈(시뮬레이션 메쉬 오차 등) 흡수

### 학습 전략
- 타겟별 개별 GPR 학습 (15개 독립 모델)
- **StandardScaler** 적용: GPR은 입력 스케일에 민감하므로 P1~P6를 정규화
- 5-Fold CV + 홀드아웃 Test로 성능 이중 검증
"""


# =====================================================================
# 셀 [13] 대체: X/Y 분리 및 Split
# 위치: 기존 셀 마지막 줄 "print(f'  +-- Test  : {len(X_test)}개 (최종 성능 평가, 학습에 미사용)')"
# → GPR은 Early Stopping이 없으므로 Val split 불필요, 대신 StandardScaler 추가
# =====================================================================

CELL_13_REPLACEMENT = """
# == 4-1. X / Y 분리 및 Train / Test Split ==

X = df_peaks[['P1','P2','P3','P4','P5','P6']].copy()  # 입력: 6개 두께 변수
Y = df_peaks[Y_COLUMNS].copy()                         # 출력: 15개 응력/변형 피크

# 홀드아웃 테스트셋 15% 분리
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.15, random_state=SEED
)

# GPR은 입력 스케일에 민감 → StandardScaler로 P1~P6 정규화
# P1(0.80~1.10)과 P6(0.04~0.08)의 스케일 차이가 크므로 반드시 필요
scaler_X = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_train)   # 학습 데이터 기준 fit + transform
X_test_sc  = scaler_X.transform(X_test)         # 테스트 데이터는 transform만

print(f'전체 데이터: {len(X)}개')
print(f'  +-- Train : {len(X_train)}개 (GPR 학습 + 5-Fold CV)')
print(f'  +-- Test  : {len(X_test)}개 (최종 성능 평가, 학습에 미사용)')
print(f'\\nStandardScaler 적용 완료 (GPR 입력 정규화)')
"""


# =====================================================================
# 셀 [14] 대체: XGBoost 학습 → GPR 학습
# 위치: 기존 셀 마지막 줄 "print(f'평균 CV R2: {avg_cv:.4f} / 평균 Test R2: {avg_test:.4f}')"
# =====================================================================

CELL_14_REPLACEMENT = """
# == 4-2. 타겟별 개별 GPR 학습 ==
# 15개 Y 변수 각각에 대해 독립적인 Gaussian Process 모델을 학습
# GPR은 연속 함수로 예측하므로 트리 모델의 양 끝단 뿔(Spike) 현상이 발생하지 않음

# 커널 구성: ConstantKernel * Matern(nu=2.5) + WhiteKernel
# - Matern(nu=2.5): 물리 시뮬레이션에 적합 (2차 미분 가능한 매끄러운 함수)
# - ConstantKernel: 출력 진폭(amplitude) 자동 조정
# - WhiteKernel: 관측 노이즈 흡수 (시뮬레이션 메쉬 오차 등)
kernel_base = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))

# 학습 결과 저장용 딕셔너리
models = {}          # {y_col: fitted GaussianProcessRegressor}
cv_scores = {}       # {y_col: mean 5-Fold CV R2}
test_scores = {}     # {y_col: holdout test R2}
test_maes = {}       # {y_col: holdout test MAE}

print('=== 타겟별 GPR 학습 시작 ===')
print(f'커널: ConstantKernel * Matern(nu=2.5) + WhiteKernel')
print(f'학습 데이터: {len(X_train)}개 (스케일링 적용)')
print()

t_start = time.time()

for y_col in Y_COLUMNS:
    # -- 개별 GPR 모델 생성 및 학습 --
    gpr = GaussianProcessRegressor(
        kernel=kernel_base,
        n_restarts_optimizer=10,  # 커널 하이퍼파라미터 최적화를 10회 반복 (로컬 최적 회피)
        alpha=1e-6,               # 수치 안정성을 위한 대각 정규화
        random_state=SEED
    )
    
    gpr.fit(X_train_sc, Y_train[y_col])
    
    # -- 5-Fold 교차검증 (과적합 진단용) --
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_r2 = cross_val_score(
        GaussianProcessRegressor(
            kernel=kernel_base,
            n_restarts_optimizer=5,   # CV에서는 속도를 위해 5회로 축소
            alpha=1e-6,
            random_state=SEED
        ),
        X_train_sc, Y_train[y_col],
        cv=kf, scoring='r2'
    )
    
    # -- 홀드아웃 테스트 평가 --
    y_pred_test, y_std_test = gpr.predict(X_test_sc, return_std=True)
    
    # 물리적 음수 방지: SEQV, Strain은 정의상 항상 >= 0
    if y_col in POSITIVE_ONLY_COLS:
        y_pred_test = np.clip(y_pred_test, 0, None)
    
    r2_test = r2_score(Y_test[y_col], y_pred_test)
    mae_test = mean_absolute_error(Y_test[y_col], y_pred_test)
    
    # 결과 저장
    models[y_col] = gpr
    cv_scores[y_col] = cv_r2.mean()
    test_scores[y_col] = r2_test
    test_maes[y_col] = mae_test
    
    # 과적합 경고
    gap = abs(cv_r2.mean() - r2_test)
    flag = '  << OVERFIT?' if gap > 0.10 else ''
    
    print(f'{y_col:15s} | CV R2={cv_r2.mean():.4f} (+-{cv_r2.std():.4f}) | '
          f'Test R2={r2_test:.4f} | MAE={mae_test:.4f} | '
          f'mean_std={y_std_test.mean():.4f}{flag}')

elapsed = time.time() - t_start
print(f'\\n=== 전체 학습 완료 ({elapsed:.1f}초 소요) ===')

avg_cv = np.mean(list(cv_scores.values()))
avg_test = np.mean(list(test_scores.values()))
print(f'평균 CV R2: {avg_cv:.4f} / 평균 Test R2: {avg_test:.4f}')
"""


# =====================================================================
# 셀 [15] 대체: 모델 성능 시각화 (XGBoost→GPR 참조 변경 + 불확실성 시각화 추가)
# 위치: 기존 셀 마지막 줄 "plt.show()"
# =====================================================================

CELL_15_REPLACEMENT = """
# == 4-3. 모델 성능 시각화 ==

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# --- (A) 변수별 R2 비교 바 차트 (CV vs Test) ---
ax = axes[0]
x_pos = np.arange(len(Y_COLUMNS))
width = 0.35

ax.bar(x_pos - width/2, [cv_scores[c] for c in Y_COLUMNS], width,
       label='5-Fold CV R2', color='steelblue', alpha=0.8)
ax.bar(x_pos + width/2, [test_scores[c] for c in Y_COLUMNS], width,
       label='Holdout Test R2', color='coral', alpha=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(Y_COLUMNS, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('R2 Score')
ax.set_title('GPR Surrogate Performance by Target', fontweight='bold')
ax.legend(fontsize=8)
ax.axhline(0.9, color='green', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_ylim(0, 1.05)

# --- (B) Pred vs Actual 산점도 (메인 타겟 2개) ---
ax = axes[1]
for y_col, color, marker in [('WarpMax', 'steelblue', 'o'), ('T_Tip_Peel', 'coral', 's')]:
    y_actual = Y_test[y_col].values
    y_pred, y_std = models[y_col].predict(X_test_sc, return_std=True)
    if y_col in POSITIVE_ONLY_COLS:
        y_pred = np.clip(y_pred, 0, None)
    ax.scatter(y_actual, y_pred, alpha=0.4, s=15, c=color, marker=marker, label=y_col)

all_vals = np.concatenate([Y_test['WarpMax'].values, Y_test['T_Tip_Peel'].values])
lims = [all_vals.min() * 1.1, all_vals.max() * 1.1]
ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5, label='Perfect (y=x)')
ax.set_xlabel('Actual (FEA Simulation)')
ax.set_ylabel('Predicted (GPR Surrogate)')
ax.set_title('Pred vs Actual (Main Targets)', fontweight='bold')
ax.legend(fontsize=8)

# --- (C) GPR 불확실성(σ) 분포 (GPR 고유 장점) ---
# 불확실성이 높은 예측 = 외삽 영역 → 신뢰도 낮음
ax = axes[2]
for y_col, color in [('WarpMax', 'steelblue'), ('T_Tip_Peel', 'coral')]:
    _, y_std = models[y_col].predict(X_test_sc, return_std=True)
    ax.hist(y_std, bins=30, alpha=0.5, color=color, edgecolor='white', label=y_col)
ax.set_xlabel('Prediction Uncertainty (σ)')
ax.set_ylabel('Count')
ax.set_title('GPR Uncertainty Distribution (Test Set)', fontweight='bold')
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
"""


# =====================================================================
# 셀 [16] 대체: Feature Importance → GPR Length Scale 기반 중요도
# 위치: 기존 셀 마지막 줄 "plt.show()"
# =====================================================================

CELL_16_REPLACEMENT = """
# == 4-4. GPR 기반 변수 중요도 (Matern 커널 Length Scale) ==
# GPR의 Matern 커널이 학습한 length_scale 값의 역수 = 변수 민감도
# length_scale이 작을수록 → 해당 변수의 작은 변화에도 출력이 크게 반응 → 중요도 높음

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, y_col in zip(axes, ['WarpMax', 'T_Tip_Peel']):
    # 학습된 커널에서 Matern 커널의 length_scale 추출
    learned_kernel = models[y_col].kernel_
    
    # 커널 구조: ConstantKernel * Matern + WhiteKernel
    # Matern은 곱 커널(k1)의 두 번째 요소(k2)에 위치
    matern_kernel = learned_kernel.k1.k2  # k1 = Constant*Matern, k2 = Matern
    length_scales = matern_kernel.length_scale
    
    # length_scale이 스칼라(단일값)인 경우 = isotropic → 모든 변수 동일 중요도
    if np.isscalar(length_scales) or (hasattr(length_scales, '__len__') and len(length_scales) == 1):
        print(f'{y_col}: Isotropic 커널 (모든 변수 동일 length_scale={length_scales:.4f})')
        importance = np.ones(6) / 6  # 균등
    else:
        # 역수를 취하여 중요도로 변환 (length_scale 작을수록 중요)
        importance = 1.0 / np.array(length_scales)
        importance = importance / importance.sum()  # 정규화
    
    ax.barh(['P1','P2','P3','P4','P5','P6'], importance, color='steelblue')
    ax.set_xlabel('Relative Importance (1 / length_scale)')
    ax.set_title(f'{y_col} - GPR Variable Sensitivity', fontweight='bold')

plt.tight_layout()
plt.show()

# 참고: isotropic 커널이 나오면 변수별 차이를 구분하지 못하는 것이므로
# 이 경우 ARD 커널(변수별 독립 length_scale)로 업그레이드를 고려할 수 있음
"""


# =====================================================================
# 셀 [19] 대체: 10만 개 예측 (XGBoost→GPR + 음수 방지 + 불확실성 저장)
# 위치: 기존 셀 마지막 줄 "print(f'예측 완료! ({elapsed:.2f}초 소요)')"
# =====================================================================

CELL_19_REPLACEMENT = """
# == 5-2. GPR 대리 모델로 10만 개 Y값 예측 ==

# 가상 P1~P6도 동일한 StandardScaler로 변환
X_virtual_sc = scaler_X.transform(df_virtual_X)

print(f'학습된 GPR로 {N_VIRTUAL:,}개의 Y 값 예측 중...')
print('(GPR 예측은 XGBoost보다 느릴 수 있음, 타겟당 수 초~수십 초 소요)')
t_start = time.time()

virtual_Y_dict = {}
virtual_std_dict = {}  # 불확실성도 함께 저장

for y_col in Y_COLUMNS:
    t_col = time.time()
    
    # GPR 예측: 평균값 + 불확실성(σ)
    y_pred, y_std = models[y_col].predict(X_virtual_sc, return_std=True)
    
    # 물리적 음수 방지: SEQV, Strain은 정의상 항상 >= 0
    if y_col in POSITIVE_ONLY_COLS:
        y_pred = np.clip(y_pred, 0, None)
    
    virtual_Y_dict[y_col] = y_pred
    virtual_std_dict[y_col + '_std'] = y_std
    
    print(f'  {y_col:15s} 완료 ({time.time()-t_col:.1f}초) | '
          f'mean_std={y_std.mean():.4f}, max_std={y_std.max():.4f}')

df_virtual_Y = pd.DataFrame(virtual_Y_dict)
df_virtual_std = pd.DataFrame(virtual_std_dict)  # 불확실성 별도 저장 (선택적 활용)

elapsed = time.time() - t_start
print(f'\\n예측 완료! (총 {elapsed:.1f}초 소요)')
print(f'평균 불확실성(σ)이 높은 상위 3개 변수:')
mean_stds = {col: df_virtual_std[col+'_std'].mean() for col in Y_COLUMNS}
for col, std in sorted(mean_stds.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f'  {col}: mean σ = {std:.4f}')
"""


# =====================================================================
# 참고: 아래 셀들은 코드 변경 불필요 (변수명 호환)
# =====================================================================
# 셀 [18] LHS 생성: 변경 없음 (df_virtual_X 그대로 사용)
# 셀 [20] 물리적 범위 검증: 변경 없음 (df_virtual_Y 변수명 동일)
# 셀 [21] CSV 저장: 변경 없음 (df_augmented 구조 동일)
# 셀 [23] 분포 비교 그래프: 변경 없음
# 셀 [24] 2D 산점도: 변경 없음
# 셀 [27~32] Step 2 Gatekeeper: 변경 없음 (Augmented_100k_Data.csv 포맷 동일)
