import os
import re
import glob
import platform
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

# ====================================================================
# [1. 환경 설정] Windows와 WSL(Linux) 환경 자동 감지 및 경로 할당
# ====================================================================
if platform.system() == 'Linux':
    CSV_FOLDER = '/mnt/i/ai_model_dev/cfd/SIM_CSV_DATA'
    MASTER_CSV = '/mnt/i/ai_model_dev/cfd/Master_DOE_1200.csv'
    BASE_DIR   = '/mnt/i/ai_model_dev/cfd'
else:
    CSV_FOLDER = r'I:\ai_model_dev\cfd\SIM_CSV_DATA'
    MASTER_CSV = r'I:\ai_model_dev\cfd\Master_DOE_1200.csv'
    BASE_DIR   = r'I:\ai_model_dev\cfd'

# Step 1 증강 데이터 입력 / Step 2 필터링 결과 출력 (절대 경로 통일)
AUGMENTED_INPUT  = os.path.join(BASE_DIR, 'Augmented_100k_Data.csv')
FILTERED_OUTPUT  = os.path.join(BASE_DIR, 'Augmented_Class_Data.csv')

SEED = 42
np.random.seed(SEED)

print('=== [Step 2] Gatekeeper 분류기 가동 준비 완료 ===')
print(f'현재 감지된 OS  : {platform.system()}')
print(f'시계열 CSV 폴더 : {CSV_FOLDER}')
print(f'마스터 DOE 파일 : {MASTER_CSV}')
print(f'증강 데이터 입력: {AUGMENTED_INPUT}')
print(f'필터링 결과 출력: {FILTERED_OUTPUT}\n')

# ====================================================================
# [2. 실제 해석 결과 기반 생존/파탄 라벨링]
# ====================================================================
# glob으로 해당 폴더의 모든 CSV 파일 탐색
pattern = os.path.join(CSV_FOLDER, 'ML_DATA_Extract_Row_*.csv')
found_files = sorted(glob.glob(pattern))

# 파일명에서 Row_ID 추출
survived_ids = set()
for fpath in found_files:
    fname = os.path.basename(fpath)
    match = re.search(r'Row_(\d+)\.csv', fname)
    if match:
        survived_ids.add(int(match.group(1)))

if not survived_ids:
    raise FileNotFoundError("지정된 경로에서 CSV 파일을 하나도 찾지 못했습니다.")

# 마스터 DOE 데이터 로드
try:
    df_master = pd.read_csv(MASTER_CSV)
except FileNotFoundError:
    raise FileNotFoundError(f"마스터 파일을 찾을 수 없습니다: {MASTER_CSV}")

# 전체 DP 수 = 폴더 내 최대 Row_ID (단, 마스터 DOE 행 수 초과 방지)
max_row_id = min(max(survived_ids), len(df_master))
print(f"라벨링 범위: Row_ID 1 ~ {max_row_id} (마스터 DOE {len(df_master)}행 중)")

# 1번부터 max_row_id까지 라벨링
# Row_ID=1 → df_master.iloc[0] (CSV 1행은 헤더, 2행부터 데이터)
training_data = []
for row_id in range(1, max_row_id + 1):
    idx = row_id - 1  # 0-based 인덱스

    if idx >= len(df_master):
        print(f'[경고] Row_ID={row_id}가 마스터 DOE 범위를 초과 → 스킵')
        continue

    # 생존 여부: 폴더에 해당 CSV가 있으면 1(Safe), 없으면 0(Fail)
    is_safe = 1 if row_id in survived_ids else 0

    row_data = df_master.iloc[idx].to_dict()
    row_data['Row_ID'] = row_id
    row_data['is_safe'] = is_safe
    training_data.append(row_data)

df_train = pd.DataFrame(training_data)

n_total = len(df_train)
n_safe  = df_train['is_safe'].sum()
n_fail  = n_total - n_safe

print(f"학습 데이터: 총 {n_total}개 | 생존(Safe) {n_safe}개 ({n_safe/n_total*100:.1f}%) | 파탄(Fail) {n_fail}개 ({n_fail/n_total*100:.1f}%)\n")

# ====================================================================
# [3. Random Forest Gatekeeper 모델 학습 + 성능 평가]
# ====================================================================
X_train = df_train[['P1', 'P2', 'P3', 'P4', 'P5', 'P6']]
y_train = df_train['is_safe']

# -- 3-1. 5-Fold Stratified CV로 성능 사전 평가 --
# Stratified: 클래스 비율(Safe/Fail)을 각 Fold에서 동일하게 유지
gatekeeper_cv = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    class_weight='balanced',  # 클래스 불균형 해소
    random_state=SEED
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_f1 = cross_val_score(gatekeeper_cv, X_train, y_train, cv=skf, scoring='f1')
cv_acc = cross_val_score(gatekeeper_cv, X_train, y_train, cv=skf, scoring='accuracy')

print('-- 5-Fold Stratified CV 성능 --')
print(f'  F1 Score : {cv_f1.mean():.4f} (+-{cv_f1.std():.4f})')
print(f'  Accuracy : {cv_acc.mean():.4f} (+-{cv_acc.std():.4f})')

# -- 3-2. 전체 데이터로 최종 학습 (OOB 평가 포함) --
gatekeeper = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    class_weight='balanced',
    oob_score=True,   # Out-of-Bag 스코어로 보조 검증
    random_state=SEED
)

print("\nRandom Forest Gatekeeper 학습 중...")
gatekeeper.fit(X_train, y_train)
print(f"학습 완료. OOB Accuracy: {gatekeeper.oob_score_:.4f}")

# -- 3-3. Feature Importance (어떤 P가 파탄에 가장 큰 영향?) --
print('\n-- Feature Importance (파탄 예측 기여도) --')
importances = gatekeeper.feature_importances_
for col, imp in sorted(zip(['P1','P2','P3','P4','P5','P6'], importances),
                        key=lambda x: x[1], reverse=True):
    bar = '#' * int(imp * 50)
    print(f'  {col}: {imp:.4f} {bar}')

# ====================================================================
# [4. 증강 데이터(10만 개) 필터링]
# ====================================================================
try:
    df_aug = pd.read_csv(AUGMENTED_INPUT)
except FileNotFoundError:
    raise FileNotFoundError(f"Step 1에서 생성된 '{AUGMENTED_INPUT}' 파일을 찾을 수 없습니다.")

print(f"\n{len(df_aug):,}개의 가상 증강 데이터 필터링을 시작합니다.")

# P1~P6 추출 후 Gatekeeper로 0/1 이진 판정
X_aug = df_aug[['P1', 'P2', 'P3', 'P4', 'P5', 'P6']]
aug_preds = gatekeeper.predict(X_aug)

fail_count = (aug_preds == 0).sum()
safe_count = (aug_preds == 1).sum()
total_count = len(aug_preds)

print("-" * 50)
print(f"  Safe(1) : {safe_count:,}개 ({safe_count/total_count*100:.2f}%)")
print(f"  Fail(0) : {fail_count:,}개 ({fail_count/total_count*100:.2f}%) → 삭제 대상")
print("-" * 50)

# ====================================================================
# [5. Fail 행 제거 및 저장]
# ====================================================================
# Fail(0)인 행 제거, Safe(1)만 남김
df_aug_filtered = df_aug[aug_preds == 1].reset_index(drop=True)

df_aug_filtered.to_csv(FILTERED_OUTPUT, index=False)

print(f"\n결측치 제거 완료!")
print(f"  입력: {total_count:,}개 → 출력: {len(df_aug_filtered):,}개")
print(f"  저장: {FILTERED_OUTPUT}")
print("이 데이터는 [Step 3: 파레토 타겟 추출]의 베이스라인으로 사용됩니다.")
