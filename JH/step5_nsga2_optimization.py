"""
================================================================================
# [Step 5] NSGA-II + GPR 강건 최적화 + Gatekeeper + 통합 결승전
================================================================================

## 목표
Step 4에서 도출된 32개 P1~P6 초안을 시작점으로,
GPR 대리모델(Step 1)과 Gatekeeper 분류기(Step 2)를 결합하여
NSGA-II로 7대 핵심 채널의 절댓값 Max Peak를 최소화하는
최종 P1~P6를 도출한다.

## 핵심 보강 사항
1. GPR 불확실성(σ) 활용: μ + 2σ < 한계치 (95% 신뢰구간 강건 제약)
2. pymoo Feasibility Rule: 위반량 기반 부드러운 수렴 (단순 페널티 대체)
3. 32개 로컬 파레토 → 통합 결승전 → 글로벌 Top 5 선정

## 출력
GA_Optimized_Results.csv:
  Rank | Label | P1~P6 | 7채널_pred | 7채널_σ
================================================================================
"""

# ====================================================================
# [0. 라이브러리 및 환경 설정]
# ====================================================================
import os
import platform
import numpy as np
import pandas as pd
import joblib
import warnings
import time
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 9

# -- 경로 설정 --
if platform.system() == 'Linux':
    BASE_DIR = '/mnt/i/ai_model_dev/cfd'
else:
    BASE_DIR = r'I:\ai_model_dev\cfd'

SEED = 42
np.random.seed(SEED)

print('=== [Step 5] NSGA-II + GPR 강건 최적화 ===')
print(f'Base Directory: {BASE_DIR}')

# ====================================================================
# [1. Step 1~2에서 학습된 모델 로드]
# ====================================================================
print('\n[1] 기존 모델 로드...')

# -- GPR 대리모델 체크포인트 로드 (Step 1) --
GPR_CHECKPOINT = os.path.join(BASE_DIR, 'checkpoints', 'gpr_models_ard.pkl')

if not os.path.exists(GPR_CHECKPOINT):
    raise FileNotFoundError(
        f'GPR 체크포인트를 찾을 수 없습니다: {GPR_CHECKPOINT}\n'
        'Step 1의 GPR 학습을 먼저 실행하세요.'
    )

ckpt = joblib.load(GPR_CHECKPOINT)
models    = ckpt['models']       # {y_col: fitted GPR} 15개 모델
scaler_X  = ckpt['scaler_X']    # P1~P6용 StandardScaler

print(f'  GPR 모델 로드 완료: {len(models)}개')

# -- Gatekeeper는 노트북에서 이미 학습된 상태 (변수명: gatekeeper) --
# 독립 .py로 실행 시 아래 주석 해제하여 별도 로드
# import pickle
# with open(os.path.join(BASE_DIR, 'checkpoints', 'gatekeeper.pkl'), 'rb') as f:
#     gatekeeper = pickle.load(f)

# Gatekeeper가 메모리에 없으면 여기서 학습 (독립 실행 대비)
try:
    gatekeeper
    print(f'  Gatekeeper 모델 확인 완료')
except NameError:
    print('  [경고] Gatekeeper가 메모리에 없습니다. 노트북에서 Step 2를 먼저 실행하세요.')

# -- Step 4 역설계 초안 로드 --
DRAFT_CSV = os.path.join(BASE_DIR, 'Inverse_Design_Results.csv')
df_drafts = pd.read_csv(DRAFT_CSV)
print(f'  Step 4 초안 로드: {len(df_drafts)}개')

# ====================================================================
# [2. 상수 정의]
# ====================================================================

# 7대 핵심 채널 (GPR R² 기준 신뢰 가능)
CORE_7_CHANNELS = [
    'WarpMax', 'T_Tip_Peel', 'Die_SY_Max', 'B_Avg_Peel',
    'B_Tip_SEQV', 'T_Tip_Strain', 'T_Tip_SEQV'
]

# 목적함수 채널 (2개)
OBJ_CHANNELS = ['WarpMax', 'T_Tip_Peel']

# Hard Constraint 채널 (5개) 및 물리적 한계치
CONSTRAINT_LIMITS = {
    'Die_SY_Max':   120.0,    # MPa, 실리콘 파괴 방어 (균열 임계 150~200 대비 안전 마진)
    'B_Avg_Peel':   0.1,      # MPa, 기판 박리 방어 (중앙부 Void/층간분리 시작점)
    'B_Tip_SEQV':   45.0,     # MPa, 하단 솔더 소성 변형 방어 (솔더 항복 응력 기준)
    'T_Tip_Strain': 0.02,     # 무차원, 계면 피로 누적 방어 (2% 초과 시 피로 파괴 가속)
    'T_Tip_SEQV':   65.0,     # MPa, 상단 계면 소성 변형 방어 (IHS/TIM 영구변형 방지)
}

# σ 계수 (강건 최적화: μ + SIGMA_COEFF * σ < Limit)
SIGMA_COEFF = 2.0  # 95% 신뢰구간

# DOE 바운더리
DOE_BOUNDS = {
    'P1': (0.8005, 1.0998),
    'P2': (0.0500, 0.0899),
    'P3': (0.6001, 0.7198),
    'P4': (0.1000, 0.2994),
    'P5': (1.2003, 1.7997),
    'P6': (0.0401, 0.0800),
}

P_LABELS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
N_VAR = 6          # 설계변수 수
N_OBJ = 2          # 목적함수 수
N_CONSTR = 6       # 제약조건 수 (Hard 5개 + Gatekeeper 1개)

# NSGA-II 파라미터
POP_SIZE = 200     # 인구 크기
N_GEN = 100        # 세대 수
DRAFT_MARGIN = 0.10  # 초안 기준 ±10% 탐색 범위

print('\n[2] 상수 정의 완료')
print(f'  목적함수: {OBJ_CHANNELS}')
print(f'  Hard Constraints: {list(CONSTRAINT_LIMITS.keys())}')
print(f'  σ 계수: {SIGMA_COEFF} (95% 신뢰구간)')
print(f'  NSGA-II: pop={POP_SIZE}, gen={N_GEN}, margin=±{DRAFT_MARGIN*100:.0f}%')


# ====================================================================
# [3. GPR 예측 헬퍼 함수]
# ====================================================================
def predict_with_gpr(X_raw, channels):
    """
    P1~P6 원본값 배열을 받아 GPR로 예측값(μ)과 불확실성(σ)을 반환.
    
    Parameters:
        X_raw: (n_samples, 6) numpy array, 원본 P1~P6 스케일
        channels: 예측할 채널명 리스트
    Returns:
        mu_dict: {채널명: (n_samples,) 예측값 배열}
        sigma_dict: {채널명: (n_samples,) 불확실성 배열}
    """
    # GPR은 정규화된 입력을 기대
    X_sc = scaler_X.transform(X_raw)
    
    mu_dict = {}
    sigma_dict = {}
    
    for ch in channels:
        mu, sigma = models[ch].predict(X_sc, return_std=True)
        mu_dict[ch] = mu
        sigma_dict[ch] = sigma
    
    return mu_dict, sigma_dict


# ====================================================================
# [4. pymoo 문제 정의 (Feasibility Rule 기반)]
# ====================================================================
class FlipchipOptProblem(Problem):
    """
    반도체 패키징 최적 설계 문제 정의 (pymoo 호환)
    
    목적함수 (2개, 최소화):
        f1 = |WarpMax_μ|      패키지 휨 최소화
        f2 = |T_Tip_Peel_μ|   계면 박리 최소화
    
    제약조건 (6개, g <= 0이면 만족):
        g0 = (|Die_SY_Max_μ| + 2σ) - 120     실리콘 파괴 방어
        g1 = (|B_Avg_Peel_μ| + 2σ) - 0.1     기판 박리 방어
        g2 = (|B_Tip_SEQV_μ| + 2σ) - 45      하단 솔더 소성 방어
        g3 = (|T_Tip_Strain_μ| + 2σ) - 0.02  계면 피로 방어
        g4 = (|T_Tip_SEQV_μ| + 2σ) - 65      상단 소성 방어
        g5 = 0.5 - Gatekeeper_prob            Safe 확률 50% 이상
    """
    
    def __init__(self, xl, xu):
        """
        Parameters:
            xl: (6,) 하한 배열
            xu: (6,) 상한 배열
        """
        super().__init__(
            n_var=N_VAR,
            n_obj=N_OBJ,
            n_ieq_constr=N_CONSTR,
            xl=xl,
            xu=xu
        )
    
    def _evaluate(self, X, out, *args, **kwargs):
        """
        X: (pop_size, 6) 개체 집단의 P1~P6
        out["F"]: (pop_size, 2) 목적함수값
        out["G"]: (pop_size, 6) 제약조건값 (<=0 만족)
        """
        n = X.shape[0]
        
        # -- GPR로 7채널 예측 (μ + σ) --
        mu_dict, sigma_dict = predict_with_gpr(X, CORE_7_CHANNELS)
        
        # -- 목적함수: |WarpMax_μ|, |T_Tip_Peel_μ| --
        f1 = np.abs(mu_dict['WarpMax'])
        f2 = np.abs(mu_dict['T_Tip_Peel'])
        out["F"] = np.column_stack([f1, f2])
        
        # -- 제약조건: (|μ| + 2σ) - Limit <= 0 --
        # g <= 0 이면 만족, g > 0 이면 위반 (위반량이 클수록 더 나쁨)
        constraint_channels = list(CONSTRAINT_LIMITS.keys())
        G = np.zeros((n, N_CONSTR))
        
        for i, ch in enumerate(constraint_channels):
            limit = CONSTRAINT_LIMITS[ch]
            # 강건 제약: 95% 신뢰구간 상한이 한계치 이내여야 함
            robust_value = np.abs(mu_dict[ch]) + SIGMA_COEFF * sigma_dict[ch]
            G[:, i] = robust_value - limit
        
        # -- Gatekeeper 제약: Safe(1)이어야 함 --
        gk_pred = gatekeeper.predict(X)
        # Safe(1) → g = 0-1 = -1 (만족), Fail(0) → g = 0-0 = 0... 
        # 약간의 마진을 두어: g = 0.5 - pred → Safe면 -0.5(만족), Fail이면 0.5(위반)
        G[:, 5] = 0.5 - gk_pred
        
        out["G"] = G


# ====================================================================
# [5. Phase 1 — 32개 초안 독립 NSGA-II 진화]
# ====================================================================
print('\n[5] Phase 1: 32개 초안별 독립 NSGA-II 진화')
print(f'    초안 수: {len(df_drafts)} | 인구: {POP_SIZE} | 세대: {N_GEN}')
print()

all_local_results = []  # 모든 로컬 파레토 개체 수집용
t_total_start = time.time()

for draft_idx, draft_row in df_drafts.iterrows():
    # -- 초안 P1~P6 추출 --
    draft_P = np.array([draft_row[p] for p in P_LABELS])
    row_id = int(draft_row['Row_ID'])
    
    # -- 탐색 범위: 초안 ±10%, DOE 바운더리 클리핑 --
    xl = np.zeros(N_VAR)
    xu = np.zeros(N_VAR)
    for i, p in enumerate(P_LABELS):
        doe_lo, doe_hi = DOE_BOUNDS[p]
        xl[i] = max(draft_P[i] * (1 - DRAFT_MARGIN), doe_lo)
        xu[i] = min(draft_P[i] * (1 + DRAFT_MARGIN), doe_hi)
        # 하한이 상한을 넘는 극단적 경우 방어
        if xl[i] >= xu[i]:
            xl[i] = doe_lo
            xu[i] = doe_hi
    
    # -- pymoo 문제 인스턴스 생성 --
    problem = FlipchipOptProblem(xl=xl, xu=xu)
    
    # -- NSGA-II 알고리즘 설정 --
    algorithm = NSGA2(
        pop_size=POP_SIZE,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20, prob=1.0/N_VAR),
        eliminate_duplicates=True
    )
    
    # -- 최적화 실행 --
    t_start = time.time()
    res = pymoo_minimize(
        problem,
        algorithm,
        ('n_gen', N_GEN),
        seed=SEED + draft_idx,
        verbose=False
    )
    elapsed = time.time() - t_start
    
    # -- 결과 수집 --
    if res.X is not None:
        # res.X: (n_solutions, 6) 파레토 최적 해
        # res.F: (n_solutions, 2) 목적함수값
        n_solutions = res.X.shape[0] if res.X.ndim > 1 else 1
        X_res = res.X.reshape(-1, N_VAR)
        F_res = res.F.reshape(-1, N_OBJ)
        
        for j in range(X_res.shape[0]):
            all_local_results.append({
                'draft_row_id': row_id,
                'P_values': X_res[j],
                'F_values': F_res[j],
            })
        
        print(f'  초안 #{draft_idx+1:2d} (Row_{row_id:4d}) | '
              f'파레토 해: {n_solutions:3d}개 | {elapsed:.1f}초 | '
              f'best f1={F_res[:,0].min():.6f}, f2={F_res[:,1].min():.4f}')
    else:
        print(f'  초안 #{draft_idx+1:2d} (Row_{row_id:4d}) | '
              f'실행 가능 해 없음 (모든 제약 위반) | {elapsed:.1f}초')

t_total = time.time() - t_total_start
print(f'\nPhase 1 완료: 총 {len(all_local_results):,}개 로컬 파레토 개체 수집 ({t_total:.1f}초)')


# ====================================================================
# [6. Phase 2 — 통합 결승전 (글로벌 비지배 정렬)]
# ====================================================================
print('\n[6] Phase 2: 통합 결승전 — 글로벌 파레토 추출')

if len(all_local_results) == 0:
    print('[오류] 실행 가능한 해가 하나도 없습니다.')
    print('  → Hard Constraint 한계치를 완화하거나 탐색 범위를 넓히세요.')
else:
    # 모든 로컬 결과를 행렬로 통합
    all_X = np.array([r['P_values'] for r in all_local_results])   # (N_total, 6)
    all_F = np.array([r['F_values'] for r in all_local_results])   # (N_total, 2)
    all_draft_ids = [r['draft_row_id'] for r in all_local_results]
    
    print(f'  통합 풀: {all_X.shape[0]:,}개 개체')
    
    # -- 글로벌 비지배 정렬 --
    nds = NonDominatedSorting()
    fronts = nds.do(all_F)
    
    # Frontier 0 (글로벌 1등) 추출
    global_pareto_idx = fronts[0]
    pareto_X = all_X[global_pareto_idx]
    pareto_F = all_F[global_pareto_idx]
    pareto_draft_ids = [all_draft_ids[i] for i in global_pareto_idx]
    
    print(f'  글로벌 Frontier 0: {len(global_pareto_idx)}개')
    
    # -- 글로벌 파레토에 대해 7채널 GPR 예측 (μ + σ) --
    mu_dict, sigma_dict = predict_with_gpr(pareto_X, CORE_7_CHANNELS)


# ====================================================================
# [7. Top 5 선정 — Knee Point + 4개 특성별 대표]
# ====================================================================
print('\n[7] Top 5 선정')

if len(global_pareto_idx) >= 5:
    
    # --- #1: Knee Point (원점 최근접, 정규화 유클리드 거리) ---
    # 두 목적함수를 [0,1]로 정규화한 뒤 원점과의 거리가 최소인 해
    f_min = pareto_F.min(axis=0)
    f_max = pareto_F.max(axis=0)
    f_range = f_max - f_min
    f_range[f_range == 0] = 1e-10  # 0 나눗셈 방지
    
    f_normalized = (pareto_F - f_min) / f_range
    distances = np.sqrt((f_normalized ** 2).sum(axis=1))
    knee_idx = np.argmin(distances)
    
    # --- #2: WarpMax 최소 ---
    warp_best_idx = np.argmin(pareto_F[:, 0])
    
    # --- #3: T_Tip_Peel 최소 ---
    peel_best_idx = np.argmin(pareto_F[:, 1])
    
    # --- #4: 파레토 중간점 (정규화 거리 중앙값) ---
    median_dist = np.median(distances)
    mid_idx = np.argmin(np.abs(distances - median_dist))
    
    # --- #5: σ 총합 최소 (불확실성 최소 = 예측 신뢰도 최고) ---
    total_sigma = np.zeros(len(global_pareto_idx))
    for ch in CORE_7_CHANNELS:
        total_sigma += sigma_dict[ch]
    sigma_best_idx = np.argmin(total_sigma)
    
    # 중복 제거 (같은 인덱스가 선택된 경우 차순위로 대체)
    selected_indices = []
    selected_labels = []
    candidates = [
        (knee_idx, 'Knee Point (추천, 최적 밸런스)'),
        (warp_best_idx, 'WarpMax 최소 우선'),
        (peel_best_idx, 'T_Tip_Peel 최소 우선'),
        (mid_idx, '파레토 중간 트레이드오프'),
        (sigma_best_idx, 'σ 총합 최소 (최고 신뢰도)'),
    ]
    
    # 중복 시 거리 순서로 대체 후보 추가
    dist_sorted_indices = np.argsort(distances)
    backup_ptr = 0
    
    for idx, label in candidates:
        if idx not in selected_indices:
            selected_indices.append(idx)
            selected_labels.append(label)
        else:
            # 중복 → 아직 선택되지 않은 거리순 다음 후보
            while backup_ptr < len(dist_sorted_indices):
                alt = dist_sorted_indices[backup_ptr]
                backup_ptr += 1
                if alt not in selected_indices:
                    selected_indices.append(alt)
                    selected_labels.append(label + ' (대체)')
                    break

else:
    # 파레토 해가 5개 미만인 경우 전부 선택
    selected_indices = list(range(len(global_pareto_idx)))
    selected_labels = [f'Frontier 0 #{i+1}' for i in range(len(selected_indices))]
    print(f'  [주의] 글로벌 파레토가 {len(global_pareto_idx)}개로 5개 미만 → 전부 선택')


# ====================================================================
# [8. 결과 DataFrame 구성 및 CSV 저장]
# ====================================================================
print('\n[8] 최종 결과 저장')

result_rows = []

for rank, (sel_idx, label) in enumerate(zip(selected_indices, selected_labels)):
    row = {
        'Rank': rank + 1,
        'Label': label,
        'Source_Draft_Row_ID': pareto_draft_ids[sel_idx],
    }
    
    # P1~P6
    for i, p in enumerate(P_LABELS):
        row[p] = pareto_X[sel_idx, i]
    
    # 7채널 GPR 예측값 (절댓값 Max Peak)
    for ch in CORE_7_CHANNELS:
        row[f'{ch}_pred'] = np.abs(mu_dict[ch][sel_idx])
    
    # 7채널 GPR 불확실성 (σ)
    for ch in CORE_7_CHANNELS:
        row[f'{ch}_sigma'] = sigma_dict[ch][sel_idx]
    
    result_rows.append(row)

df_final = pd.DataFrame(result_rows)

# CSV 저장
output_path = os.path.join(BASE_DIR, 'GA_Optimized_Results.csv')
df_final.to_csv(output_path, index=False)

print(f'  저장 완료: {output_path}')
print(f'  Top {len(selected_indices)}개 최적 설계안\n')

# -- 결과 테이블 출력 --
print('=' * 120)
print(f'{"Rank":>4s} | {"Label":>35s} | ', end='')
for p in P_LABELS:
    print(f'{p:>8s}', end=' ')
print(f' | {"WarpMax":>9s} {"T_Tip_Peel":>11s}')
print('=' * 120)

for _, row in df_final.iterrows():
    rank = int(row['Rank'])
    marker = '★' if rank == 1 else ' '
    print(f'{marker}{rank:>3d} | {row["Label"]:>35s} | ', end='')
    for p in P_LABELS:
        print(f'{row[p]:8.4f}', end=' ')
    print(f' | {row["WarpMax_pred"]:9.6f} {row["T_Tip_Peel_pred"]:11.4f}')

print('=' * 120)
print(f'\n★ Rank 1 = Knee Point: WarpMax와 T_Tip_Peel의 최적 밸런스 (추천 설계안)')


# ====================================================================
# [9. Hard Constraint 만족 여부 확인]
# ====================================================================
print('\n[9] Hard Constraint 만족 여부 (μ + 2σ 기준)')
print(f'{"Rank":>4s} | ', end='')
for ch in CONSTRAINT_LIMITS:
    print(f'{ch:>15s}', end=' ')
print('| 판정')
print('-' * 110)

for _, row in df_final.iterrows():
    rank = int(row['Rank'])
    print(f'{rank:>4d} | ', end='')
    all_pass = True
    for ch, limit in CONSTRAINT_LIMITS.items():
        robust_val = row[f'{ch}_pred'] + SIGMA_COEFF * row[f'{ch}_sigma']
        passed = robust_val < limit
        if not passed:
            all_pass = False
        mark = '✓' if passed else '✗'
        print(f'{robust_val:12.4f}{mark:>3s}', end=' ')
    verdict = '  PASS' if all_pass else '  FAIL'
    print(f'|{verdict}')

print(f'\n  기준: |μ| + {SIGMA_COEFF}σ < Limit (95% 신뢰구간 강건 제약)')


# ====================================================================
# [10. 파레토 프론티어 시각화]
# ====================================================================
print('\n[10] 파레토 프론티어 시각화')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- (A) 글로벌 파레토 + Top 5 표시 ---
ax = axes[0]

# 전체 로컬 해 (배경)
ax.scatter(all_F[:, 0], all_F[:, 1], s=3, alpha=0.1, c='gray', label='All local solutions')

# 글로벌 파레토 (중경)
ax.scatter(pareto_F[:, 0], pareto_F[:, 1], s=15, alpha=0.5, c='steelblue',
           edgecolors='white', linewidths=0.3, label=f'Global Pareto ({len(global_pareto_idx)})')

# Top 5 (전경)
colors_top5 = ['red', 'darkorange', 'gold', 'limegreen', 'deepskyblue']
for i, (sel_idx, label) in enumerate(zip(selected_indices, selected_labels)):
    ax.scatter(pareto_F[sel_idx, 0], pareto_F[sel_idx, 1],
               s=120, c=colors_top5[i], edgecolors='black', linewidths=1.5,
               marker='*' if i == 0 else 'o', zorder=10,
               label=f'#{i+1} {label[:20]}')

ax.set_xlabel('|WarpMax| (minimize)')
ax.set_ylabel('|T_Tip_Peel| (minimize)')
ax.set_title('Global Pareto Frontier + Top 5', fontweight='bold')
ax.legend(fontsize=6, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.3)

# --- (B) Top 5의 P1~P6 레이더 차트 (정규화) ---
ax = axes[1]

# P 값을 DOE 범위 기준 [0,1]로 정규화
angles = np.linspace(0, 2 * np.pi, N_VAR, endpoint=False).tolist()
angles += angles[:1]  # 닫기

for i, (sel_idx, label) in enumerate(zip(selected_indices, selected_labels)):
    p_vals = pareto_X[sel_idx]
    p_normalized = []
    for j, p in enumerate(P_LABELS):
        lo, hi = DOE_BOUNDS[p]
        p_normalized.append((p_vals[j] - lo) / (hi - lo))
    p_normalized += p_normalized[:1]  # 닫기
    
    color = colors_top5[i]
    linewidth = 2.5 if i == 0 else 1.2
    ax.plot(angles, p_normalized, color=color, linewidth=linewidth,
            label=f'#{i+1}', marker='o', markersize=4)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(P_LABELS, fontsize=9)
ax.set_ylim(0, 1)
ax.set_title('Top 5 Design Variables (Normalized to DOE Range)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


# ====================================================================
# [11. 완료 요약]
# ====================================================================
print('\n' + '=' * 80)
print('  [Step 5 완료] NSGA-II + GPR 강건 최적화 결과 요약')
print('=' * 80)
print(f'  Phase 1: {len(df_drafts)}개 초안 × {POP_SIZE}개체 × {N_GEN}세대 진화')
print(f'  Phase 2: {len(all_local_results):,}개 통합 → 글로벌 Frontier 0: {len(global_pareto_idx)}개')
print(f'  최종 선정: Top {len(selected_indices)}개')
print(f'  저장 파일: {output_path}')
print()
print(f'  ★ 추천 설계안 (Rank 1 — Knee Point):')
best = df_final[df_final['Rank'] == 1].iloc[0]
for p in P_LABELS:
    print(f'    {p} = {best[p]:.4f}')
print(f'    → |WarpMax| = {best["WarpMax_pred"]:.6f}')
print(f'    → |T_Tip_Peel| = {best["T_Tip_Peel_pred"]:.4f}')
print()
print('  → 이 P1~P6를 [Step 6: Ansys 디지털 트윈]에 입력하여 최종 검증')
print('=' * 80)
