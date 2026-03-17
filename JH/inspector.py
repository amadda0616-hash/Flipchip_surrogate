# ====================================================================
# [11. 완료 요약 및 대리모델 vs 실제 CAE 데이터 교차 검증]
# ====================================================================
import os
import pandas as pd
import numpy as np

print('\n' + '=' * 80)
print('  [Step 5 완료] NSGA-II + GPR 강건 최적화 결과 요약 및 검증')
print('=' * 80)
print(f'  Phase 1: {len(df_drafts)}개 초안 × {POP_SIZE}개체 × {N_GEN}세대 진화')
print(f'  Phase 2: {len(all_local_results):,}개 통합 → 글로벌 Frontier 0: {len(global_pareto_idx)}개')
print(f'  최종 선정: Top {len(selected_indices)}개')
print(f'  저장 파일: {output_path}')
print()

# 1. Knee Point 추천 설계안 추출
print(f'  ★ 추천 설계안 (Rank 1 — Knee Point):')
best = df_final[df_final['Rank'] == 1].iloc[0]
for p in P_LABELS:
    print(f'    {p} = {best[p]:.4f}')
print()

# ====================================================================
# [검증] GPR 대리모델 예측 vs 실제 CAE 시뮬레이션 교차 검증
# ====================================================================
cae_file = os.path.join(BASE_DIR, 'ML_DATA_Extract_JH.csv')  # 절대 경로 통일

if os.path.exists(cae_file):
    print('=' * 80)
    print(f'  [검증] GPR 대리모델 예측치 vs 실제 CAE 시뮬레이션 일치율 평가')
    print(f'  CAE 파일: {cae_file}')
    print(f'  평가 기준: Step 1과 동일한 "절댓값 Max Peak (부호 유지)"')
    print('-' * 100)
    print(f'  {"채널":<16s} | {"CAE 피크":>12s} | {"GPR 예측":>12s} | {"GPR σ":>10s} | {"오차율":>8s} | {"일치율":>8s} | {"GPR R²":>8s} | 판정')
    print('-' * 100)

    # 실제 CAE 데이터 로드
    df_cae = pd.read_csv(cae_file)
    df_cae.columns = [c.strip() for c in df_cae.columns]

    # GPR로 Rank 1의 P1~P6에 대해 15채널 전체 예측
    X_best = best[P_LABELS].values.astype(float).reshape(1, -1)
    mu_dict, sigma_dict = predict_with_gpr(X_best, Y_COLUMNS)

    # 신뢰 채널 구분 (GPR R² 기준)
    HIGH_R2_CHANNELS = ['WarpMax', 'T_Tip_Peel', 'Die_SY_Max', 'B_Avg_Peel',
                        'B_Tip_SEQV', 'T_Tip_Strain', 'T_Tip_SEQV']

    total_match = 0
    high_r2_match = 0
    high_r2_count = 0

    for y_col in Y_COLUMNS:
        if y_col not in df_cae.columns:
            continue

        # 실제 CAE 시계열에서 절댓값 Max Peak 추출 (Step 1과 동일 로직)
        max_abs_idx = df_cae[y_col].abs().idxmax()
        true_peak = df_cae.loc[max_abs_idx, y_col]

        # GPR 예측값
        pred_peak = mu_dict[y_col][0]
        pred_sigma = sigma_dict[y_col][0]

        # 오차율 계산
        if true_peak != 0:
            error_rate = abs((pred_peak - true_peak) / true_peak) * 100
        else:
            error_rate = 0.0 if pred_peak == 0 else 100.0

        match_rate = max(0.0, 100.0 - error_rate)
        total_match += match_rate

        # GPR R² 표시 (Step 1에서 학습된 성능)
        r2 = test_scores.get(y_col, float('nan'))
        is_reliable = y_col in HIGH_R2_CHANNELS
        reliability = '●' if is_reliable else '○'

        if is_reliable:
            high_r2_match += match_rate
            high_r2_count += 1

        print(f'  {y_col:<16s} | {true_peak:>12.4f} | {pred_peak:>12.4f} | {pred_sigma:>10.4f} | '
              f'{error_rate:>7.2f}% | {match_rate:>7.2f}% | {r2:>7.4f} | {reliability}')

    print('-' * 100)
    avg_all = total_match / len(Y_COLUMNS)
    avg_high = high_r2_match / high_r2_count if high_r2_count > 0 else 0
    print(f'  전체 15채널 평균 일치율: {avg_all:.2f}%')
    print(f'  신뢰 7채널 평균 일치율: {avg_high:.2f}%  (● 표시 채널)')
    print(f'  ● = GPR R² ≥ 0.58 (신뢰 가능)  ○ = GPR R² < 0.58 (참고용)')

else:
    print(f'  [경고] {cae_file} 파일을 찾을 수 없어 교차 검증을 건너뜁니다.')

print()
print('  → 이 P1~P6를 [Step 6: Ansys 디지털 트윈]에 입력하여 최종 검증 완료')
print('=' * 80)