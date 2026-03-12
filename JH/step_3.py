# ====================================================================
# [1. 환경 및 타겟 설정]
# ====================================================================
CORE_7_CHANNELS = [
    'WarpMax', 'Die_SY_Max', 'B_Avg_Peel', 'B_Tip_SEQV', 
    'T_Tip_Strain', 'T_Tip_Peel', 'T_Tip_SEQV'
]

UTOPIA_RATIO = 0.90 
TENSOR_DIR = os.path.join(BASE_DIR, 'Utopia_Target_Tensors')  # 절대 경로 통일
os.makedirs(TENSOR_DIR, exist_ok=True)

print("=== [Step 3] 파레토 비지배 정렬 및 유토피아 타겟 추출 ===")

# ====================================================================
# [2. 다단계 Pareto Non-dominated Sorting (Frontier 0 + 1 + ...)]
# ====================================================================
obj1 = df_peaks['WarpMax'].abs().values
obj2 = df_peaks['T_Tip_Peel'].abs().values
scores = np.column_stack((obj1, obj2))

population_size = scores.shape[0]

# 다단계 비지배 정렬: Frontier 0, 1, 2, ... 순서로 계층 분류
# Frontier 0 = 최상위 (아무에게도 지배 안 당함)
# Frontier 1 = Frontier 0 제거 후 비지배 해
# ...반복
frontier_labels = np.full(population_size, -1, dtype=int)  # 각 DP의 Frontier 등급
remaining = np.ones(population_size, dtype=bool)             # 아직 분류 안 된 DP
frontier_level = 0

while remaining.any():
    remaining_idx = np.where(remaining)[0]
    current_scores = scores[remaining_idx]
    is_pareto = np.ones(len(remaining_idx), dtype=bool)
    
    for i in range(len(remaining_idx)):
        for j in range(len(remaining_idx)):
            if i == j:
                continue
            # j가 i를 지배하는지 판정
            if all(current_scores[j] <= current_scores[i]) and any(current_scores[j] < current_scores[i]):
                is_pareto[i] = False
                break
    
    # 현재 Frontier에 해당하는 DP에 등급 부여
    for k, idx in enumerate(remaining_idx):
        if is_pareto[k]:
            frontier_labels[idx] = frontier_level
            remaining[idx] = False
    
    print(f'  Frontier {frontier_level}: {is_pareto.sum()}개')
    frontier_level += 1
    
    # 안전 장치: 최대 10단계까지만
    if frontier_level >= 10:
        break

# Frontier 등급을 df_peaks에 추가
df_peaks['frontier'] = frontier_labels

# === 파레토 상위 N% 또는 최소 수량 확보 ===
MIN_PARETO_COUNT = 30  # 최소 확보 목표 (Step 4 학습에 필요한 하한)

# Frontier 0부터 순서대로 누적하여 최소 수량 충족될 때까지 확장
selected_frontiers = []
cumulative = 0
for level in range(frontier_level):
    count_at_level = (frontier_labels == level).sum()
    selected_frontiers.append(level)
    cumulative += count_at_level
    if cumulative >= MIN_PARETO_COUNT:
        break

df_pareto = df_peaks[df_peaks['frontier'].isin(selected_frontiers)].copy()
num_pareto = len(df_pareto)
max_frontier = max(selected_frontiers)

print(f'\n총 {population_size}개 중 파레토 Frontier 0~{max_frontier}: {num_pareto}개 ({num_pareto/population_size*100:.1f}%)')
print(f'  (Frontier 0만: {(frontier_labels==0).sum()}개 → 부족하여 Frontier {max_frontier}까지 확장)')

if num_pareto < MIN_PARETO_COUNT:
    print(f'[경고] {num_pareto}개로 목표({MIN_PARETO_COUNT})에 미달. 가용 데이터 전부 사용.')

    # ====================================================================
# [3. 다단계 파레토 프론티어 시각화]
# ====================================================================
plt.figure(figsize=(9, 6))

# 전체 데이터 (회색 배경)
plt.scatter(obj1, obj2, color='lightgray', alpha=0.3, s=10, label='All Survived Data')

# Frontier 등급별 색상 구분
colors = ['red', 'darkorange', 'gold', 'limegreen', 'deepskyblue']
for level in selected_frontiers:
    mask = frontier_labels == level
    n = mask.sum()
    c = colors[level] if level < len(colors) else 'gray'
    plt.scatter(obj1[mask], obj2[mask], color=c, s=40, edgecolors='black',
                linewidths=0.5, label=f'Frontier {level} ({n}개)', zorder=5-level)

plt.title('Multi-level Pareto Frontier: WarpMax vs T_Tip_Peel', fontweight='bold')
plt.xlabel('|WarpMax| (abs)')
plt.ylabel('|T_Tip_Peel| (abs)')
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print(f'→ 빨간색(Frontier 0)에 가까울수록 WarpMax와 T_Tip_Peel이 동시에 작은 우수 설계')
print(f'→ 총 {num_pareto}개의 유토피아 타겟 텐서가 Step 4로 전달됨')