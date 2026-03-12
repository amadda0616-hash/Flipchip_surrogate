"""
================================================================================
# [Step 4] 딥러닝 기반 역설계 — 오토인코더 잠재 매핑 (Autoencoder Latent Mapping)
================================================================================

## 목표
Step 3에서 생성된 유토피아 타겟 텐서(7채널 × 617 timestep)를 입력하면,
이를 구현할 수 있는 **최적의 P1~P6 설계변수 초안**을 출력한다.

## 전략: 2단계 잠재 공간 매핑
고차원 시계열(617×7 = 4,319차원)을 직접 역매핑하면 일대다(one-to-many) 문제로
수렴이 불안정하다. 대신 오토인코더로 **잠재 공간(32차원)**에 압축한 뒤,
저차원에서 P↔z 매핑을 학습하여 안정적인 역설계를 수행한다.

## 파이프라인 흐름

    [Step 4-1] 시계열 오토인코더 학습 (비지도, 원본 ~900개 전부 사용)
        입력: 원본 시계열 (617 × 7ch)
          ↓ Encoder (1D-CNN): 시계열 → 잠재 벡터 z (32차원)
          ↓ Decoder (1D-ConvTranspose): z → 시계열 복원
          ↓ 복원 오차(MSE) 최소화
        결과: Encoder/Decoder 확보

    [Step 4-2] 잠재 공간 매핑 학습 (지도, ~900개)
        순방향: P1~P6 → z  (MLP)
        역방향: z → P1~P6  (MLP)

    [Step 4-3] 유토피아 타겟 역설계 (추론)
        유토피아 텐서 → Encoder → z_target → 역매핑 MLP → P1~P6 초안
        → Step 5 GA 미세조정의 시작점(±10%)으로 전달

## 출력
    - P1~P6 초안 (파레토 DP별)
    - Inverse_Design_Results.csv → Step 5로 전달
"""

# ====================================================================
# [0. 라이브러리 및 환경 설정]
# ====================================================================
import os
import platform
import glob
import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 9

# -- 디바이스 설정 (GPU 가용 시 자동 사용) --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- 경로 설정 --
if platform.system() == 'Linux':
    CSV_FOLDER  = '/mnt/i/ai_model_dev/cfd/SIM_CSV_DATA'
    MASTER_CSV  = '/mnt/i/ai_model_dev/cfd/Master_DOE_1200.csv'
    BASE_DIR    = '/mnt/i/ai_model_dev/cfd'
else:
    CSV_FOLDER  = r'I:\ai_model_dev\cfd\SIM_CSV_DATA'
    MASTER_CSV  = r'I:\ai_model_dev\cfd\Master_DOE_1200.csv'
    BASE_DIR    = r'I:\ai_model_dev\cfd'

TENSOR_DIR = os.path.join(BASE_DIR, 'Utopia_Target_Tensors')

# Step 3에서 확정된 7대 핵심 채널 (GPR ARD R² 기준 신뢰 가능 채널)
CORE_7_CHANNELS = [
    'WarpMax', 'Die_SY_Max', 'B_Avg_Peel', 'B_Tip_SEQV',
    'T_Tip_Strain', 'T_Tip_Peel', 'T_Tip_SEQV'
]

N_CHANNELS  = len(CORE_7_CHANNELS)   # 7
LATENT_DIM  = 32                      # 잠재 공간 차원 수
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f'=== [Step 4] 오토인코더 잠재 매핑 역설계 ===')
print(f'디바이스    : {device}')
print(f'핵심 채널   : {N_CHANNELS}개 {CORE_7_CHANNELS}')
print(f'잠재 차원   : {LATENT_DIM}')
print(f'텐서 폴더   : {TENSOR_DIR}')


# ====================================================================
# [1. 원본 시계열 데이터 로드 (학습용)]
# ====================================================================
# 원본 ~900개의 생존 CSV에서 7채널 시계열을 텐서로 변환
# 오토인코더는 비지도 학습이므로 P1~P6 라벨 없이 시계열만 필요
# (단, 4-2에서 P↔z 매핑 학습 시 P1~P6도 사용)

print('\n[1] 원본 시계열 로드 중...')

# 마스터 DOE 로드 (P1~P6)
df_master = pd.read_csv(MASTER_CSV)
df_master.insert(0, 'Row_ID', range(1, len(df_master) + 1))

# 생존 CSV 탐색
pattern = os.path.join(CSV_FOLDER, 'ML_DATA_Extract_Row_*.csv')
found_files = sorted(glob.glob(pattern))

survived_ids = []
for fpath in found_files:
    match = re.search(r'Row_(\d+)\.csv', os.path.basename(fpath))
    if match:
        survived_ids.append(int(match.group(1)))
survived_ids = sorted(survived_ids)

# 시계열 텐서 + P1~P6 동시 수집
all_timeseries = []   # shape: (N, timesteps, 7)
all_params = []       # shape: (N, 6)
valid_row_ids = []

for row_id in survived_ids:
    fpath = os.path.join(CSV_FOLDER, f'ML_DATA_Extract_Row_{row_id}.csv')
    master_row = df_master[df_master['Row_ID'] == row_id]
    if master_row.empty:
        continue

    try:
        df_ts = pd.read_csv(fpath)
        df_ts.columns = [c.strip() for c in df_ts.columns]

        # 7채널만 추출
        ts_7ch = df_ts[CORE_7_CHANNELS].values  # (617, 7)

        # NaN이나 Inf 체크
        if np.isnan(ts_7ch).any() or np.isinf(ts_7ch).any():
            continue

        all_timeseries.append(ts_7ch)
        all_params.append(master_row[['P1','P2','P3','P4','P5','P6']].values[0])
        valid_row_ids.append(row_id)

    except Exception as e:
        pass

# numpy 배열로 변환
X_ts = np.array(all_timeseries, dtype=np.float32)   # (N, 617, 7)
X_params = np.array(all_params, dtype=np.float32)    # (N, 6)

N_SAMPLES, N_TIMESTEPS, _ = X_ts.shape

print(f'로드 완료: {N_SAMPLES}개 시계열')
print(f'시계열 shape: {X_ts.shape} → (샘플, 타임스텝, 채널)')
print(f'P1~P6 shape : {X_params.shape}')


# ====================================================================
# [2. 데이터 전처리 (채널별 StandardScaler)]
# ====================================================================
# 오토인코더 학습 전 각 채널을 독립적으로 정규화
# 채널별 스케일 차이가 크므로 (WarpMax: ~0.1, T_Tip_SEQV: ~40) 필수

print('\n[2] 채널별 StandardScaler 적용...')

# 채널별 scaler 저장 (추론 시 역변환에 필요)
channel_scalers = []
X_ts_scaled = np.zeros_like(X_ts)

for ch in range(N_CHANNELS):
    scaler = StandardScaler()
    # (N, 617) → fit_transform → (N, 617)
    ch_data = X_ts[:, :, ch]                        # 모든 샘플의 ch번째 채널
    X_ts_scaled[:, :, ch] = scaler.fit_transform(ch_data)
    channel_scalers.append(scaler)
    print(f'  {CORE_7_CHANNELS[ch]:15s} | mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}')

# P1~P6도 정규화 (Step 4-2에서 사용)
param_scaler = StandardScaler()
X_params_scaled = param_scaler.fit_transform(X_params)

# PyTorch 텐서로 변환
# Conv1d는 (batch, channels, length) 형태를 기대하므로 transpose
X_tensor = torch.FloatTensor(X_ts_scaled).permute(0, 2, 1)  # (N, 7, 617)
P_tensor = torch.FloatTensor(X_params_scaled)                 # (N, 6)

print(f'\nPyTorch 텐서 shape: X={X_tensor.shape}, P={P_tensor.shape}')


# ====================================================================
# [3. 오토인코더 모델 정의]
# ====================================================================
# Encoder: 1D-CNN으로 (7, 617) 시계열을 32차원 잠재 벡터로 압축
# Decoder: 1D-ConvTranspose로 32차원에서 (7, 617) 시계열을 복원

class Encoder(nn.Module):
    """
    1D-CNN Encoder: (batch, 7, 617) → (batch, 32)
    3단계 Conv Block으로 시간축을 점진적으로 압축
    """
    def __init__(self, n_channels=7, latent_dim=32):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1: (7, 617) → (32, 308)
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            # Block 2: (32, 308) → (64, 154)
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            # Block 3: (64, 154) → (128, 77)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )
        # Global Average Pooling + FC로 잠재 벡터 생성
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (128, 77) → (128, 1)
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x):
        # x: (batch, 7, 617)
        h = self.conv_blocks(x)        # (batch, 128, 77)
        h = self.global_pool(h)        # (batch, 128, 1)
        h = h.squeeze(-1)              # (batch, 128)
        z = self.fc(h)                 # (batch, 32)
        return z


class Decoder(nn.Module):
    """
    1D-CNN Decoder: (batch, 32) → (batch, 7, 617)
    Encoder의 역순으로 시간축을 복원
    """
    def __init__(self, n_channels=7, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 77)
        self.deconv_blocks = nn.Sequential(
            # Block 1: (128, 77) → (64, 154)
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            # Block 2: (64, 154) → (32, 308)
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            # Block 3: (32, 308) → (7, 617)
            # output_padding=0으로 308*2=616 → 이후 Interpolate로 617에 맞춤
            nn.ConvTranspose1d(32, n_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, z):
        # z: (batch, 32)
        h = self.fc(z)                              # (batch, 128*77)
        h = h.view(-1, 128, 77)                     # (batch, 128, 77)
        h = self.deconv_blocks(h)                   # (batch, 7, ~617)
        # ConvTranspose 출력 길이가 정확히 617이 아닐 수 있으므로 보간
        h = torch.nn.functional.interpolate(h, size=617, mode='linear', align_corners=False)
        return h


class TimeSeriesAutoencoder(nn.Module):
    """
    시계열 오토인코더 = Encoder + Decoder
    학습 시: 입력 시계열 → Encoder → z → Decoder → 복원 시계열
    추론 시: Encoder만 사용하여 잠재 벡터 z를 추출
    """
    def __init__(self, n_channels=7, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(n_channels, latent_dim)
        self.decoder = Decoder(n_channels, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


print('[3] 오토인코더 모델 구조:')
ae_model = TimeSeriesAutoencoder(N_CHANNELS, LATENT_DIM).to(device)
total_params = sum(p.numel() for p in ae_model.parameters())
print(f'  총 파라미터: {total_params:,}개')
print(f'  Encoder: (batch, {N_CHANNELS}, {N_TIMESTEPS}) → (batch, {LATENT_DIM})')
print(f'  Decoder: (batch, {LATENT_DIM}) → (batch, {N_CHANNELS}, {N_TIMESTEPS})')


# ====================================================================
# [4. 오토인코더 학습 (비지도)]
# ====================================================================
# 원본 ~900개 시계열의 복원 오차(MSE)를 최소화
# 라벨(P1~P6)이 필요 없으므로 전체 데이터를 비지도로 학습

AE_EPOCHS     = 200       # 에포크 수
AE_BATCH_SIZE = 32        # 배치 크기
AE_LR         = 1e-3      # 학습률
AE_PATIENCE   = 20        # Early Stopping 인내심

print(f'\n[4] 오토인코더 학습 시작 (epochs={AE_EPOCHS}, batch={AE_BATCH_SIZE}, lr={AE_LR})')

# Train/Val 분리 (85:15)
n_train = int(N_SAMPLES * 0.85)
indices = np.random.permutation(N_SAMPLES)
train_idx, val_idx = indices[:n_train], indices[n_train:]

train_loader = DataLoader(
    TensorDataset(X_tensor[train_idx]),
    batch_size=AE_BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_tensor[val_idx]),
    batch_size=AE_BATCH_SIZE, shuffle=False
)

# 옵티마이저 및 스케줄러
optimizer_ae = optim.Adam(ae_model.parameters(), lr=AE_LR, weight_decay=1e-5)
scheduler_ae = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ae, patience=10, factor=0.5)
criterion_ae = nn.MSELoss()

# 학습 루프
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0

t_start = time.time()

for epoch in range(AE_EPOCHS):
    # -- Train --
    ae_model.train()
    epoch_train_loss = 0
    for (batch_x,) in train_loader:
        batch_x = batch_x.to(device)
        x_recon, z = ae_model(batch_x)
        loss = criterion_ae(x_recon, batch_x)
        optimizer_ae.zero_grad()
        loss.backward()
        optimizer_ae.step()
        epoch_train_loss += loss.item() * len(batch_x)
    epoch_train_loss /= n_train

    # -- Validation --
    ae_model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for (batch_x,) in val_loader:
            batch_x = batch_x.to(device)
            x_recon, z = ae_model(batch_x)
            loss = criterion_ae(x_recon, batch_x)
            epoch_val_loss += loss.item() * len(batch_x)
    epoch_val_loss /= (N_SAMPLES - n_train)

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    scheduler_ae.step(epoch_val_loss)

    # Early Stopping 체크
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        # 베스트 모델 저장
        best_ae_state = ae_model.state_dict().copy()
    else:
        patience_counter += 1

    # 진행 출력 (20에포크마다)
    if (epoch + 1) % 20 == 0:
        lr_now = optimizer_ae.param_groups[0]['lr']
        print(f'  Epoch {epoch+1:3d}/{AE_EPOCHS} | '
              f'Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | '
              f'LR: {lr_now:.2e} | Patience: {patience_counter}/{AE_PATIENCE}')

    if patience_counter >= AE_PATIENCE:
        print(f'  → Early Stopping at epoch {epoch+1}')
        break

# 베스트 모델 복원
ae_model.load_state_dict(best_ae_state)
elapsed = time.time() - t_start
print(f'\n오토인코더 학습 완료 ({elapsed:.1f}초)')
print(f'최종 Val Loss: {best_val_loss:.6f}')


# ====================================================================
# [5. 오토인코더 학습 결과 시각화]
# ====================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# --- (A) 학습 곡선 ---
ax = axes[0]
ax.plot(train_losses, label='Train Loss', color='steelblue')
ax.plot(val_losses, label='Val Loss', color='coral')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Autoencoder Training Curve', fontweight='bold')
ax.legend()
ax.set_yscale('log')

# --- (B) 복원 품질 예시 (첫 번째 검증 샘플의 WarpMax 채널) ---
ax = axes[1]
ae_model.eval()
with torch.no_grad():
    sample_x = X_tensor[val_idx[0]:val_idx[0]+1].to(device)
    sample_recon, _ = ae_model(sample_x)
    sample_x_np = sample_x.cpu().numpy()[0]           # (7, 617)
    sample_recon_np = sample_recon.cpu().numpy()[0]    # (7, 617)

# WarpMax 채널 (index 0) 비교
ch_idx = 0
ax.plot(sample_x_np[ch_idx], label='Original', color='steelblue', linewidth=1.5)
ax.plot(sample_recon_np[ch_idx], label='Reconstructed', color='coral', linewidth=1.5, linestyle='--')
ax.set_xlabel('Timestep')
ax.set_ylabel(f'{CORE_7_CHANNELS[ch_idx]} (scaled)')
ax.set_title(f'Reconstruction Quality: {CORE_7_CHANNELS[ch_idx]}', fontweight='bold')
ax.legend(fontsize=8)

# --- (C) 잠재 공간 2D 시각화 (t-SNE 대신 첫 2차원 사용) ---
ax = axes[2]
ae_model.eval()
with torch.no_grad():
    all_z = ae_model.encoder(X_tensor.to(device)).cpu().numpy()  # (N, 32)
ax.scatter(all_z[:, 0], all_z[:, 1], s=5, alpha=0.5, c='steelblue')
ax.set_xlabel('Latent Dim 0')
ax.set_ylabel('Latent Dim 1')
ax.set_title('Latent Space (First 2 Dims)', fontweight='bold')

plt.tight_layout()
plt.show()


# ====================================================================
# [6. 잠재 벡터 추출 (전체 데이터)]
# ====================================================================
# 학습된 Encoder로 ~900개 전체의 잠재 벡터 z를 추출
# 이 z와 P1~P6의 쌍으로 Step 4-2 매핑을 학습

print('\n[6] 전체 데이터 잠재 벡터 추출...')
ae_model.eval()
with torch.no_grad():
    Z_all = ae_model.encoder(X_tensor.to(device)).cpu().numpy()  # (N, 32)

print(f'잠재 벡터 shape: {Z_all.shape} → (샘플, 잠재차원)')

# 잠재 벡터도 정규화 (MLP 매핑 학습 안정화)
z_scaler = StandardScaler()
Z_all_scaled = z_scaler.fit_transform(Z_all)


# ====================================================================
# [7. 역방향 매핑 MLP 정의 및 학습 (z → P1~P6)]
# ====================================================================
# 핵심: 잠재 벡터 z(32차원)로부터 설계변수 P1~P6(6차원)을 역추정
# 4,319차원(617×7) → 6차원 직접 매핑 대비 훨씬 안정적

class InverseMapper(nn.Module):
    """
    MLP 역매핑: 잠재 벡터 z (32차원) → P1~P6 (6차원)
    Dropout으로 과적합 방지, BatchNorm으로 학습 안정화
    """
    def __init__(self, latent_dim=32, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Linear(32, output_dim),
        )

    def forward(self, z):
        return self.net(z)

# -- 모델 생성 --
inv_mapper = InverseMapper(LATENT_DIM, 6).to(device)
print(f'\n[7] 역매핑 MLP 파라미터: {sum(p.numel() for p in inv_mapper.parameters()):,}개')

# -- 학습 설정 --
INV_EPOCHS     = 300
INV_BATCH_SIZE = 32
INV_LR         = 1e-3
INV_PATIENCE   = 30

# Train/Val 분리 (동일 인덱스 사용)
Z_tensor = torch.FloatTensor(Z_all_scaled)

inv_train_loader = DataLoader(
    TensorDataset(Z_tensor[train_idx], P_tensor[train_idx]),
    batch_size=INV_BATCH_SIZE, shuffle=True
)
inv_val_loader = DataLoader(
    TensorDataset(Z_tensor[val_idx], P_tensor[val_idx]),
    batch_size=INV_BATCH_SIZE, shuffle=False
)

optimizer_inv = optim.Adam(inv_mapper.parameters(), lr=INV_LR, weight_decay=1e-5)
scheduler_inv = optim.lr_scheduler.ReduceLROnPlateau(optimizer_inv, patience=15, factor=0.5)
criterion_inv = nn.MSELoss()

# -- 학습 루프 --
print(f'역매핑 MLP 학습 시작 (epochs={INV_EPOCHS})')

inv_train_losses = []
inv_val_losses = []
best_inv_val_loss = float('inf')
inv_patience_counter = 0

t_start = time.time()

for epoch in range(INV_EPOCHS):
    # -- Train --
    inv_mapper.train()
    epoch_loss = 0
    for batch_z, batch_p in inv_train_loader:
        batch_z, batch_p = batch_z.to(device), batch_p.to(device)
        p_pred = inv_mapper(batch_z)
        loss = criterion_inv(p_pred, batch_p)
        optimizer_inv.zero_grad()
        loss.backward()
        optimizer_inv.step()
        epoch_loss += loss.item() * len(batch_z)
    epoch_loss /= n_train
    inv_train_losses.append(epoch_loss)

    # -- Validation --
    inv_mapper.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_z, batch_p in inv_val_loader:
            batch_z, batch_p = batch_z.to(device), batch_p.to(device)
            p_pred = inv_mapper(batch_z)
            loss = criterion_inv(p_pred, batch_p)
            val_loss += loss.item() * len(batch_z)
    val_loss /= (N_SAMPLES - n_train)
    inv_val_losses.append(val_loss)
    scheduler_inv.step(val_loss)

    # Early Stopping
    if val_loss < best_inv_val_loss:
        best_inv_val_loss = val_loss
        inv_patience_counter = 0
        best_inv_state = inv_mapper.state_dict().copy()
    else:
        inv_patience_counter += 1

    if (epoch + 1) % 30 == 0:
        print(f'  Epoch {epoch+1:3d}/{INV_EPOCHS} | '
              f'Train: {epoch_loss:.6f} | Val: {val_loss:.6f} | '
              f'Patience: {inv_patience_counter}/{INV_PATIENCE}')

    if inv_patience_counter >= INV_PATIENCE:
        print(f'  → Early Stopping at epoch {epoch+1}')
        break

inv_mapper.load_state_dict(best_inv_state)
elapsed = time.time() - t_start
print(f'\n역매핑 MLP 학습 완료 ({elapsed:.1f}초)')
print(f'최종 Val Loss: {best_inv_val_loss:.6f}')


# ====================================================================
# [8. 역매핑 성능 검증]
# ====================================================================
# Validation 데이터에서 z → P1~P6 예측 정확도 확인

print('\n[8] 역매핑 성능 검증 (Validation Set)')

inv_mapper.eval()
with torch.no_grad():
    z_val = Z_tensor[val_idx].to(device)
    p_pred_val = inv_mapper(z_val).cpu().numpy()

# 역정규화하여 실제 P 스케일로 복원
p_pred_actual = param_scaler.inverse_transform(p_pred_val)
p_true_actual = param_scaler.inverse_transform(P_tensor[val_idx].numpy())

# 변수별 MAE 및 상대 오차율 출력
p_labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
print(f'{"변수":>5s} | {"MAE":>10s} | {"상대오차":>10s} | {"실제범위":>20s}')
print('-' * 55)
for i, p in enumerate(p_labels):
    mae = np.mean(np.abs(p_pred_actual[:, i] - p_true_actual[:, i]))
    mean_val = np.mean(np.abs(p_true_actual[:, i]))
    rel_err = mae / mean_val * 100  # 상대 오차(%)
    p_min, p_max = p_true_actual[:, i].min(), p_true_actual[:, i].max()
    print(f'{p:>5s} | {mae:10.4f} | {rel_err:8.2f}%  | [{p_min:.4f}, {p_max:.4f}]')

# Pred vs Actual 시각화
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Inverse Mapping Validation: Predicted vs Actual P1~P6', fontweight='bold')

for i, (ax, p) in enumerate(zip(axes.flat, p_labels)):
    ax.scatter(p_true_actual[:, i], p_pred_actual[:, i], s=15, alpha=0.6, c='steelblue')
    lims = [p_true_actual[:, i].min(), p_true_actual[:, i].max()]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel(f'Actual {p}')
    ax.set_ylabel(f'Predicted {p}')
    ax.set_title(p, fontweight='bold')

plt.tight_layout()
plt.show()


# ====================================================================
# [9. 유토피아 타겟 역설계 (최종 추론)]
# ====================================================================
# Step 3에서 생성된 유토피아 텐서를 Encoder → z → 역매핑 → P1~P6

print('\n[9] 유토피아 타겟 역설계 수행...')

# 유토피아 텐서 파일 탐색
utopia_pattern = os.path.join(TENSOR_DIR, 'Utopia_Target_Row_*.csv')
utopia_files = sorted(glob.glob(utopia_pattern))

if not utopia_files:
    print(f'[경고] {TENSOR_DIR}에 유토피아 텐서 파일이 없습니다.')
    print('Step 3를 먼저 실행하세요.')
else:
    results = []

    ae_model.eval()
    inv_mapper.eval()

    for fpath in utopia_files:
        fname = os.path.basename(fpath)
        match = re.search(r'Row_(\d+)\.csv', fname)
        row_id = int(match.group(1)) if match else -1

        # 유토피아 텐서 로드
        df_utopia = pd.read_csv(fpath)
        ts_utopia = df_utopia[CORE_7_CHANNELS].values  # (617, 7)

        # 채널별 정규화 (학습 시와 동일한 scaler 사용)
        ts_scaled = np.zeros_like(ts_utopia)
        for ch in range(N_CHANNELS):
            ts_scaled[:, ch] = channel_scalers[ch].transform(ts_utopia[:, ch].reshape(-1, 1)).flatten()

        # (1, 7, 617) 텐서로 변환
        x_input = torch.FloatTensor(ts_scaled).unsqueeze(0).permute(0, 2, 1).to(device)

        with torch.no_grad():
            # Encoder → 잠재 벡터
            z_target = ae_model.encoder(x_input)

            # z 정규화 (학습 시와 동일한 scaler)
            z_target_scaled = z_scaler.transform(z_target.cpu().numpy())
            z_target_tensor = torch.FloatTensor(z_target_scaled).to(device)

            # 역매핑 → P1~P6 (정규화된 상태)
            p_pred_scaled = inv_mapper(z_target_tensor).cpu().numpy()

        # 역정규화하여 실제 P 스케일로 복원
        p_pred = param_scaler.inverse_transform(p_pred_scaled)[0]

        result = {'Row_ID': row_id}
        for i, p in enumerate(p_labels):
            result[p] = p_pred[i]
        results.append(result)

        print(f'  Row_{row_id}: P1={p_pred[0]:.4f}, P2={p_pred[1]:.4f}, P3={p_pred[2]:.4f}, '
              f'P4={p_pred[3]:.4f}, P5={p_pred[4]:.4f}, P6={p_pred[5]:.4f}')

    # 결과 저장
    df_results = pd.DataFrame(results)
    output_path = os.path.join(BASE_DIR, 'Inverse_Design_Results.csv')
    df_results.to_csv(output_path, index=False)

    print(f'\n=== 역설계 완료! ===')
    print(f'결과: {len(results)}개 P1~P6 초안')
    print(f'저장: {output_path}')
    print(f'→ 이 초안은 [Step 5: GA 미세조정]의 시작점(±10% 탐색 범위)으로 사용됩니다.')

    display(df_results) if 'display' in dir() else print(df_results)


# ====================================================================
# [10. 역설계 결과 물리적 범위 검증]
# ====================================================================
# 출력된 P1~P6가 마스터 DOE의 실제 범위 내에 있는지 확인
# 범위를 벗어나면 Step 5에서 클리핑 또는 페널티 부여 필요

print('\n[10] 역설계 P1~P6 범위 검증')
print(f'{"변수":>5s} | {"DOE Min":>10s} | {"DOE Max":>10s} | {"예측 Min":>10s} | {"예측 Max":>10s} | 범위이탈')
print('-' * 70)

for i, p in enumerate(p_labels):
    doe_min = df_master[p].min()
    doe_max = df_master[p].max()
    pred_min = df_results[p].min()
    pred_max = df_results[p].max()
    out_low = (df_results[p] < doe_min).sum()
    out_high = (df_results[p] > doe_max).sum()
    flag = f' << {out_low+out_high}건 이탈!' if (out_low + out_high) > 0 else ''
    print(f'{p:>5s} | {doe_min:10.4f} | {doe_max:10.4f} | {pred_min:10.4f} | {pred_max:10.4f} |{flag}')

print('\n→ 범위 이탈 시 Step 5 GA에서 바운더리 클리핑 적용 예정')
print('→ 이 P1~P6 초안을 중심으로 ±10% 범위에서 NSGA-II 미세조정 진행')
