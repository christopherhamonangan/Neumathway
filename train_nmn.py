import os
import sys
import glob
import math
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel
import numpy as np

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# =====================================================================
# 1. SHARDED STREAMING DATASET LOADER (MAX_NODES = 32)
# =====================================================================
class ShardedMathDataset(Dataset):
    def __init__(self, shard_dir):
        self.shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.pt")))
        if len(self.shard_files) == 0:
            raise FileNotFoundError(f"[-] Tidak ada file shard ditemukan di '{shard_dir}'!")

        self.shard_sizes = []
        for f in self.shard_files:
            d = torch.load(f, weights_only=False)
            self.shard_sizes.append(len(d['ops']))

        self.cumulative_sizes = np.cumsum(self.shard_sizes)
        self.total_samples = int(self.cumulative_sizes[-1])
        self.current_shard_idx = -1
        self.current_data = None

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        shard_idx = int(np.searchsorted(self.cumulative_sizes, idx, side='right'))
        internal_idx = idx if shard_idx == 0 else idx - int(self.cumulative_sizes[shard_idx - 1])

        if shard_idx != self.current_shard_idx:
            self.current_shard_idx = shard_idx
            self.current_data = torch.load(self.shard_files[shard_idx], weights_only=False)

        return (
            self.current_data['feat'][internal_idx],
            self.current_data['adj'][internal_idx],
            self.current_data['mask'][internal_idx],
            self.current_data['ops'][internal_idx],
            self.current_data['raw'][internal_idx],
            self.current_data['norm'][internal_idx]
        )

def collate_fn(batch):
    feat = torch.stack([item[0] for item in batch])
    adj = torch.stack([item[1] for item in batch])
    mask = torch.stack([item[2] for item in batch])
    ops = torch.stack([item[3] for item in batch])
    raw = torch.stack([item[4] for item in batch])
    norm = torch.stack([item[5] for item in batch])
    return feat, adj, mask, ops, raw, norm

# =====================================================================
# 2. RELATION-AWARE RESIDUAL GAT (FP16 SAFE)
# =====================================================================
class RelationAwareGAT(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super(RelationAwareGAT, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(num_heads, self.head_dim, 1))
        self.a_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        B, N, D = h.shape
        h_proj = self.W(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_src = torch.matmul(h_proj, self.a_src)
        attn_dst = torch.matmul(h_proj, self.a_dst)
        attn = self.leaky_relu(attn_src + attn_dst.transpose(-2, -1))

        mask = adj.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attn = attn.masked_fill(mask == 0, -1e4)
        attn_weights = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn_weights, h_proj).transpose(1, 2).contiguous().view(B, N, D)
        return self.norm(self.relu(out) + h)

# =====================================================================
# 3. SOFT DECISION-TREE ROUTER
# =====================================================================
class SoftDecisionRouter(nn.Module):
    def __init__(self, hidden_dim=256):
        super(SoftDecisionRouter, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, state):
        return self.gate(state)

# =====================================================================
# 4. NEUMATHWORK ARCHITECTURE V4.0 (16 UNIFIED CLASSES)
# =====================================================================
class NeumathworkModel(nn.Module):
    def __init__(self, in_features=8, hidden_dim=256, num_heads=4, num_classes=16):
        super(NeumathworkModel, self).__init__()

        # [A] Expression Graph Encoder (Membaca Graf hingga 32 Node)
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.gat1 = RelationAwareGAT(hidden_dim, num_heads)
        self.gat2 = RelationAwareGAT(hidden_dim, num_heads)
        self.gat3 = RelationAwareGAT(hidden_dim, num_heads)
        self.gat4 = RelationAwareGAT(hidden_dim, num_heads)

        # [B] Soft Decision-Tree Router
        self.router = SoftDecisionRouter(hidden_dim)

        # [C] Recurrent Reasoning Controller
        self.controller = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # [D] NOP Head (Dukungan 16 Kelas Aksi Matematika Lengkap)
        self.temperature = nn.Parameter(torch.ones(1) * 0.886)
        self.nop_mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.nop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

        # [E] NMC Head (Estimasi Ruang Laten)
        self.nmc_mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.nmc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, node_features, adj_matrix, node_mask):
        h = self.in_proj(node_features)
        h = self.gat1(h, adj_matrix)
        h = self.gat2(h, adj_matrix)
        h = self.gat3(h, adj_matrix)
        h = self.gat4(h, adj_matrix)

        mask_exp = node_mask.unsqueeze(-1).float()
        h_masked = h * mask_exp
        graph_summary = (h_masked.sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1.0)).unsqueeze(1)

        route_weight = self.router(graph_summary.squeeze(1))
        gru_out, _ = self.controller(graph_summary)

        attn_mask = ~node_mask
        cross_out, _ = self.cross_attn(query=gru_out, key=h, value=h, key_padding_mask=attn_mask)
        reasoned_state = gru_out + cross_out

        # NOP Path
        nop_attn, _ = self.nop_mha(reasoned_state, reasoned_state, reasoned_state)
        raw_logits = self.nop_head(nop_attn.squeeze(1))
        temp = torch.clamp(self.temperature, min=0.1, max=2.0)
        calibrated_logits = raw_logits / temp

        # NMC Path
        nmc_attn, _ = self.nmc_mha(reasoned_state, reasoned_state, reasoned_state)
        norm_val_pred = self.nmc_head(nmc_attn.squeeze(1)).squeeze(-1)

        return calibrated_logits, norm_val_pred, route_weight

# =====================================================================
# 5. ADAPTIVE LOSS & UTILITIES
# =====================================================================
class AdaptiveMultiTaskLoss(nn.Module):
    def __init__(self):
        super(AdaptiveMultiTaskLoss, self).__init__()
        self.log_var_nop = nn.Parameter(torch.zeros(1))
        self.log_var_nmc = nn.Parameter(torch.zeros(1))
        self.criterion_op = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.criterion_val = nn.SmoothL1Loss()

    def forward(self, op_logits, norm_val_pred, target_ops, norm_vals):
        loss_nop = self.criterion_op(op_logits, target_ops)
        loss_nmc = self.criterion_val(norm_val_pred, norm_vals)
        precision_nop = torch.exp(-self.log_var_nop)
        precision_nmc = torch.exp(-self.log_var_nmc)
        total_loss = (precision_nop * loss_nop + self.log_var_nop) + \
                     (precision_nmc * loss_nmc + self.log_var_nmc)
        return total_loss, loss_nop, loss_nmc

def apply_gradient_centralization(model):
    for p in model.parameters():
        if p.grad is not None and p.dim() > 1:
            p.grad.data.add_(-p.grad.data.mean(dim=tuple(range(1, p.dim())), keepdim=True))

def get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

# =====================================================================
# 6. TRANSFER LEARNING & TRAINING PIPELINE V4.0
# =====================================================================
def train_nmn_v4():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "="*70)
    print("      NEUMATHWORK v4.0 - ADVANCED TRANSFER LEARNING ENGINE      ")
    print("="*70)
    print(f"[*] Perangkat Akselerasi : {device}")

    train_dir = 'data/train_shards'
    print(f"[*] Menginisialisasi Dataset Sharded dari '{train_dir}'...")
    train_dataset = ShardedMathDataset(train_dir)

    batch_size = 256
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Inisialisasi Model v4 dengan 16 Kelas Aksi
    model = NeumathworkModel(in_features=8, hidden_dim=256, num_heads=4, num_classes=16).to(device)
    loss_module = AdaptiveMultiTaskLoss().to(device)

    # PERISAI ANTI-LUPA: Muat Bobot Pintar dari Fondasi v3 (Warm-Start)
    old_ckpt_path = 'v3/nmn_checkpoint.pth'
    if os.path.exists(old_ckpt_path):
        print(f"[!] MENDETEKSI MODEL FONDASI V3 DI '{old_ckpt_path}'. Mewarisi kepintaran lama...")
        old_state = torch.load(old_ckpt_path, map_location=device, weights_only=False)
        model_dict = model.state_dict()
        
        # Salin semua bobot GAT, Router, Controller, dan NMC (Kecuali layer linear terakhir NOP yang bertambah dari 8 ke 16)
        pretrained_dict = {k: v for k, v in old_state.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"[+] Berhasil mentransfer {len(pretrained_dict)} layer bobot pintar ke arsitektur v4!")
    else:
        print("[*] Melatih dari awal (Cold-Start)...")

    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4}, # Learning rate halus untuk fine-tuning
        {'params': loss_module.parameters(), 'lr': 1e-3}
    ])

    epochs = 6  # 6 Epoch sudah sangat cukup berkat transfer learning
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * 1
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_steps, total_steps)

    swa_model = AveragedModel(model)
    swa_start_epoch = 4
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f"[*] Memulai Fine-Tuning v4.0 ({len(train_dataset):,} sampel, {epochs} Epochs)...")
    for epoch in range(epochs):
        model.train()
        total_loss, correct_op, total_samples = 0.0, 0, 0

        for feat, adj, mask, target_ops, _, norm_vals in train_loader:
            feat = feat.to(device, non_blocking=True)
            adj = adj.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target_ops = target_ops.to(device, non_blocking=True)
            norm_vals = norm_vals.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                op_logits, norm_val_pred, route_weight = model(feat, adj, mask)
                loss, loss_nop, loss_nmc = loss_module(op_logits, norm_val_pred, target_ops, norm_vals)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            apply_gradient_centralization(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item() * len(target_ops)
            preds = torch.argmax(op_logits, dim=-1)
            correct_op += (preds == target_ops).sum().item()
            total_samples += len(target_ops)

        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)

        acc = (correct_op / total_samples) * 100
        avg_loss = total_loss / total_samples
        current_lr = scheduler.get_last_lr()[0]
        temp_val = model.temperature.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - NOP Acc: {acc:.2f}% (T: {temp_val:.3f}, LR: {current_lr:.6f})")

    torch.save(swa_model.module.state_dict(), 'nmn_checkpoint.pth')
    print("\n=======================================================")
    print("[+] MODEL V4.0 SELESAI DILATIH DENGAN SUKSES!")
    print("[+] Model disimpan di 'nmn_checkpoint.pth'.")
    print("=======================================================\n")

if __name__ == '__main__':
    train_nmn_v4()