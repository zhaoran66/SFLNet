# -*- coding: utf-8 -*-
"""
Direction 3: Test-Time Retrieval
- Build ego_feat library from training set
- At inference, use exo_global to retrieve top-k nearest ego_feats
- Use the mean of retrieved feats as endpoint
- Evaluate: MPJPE vs k to find optimal retrieval strategy
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from losses import MPJPE, PCK
from dataset import build_dataloader
from model import SPLLatentKeypoint


def compute_pa_epe(pred, gt):
    if pred.dim() == 4:
        B, T, J, C = pred.shape
        pred = pred.reshape(B * T, J, C)
        gt = gt.reshape(B * T, J, C)
    pred_center = pred.mean(dim=1, keepdim=True)
    gt_center = gt.mean(dim=1, keepdim=True)
    epe = torch.norm(pred - gt, p=2, dim=-1)
    return epe.mean(dim=-1) * 100


def compute_pa_auc(errors, max_threshold=10.0):
    thresholds = torch.linspace(0, max_threshold, 100)
    pck_curve = []
    for thresh in thresholds:
        pck_thresh = (errors < thresh).float().mean()
        pck_curve.append(pck_thresh)
    pck_curve = torch.tensor(pck_curve)
    auc = torch.trapz(pck_curve, thresholds) / max_threshold * 100
    return auc.item()


def wrist_center(kp):
    return kp - kp[..., 0:1, :]


class EgoFeatRetriever:
    """Build and query ego feature library from training set"""
    
    def __init__(self, model, train_loader, device):
        print("Building ego feature library from training set...")
        self.model = model
        self.device = device
        model.eval()
        
        all_exo = []
        all_ego_feat = []
        
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                exo_video = batch['exo_video'].to(device)
                ego_kp = batch['ego_keypoints'].to(device)
                
                # Get exo_global features
                spatial_feat, exo_global = model.feature_extractor(exo_video)
                
                # Get ego_feat (ground truth)
                ego_feat = model.feature_extractor.encode_ego_keypoints(ego_kp)
                
                # Pool across sequence
                exo_pooled = exo_global.mean(dim=1)  # [B, 256]
                ego_pooled = ego_feat.mean(dim=1)  # [B, 256]
                
                all_exo.append(exo_pooled.cpu())
                all_ego_feat.append(ego_pooled.cpu())
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1} batches ({len(all_exo)*4} samples)")
        
        self.exo_library = torch.cat(all_exo, dim=0)  # [N_train, 256]
        self.ego_library = torch.cat(all_ego_feat, dim=0)  # [N_train, 256]
        self.N = self.exo_library.shape[0]
        print(f"Library built: {self.N} samples, dim={self.exo_library.shape[1]}")
        
        # Move to same device as model
        self.exo_library = self.exo_library.to(device)
        self.ego_library = self.ego_library.to(device)
        
        # Normalize for cosine similarity
        self.exo_library_norm = torch.nn.functional.normalize(self.exo_library, dim=1)
    
    def retrieve(self, query_exo, k=5, metric='cosine'):
        """
        Retrieve top-k ego feats for each query exo_global.
        
        Args:
            query_exo: [B, 256] pooled exo features
            k: number of neighbors to retrieve
            metric: 'cosine' or 'euclidean'
        
        Returns:
            retrieved_ego: [B, 256] mean of top-k retrieved ego feats
            similarities: [B, k] retrieval similarities
        """
        if metric == 'cosine':
            query_norm = torch.nn.functional.normalize(query_exo, dim=1)  # [B, 256]
            sim = query_norm @ self.exo_library_norm.T  # [B, N]
            topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)  # [B, k]
            
        elif metric == 'euclidean':
            # query_exo: [B, 256], library: [N, 256]
            # dist[b, n] = ||query[b] - lib[n]||
            dist = torch.cdist(query_exo, self.exo_library, p=2)  # [B, N]
            topk_sim_neg, topk_idx = torch.topk(-dist, k=k, dim=1)  # [B, k]
            topk_sim = -topk_sim_neg  # convert back to similarity
        
        # Get top-k ego feats
        retrieved_ego = []
        for b in range(query_exo.shape[0]):
            topk_ego = self.ego_library[topk_idx[b]]  # [k, 256]
            retrieved_ego.append(topk_ego.mean(dim=0))  # [256]
        
        retrieved_ego = torch.stack(retrieved_ego, dim=0)  # [B, 256]
        
        return retrieved_ego, topk_sim
    
    def retrieve_sequence_level(self, query_exo_seq, k=5, metric='cosine'):
        """
        Retrieve using sequence-level features (not pooled).
        Query: [B, T, 256], Library: [N, 256] (pooled across sequences)
        
        We compute similarity for each timestep and aggregate.
        """
        B, T, H = query_exo_seq.shape
        
        # Average pool across time for query
        query_pooled = query_exo_seq.mean(dim=1)  # [B, 256]
        return self.retrieve(query_pooled, k=k, metric=metric)


def run_retrieval_eval(model, retriever, val_loader, device, k_values=[1, 3, 5, 10, 20], 
                       metrics_list=['cosine'], max_batches=50):
    """Evaluate retrieval-based endpoint prediction"""
    model.eval()
    mpjpe_metric = MPJPE()
    
    results = {k: {'mpjpe': [], 'pa_epe': []} for k in k_values}
    
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= max_batches:
                break
            
            exo_video = batch['exo_video'].to(device)
            gt_kp = batch['ego_keypoints'].to(device)
            
            # Get exo_global features
            spatial_feat, exo_global = model.feature_extractor(exo_video)
            
            # For each k value
            for k in k_values:
                # Retrieve ego endpoints
                retrieved_ego, _ = retriever.retrieve(exo_global.mean(dim=1), k=k)
                
                # Expand to sequence length
                ego_endpoint = retrieved_ego.unsqueeze(1).expand(-1, 16, -1)  # [B, 16, 256]
                
                # Run interpolation with retrieved endpoint
                full_sequence = model.geodesic_interpolator(exo_global, ego_endpoint)
                
                # Encode
                num_total_steps = full_sequence.shape[2]
                full_flat = full_sequence.view(exo_global.shape[0], -1, model.hidden_dim)
                full_flat = model.seq_norm(full_flat)
                full_flat = full_flat + model.pos_encoding[:full_flat.shape[1], :].unsqueeze(0)
                encoded = model.sequence_encoder(full_flat)
                encoded = encoded.view(exo_global.shape[0], -1, num_total_steps, model.hidden_dim)
                
                # Decode
                all_kps = []
                for step_idx in range(num_total_steps):
                    start_idx = max(0, step_idx - 1)
                    end_idx = min(num_total_steps, step_idx + 2)
                    context = encoded[:, :, start_idx:end_idx, :].flatten(2, 3)
                    step_feat = encoded[:, :, step_idx, :]
                    dec_input = torch.cat([step_feat, context], dim=-1)[:, :, :model.hidden_dim]
                    kps = model.keypoint_decoder(dec_input, spatial_feat)
                    all_kps.append(kps)
                
                pred_kp = all_kps[-1]
                pred_centered = wrist_center(pred_kp)
                gt_centered = wrist_center(gt_kp)
                
                results[k]['mpjpe'].append(mpjpe_metric(pred_centered, gt_centered).item())
                results[k]['pa_epe'].extend(compute_pa_epe(pred_centered, gt_centered).cpu().numpy())
            
            count += 1
    
    return results


def main():
    cfg = load_config('config.yaml')
    device = torch.device(cfg.training.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = SPLLatentKeypoint(cfg).to(device)
    ckpt = torch.load('check/best_model.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')}")

    # Build retrievers
    train_loader = build_dataloader(cfg, split='train')
    val_loader = build_dataloader(cfg, split='val')
    
    # Build ego feat library
    retriever = EgoFeatRetriever(model, train_loader, device)

    # Run retrieval evaluation
    print("\n" + "=" * 70)
    print("RETRIEVAL-BASED ENDPOINT PREDICTION RESULTS")
    print("=" * 70)
    
    k_values = [1, 3, 5, 10, 20, 50]
    results = run_retrieval_eval(model, retriever, val_loader, device, k_values=k_values, max_batches=50)
    
    print(f"\n{'k':<8} {'MPJPE (mm)':<15} {'PA-EPE (cm)':<15} {'PA-AUC (%)':<12}")
    print("-" * 50)
    
    best_k = None
    best_mpjpe = float('inf')
    
    for k in k_values:
        mpjpe_mean = np.mean(results[k]['mpjpe'])
        pa_epe_mean = np.mean(results[k]['pa_epe'])
        pa_auc = compute_pa_auc(torch.tensor(results[k]['pa_epe']))
        
        print(f"k={k:<5} {mpjpe_mean:>10.2f} mm    {pa_epe_mean:>10.2f} cm    {pa_auc:>10.2f} %")
        
        if mpjpe_mean < best_mpjpe:
            best_mpjpe = mpjpe_mean
            best_k = k
    
    print("-" * 50)
    print(f"\nBaseline comparison:")
    print(f"  Full SPL (learnable_ego_anchor):  69.1 mm")
    print(f"  Mean ego_feat (oracle):            19.2 mm")
    print(f"  Retrieval (k={best_k}, best):     {best_mpjpe:.1f} mm")
    print(f"\n  Gap to oracle: {best_mpjpe - 19.2:.1f} mm ({(best_mpjpe/19.2 - 1)*100:.1f}%)")
    
    if best_mpjpe < 30:
        print("\n[+] Retrieval achieves good results!")
        print("    => Direction 1 (Endpoint Predictor) is worth pursuing")
    elif best_mpjpe < 50:
        print("\n[~] Retrieval shows moderate potential")
        print("    => Consider Direction 2 (probabilistic modeling)")
    else:
        print("\n[!] Retrieval gap is large")
        print("    => exo->ego mapping may lack sufficient information")
        print("    => Consider using partial ego information (1 frame)")


if __name__ == '__main__':
    main()
