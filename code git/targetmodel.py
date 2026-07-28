import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math
import numpy as np

"""
** source-free OPEN-set domain adaptation **
- Source model trains on labeled source data that has k known classes, then the source data are discarded (hence, source-free).
- Target data are unlabeled and contain unknown classes that didn't exist in source data.
- Target model consists of two branches: top branch (source model with source classifier),
  bottom branch (source model (without source classifier) with residual and style module).
- Clustering (2*k clusters) is done on bottom branch.
- The top and bottom branches are parallel: every unlabeled target image is fanned-out once to both branches.
- The final goal: Adapt source model on target data so it can accurately classify known target data (k classes) and detect unknown target data using    score-based OSR (put them in k+1th class). 
- THE TOP BRANCH THAT WAS TRAINED ON SOURCE DATA SHOULD NOW LEARN FROM THE BOTTOM BRANCH THAT HAS F_T CLUSTERS.
"""

EPS = 1e-12
TEMP = 0.05 # q_ij temperature. Smaller --> sharper q

# =============================================================================
# UTILITY FUNCTIONS FOR DISTANCE CALCULATIONS
# =============================================================================
def squared_cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1).unsqueeze(0)
    xy = x @ y.t()
    d2 = x2 + y2 - 2.0 * xy
    return d2.clamp_min(0.0)
def squared_cosine_cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: # normalizing the input tensors
    x = F.normalize(x, dim=1, eps=1e-12)
    y = F.normalize(y, dim=1, eps=1e-12)
    cos = x @ y.t()
    d = (1.0 - cos).clamp_min(0.0)
    return d * d

# ============================================================
# CLUSTER ASSIGNMENT AND ENTROPY COMPUTATION
# ============================================================
# This function computes the probability that each target
# feature belongs to every cluster center.
#
# Instead of assigning each sample to only one cluster,
# a temperature-scaled softmax over cosine distances is used.
# These probabilities are later used for entropy estimation,
# confidence weighting, and open-set recognition.
# ============================================================
def compute_q_ij(f: torch.Tensor, centers: torch.Tensor, temp: float = TEMP) -> torch.Tensor: # assign each feature to its closest cluster center
    """
    Compute soft cluster assignment probabilities (q_ij).
    
    Instead of hard assignment to one cluster, uses temperature-scaled softmax
    over cosine distances. These probabilities are used for:
    - Entropy estimation (uncertainty)
    - Confidence weighting
    - Open-set recognition
    
    Args:
        f: Feature vectors of shape (N, D)
        centers: Cluster centers of shape (C, D)
        temp: Temperature parameter for softmax scaling
    
    Returns:
        Tensor of shape (N, C) containing assignment probabilities
    """
    f = F.normalize(f, dim=1, eps=1e-12)
    centers = F.normalize(centers, dim=1, eps=1e-12)
    d2 = squared_cosine_cdist(f, centers)
    temp = max(float(temp), EPS)
    logits = -d2 / temp
    return F.softmax(logits, dim=1)
    
# Compute cosine distance between the features and centers, assign each feature to the closest center
def assign_clusters(f: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    Hard cluster assignment using nearest neighbor.
    
    Args:
        f: Feature vectors of shape (N, D)
        centers: Cluster centers of shape (C, D)
    
    Returns:
        Tensor of shape (N,) containing cluster indices
    """
    f_n = F.normalize(f, dim=1, eps=1e-12)
    c_n = F.normalize(centers, dim=1, eps=1e-12)
    d2 = squared_cosine_cdist(f_n, c_n)
    return d2.argmin(dim=1)
    
# compute the entropy of the clusters
def compute_cluster_entropy(
    features: torch.Tensor,
    centers: torch.Tensor,
    cluster_labels: torch.Tensor,
    temp: float = TEMP
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-cluster entropy using soft assignments. 
    
    High entropy = cluster is spread out/ambiguous
    Low entropy = cluster is compact/distinct
    
    Args:
        features: Feature vectors of shape (N, D)
        centers: Cluster centers of shape (C, D)
        cluster_labels: Hard cluster assignments of shape (N,)
        temp: Temperature for soft assignment
    
    Returns:
        cluster_entropy: Per-cluster entropy of shape (C,)
        counts: Number of samples per cluster of shape (C,)
    """
    device = features.device
    C = centers.size(0)

    # Soft assignments    
    q_ij = compute_q_ij(features, centers, temp=temp)

    # Sample entropy: uncertainty of each sample's cluster assignment
    H_i = -(q_ij * torch.log(q_ij + EPS)).sum(dim=1)

    # Aggregate per cluster
    counts = torch.bincount(cluster_labels, minlength=C).float().to(device)
    entropy_sum = torch.bincount(cluster_labels, weights=H_i, minlength=C).float().to(device)

    # Initialize with maximum entropy
    total = features.size(0)
    min_count = 1 
    maxH = math.log(C + EPS)
    cluster_entropy = torch.full((C,), maxH, device=device)

    # Compute average entropy for valid clusters
    valid = counts >= min_count
    if valid.any():
        cluster_entropy[valid] = entropy_sum[valid] / counts[valid].clamp_min(1.0)
    cluster_entropy = cluster_entropy.clamp(0.0, maxH)
    return cluster_entropy, counts
    
@torch.no_grad()
def compute_cluster_entropy_from_loader(model, target_loader, device, temp: float = TEMP) -> torch.Tensor: # computes a per-cluster entropy ********
    """
    Compute per-cluster entropy by running through the entire target dataset.
    
    This is a global, stable estimate of cluster entropy that is used to
    calibrate the open-set threshold.
    
    Args:
        model: TargetModel instance
        target_loader: DataLoader for target data
        device: Device to run on
        temp: Temperature for soft assignment
    
    Returns:
        cluster_entropy: Per-cluster entropy of shape (num_clusters,)
    """
    model.eval()

    # Collect all target features
    feats = []
    for batch in target_loader:
        x = batch[0].to(device)
        out = model.forward(x, return_all=True)
        f_t = out["f_t"]
        feats.append(f_t.detach().cpu())
    feats_all = torch.cat(feats, dim=0).to(device)
    centers = model.cluster_centers.to(device)

    # Assign clusters and compute entropy
    cluster_labels = assign_clusters(feats_all, centers)
    q_ij = compute_q_ij(feats_all, centers, temp=temp)

    # Debug output
    qmax = q_ij.max(dim=1).values
    print("qmax mean/min/max:", qmax.mean().item(), qmax.min().item(), qmax.max().item())
    cluster_entropy, counts = compute_cluster_entropy(feats_all, centers, cluster_labels, temp=temp) 

    # Save for later use
    model.saved_cluster_entropies.copy_(cluster_entropy)
    model.saved_cluster_counts.copy_(counts)
    print("cluster counts:", counts.tolist())
    return cluster_entropy

# =============================================================================
# CALIBRATION FUNCTIONS FOR OPEN-SET RECOGNITION
# =============================================================================
@torch.no_grad()
def calibrate_entropy_threshold(model, loader, device, q=0.75, return_all=False):
    """
    Calibrate entropy threshold using quantile of target data entropies.
    
    This threshold separates known (low entropy) from unknown (high entropy).
    
    Args:
        model: TargetModel instance
        loader: DataLoader for calibration data
        device: Device to run on
        q: Quantile to use as threshold (default: 0.75)
        return_all: If True, also return all entropies
    
    Returns:
        tau: Entropy threshold value
        entropies: (Optional) All entropies computed
    """
    model.eval()
    entropies = []
    
    for batch in loader:
        x = batch[0].to(device)
        out = model.forward(x, return_all=True)
        probs = F.softmax(out["logits_t"], dim=1)
        entropy = -(probs * torch.log(probs + EPS)).sum(dim=1)
        entropies.append(entropy.detach().cpu())
        
    entropies = torch.cat(entropies, dim=0)
    tau = torch.quantile(entropies, q).item()
    
    if return_all:
        return tau, entropies
    return tau
    
def otsu_threshold(values: torch.Tensor, num_bins: int = 50) -> float:
    """
    1D Otsu's method: finds the threshold that maximizes between-class
    variance for a bimodal distribution.
    
    Unlike quantile-based thresholding, Otsu discovers the natural split
    in the data distribution without assuming a fixed known/unknown ratio.
    
    Args:
        values: Tensor of values to threshold
        num_bins: Number of histogram bins
    
    Returns:
        Threshold that maximizes between-class variance
    """
    v = values.detach().float().flatten()
    vmin, vmax = v.min().item(), v.max().item()
    
    if vmax - vmin < 1e-8:
        return float(v.mean().item())

    # Build histogram
    hist = torch.histc(v, bins=num_bins, min=vmin, max=vmax)
    edges = torch.linspace(vmin, vmax, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    # Otsu's method: maximize between-class variance
    total = hist.sum().item()
    sum_all = (hist * centers).sum().item()

    wB, sumB = 0.0, 0.0
    best_var, best_thresh = -1.0, centers[0].item()
    
    for i in range(num_bins):
        wB += hist[i].item()
        if wB == 0:
            continue
        wF = total - wB
        if wF <= 0:
            break
        sumB += (hist[i] * centers[i]).item()
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = centers[i].item()
            
    return best_thresh


@torch.no_grad()
def calibrate_cluster_entropy_threshold(model, target_loader, device, temp: float = TEMP):
    """
    Data-driven calibration of cluster entropy threshold using Otsu's method.
    
    This replaces fixed-quantile calibration by finding the natural split
    in the blended (sample + population) entropy distribution.
    Computes the SAME blended per-sample entropy used in forward() (sample + population
    level), across the WHOLE loader — not just the num_clusters-sized
    population array — then finds a natural split via Otsu instead of
    assuming a fixed known/unknown ratio.
    
    Args:
        model: TargetModel instance
        target_loader: DataLoader for target data
        device: Device to run on
        temp: Temperature for soft assignment
    
    Returns:
        tau: Calibrated threshold
        all_entropy_n: All normalized entropies computed
    """
    model.eval()
    maxH = math.log(model.num_clusters + EPS)
    all_entropy_n = []
    
    for batch in target_loader:
        x = batch[0].to(device)
        out = model.forward(x, return_all=True)
        f_t = out["f_t"]

        # Sample-level entropy (per-sample uncertainty) ****************************************************
        q_ij = compute_q_ij(f_t, model.cluster_centers, temp=temp)
        sample_entropy = -(q_ij * torch.log(q_ij + EPS)).sum(dim=1)
        sample_entropy_n = (sample_entropy / maxH).clamp(0.0, 1.0)

        # Population-level entropy (per-cluster uncertainty) ***************************************************************
        cluster_labels = assign_clusters(f_t, model.cluster_centers)
        pop_entropy = torch.nan_to_num(
            model.saved_cluster_entropies, nan=maxH, posinf=maxH, neginf=0.0
        )[cluster_labels]
        pop_entropy_n = (pop_entropy / maxH).clamp(0.0, 1.0)

        # Blend both estimates
        blended = 0.5 * sample_entropy_n + 0.5 * pop_entropy_n
        all_entropy_n.append(blended.detach().cpu())

    all_entropy_n = torch.cat(all_entropy_n, dim=0)
    tau = otsu_threshold(all_entropy_n)
    model.cluster_entropy_threshold.fill_(float(tau))
    
    return tau, all_entropy_n

# =============================================================================
# K-MEANS CLUSTERING IMPLEMENTATION
# =============================================================================
# initializes cluster centers for k-means clustering by selecting points that are farthest apart 
def kmeans_init(features: torch.Tensor, num_clusters: int, device: torch.device) -> torch.Tensor:
    """
    Initialize k-means centers using farthest point sampling (k-means++).
    
    Args:
        features: Feature vectors of shape (N, D)
        num_clusters: Number of clusters to initialize
        device: Device to place centers on
    
    Returns:
        centers: Initial cluster centers of shape (num_clusters, D)
    """
    N = features.shape[0]
    centers = torch.zeros(num_clusters, features.shape[1], device=device)

    # Pick first center randomly
    centers[0] = features[torch.randint(0, N, (1,), device=device)].squeeze(0)

    # Pick remaining centers using k-means++ algorithm
    for c in range(1, num_clusters):
        dist_sq = squared_cosine_cdist(features, centers[:c])
        min_dist_sq = dist_sq.min(dim=1)[0]
        total = min_dist_sq.sum()
        prob_dist = min_dist_sq / total if total > 0 else torch.ones(N, device=device) / N
        new_center_idx = torch.multinomial(prob_dist, 1)
        centers[c] = features[new_center_idx].squeeze(0)
        
    return centers
    
# performs k-means clustering on the features, updates the cluster centers, assigns each point to a cluster
def run_kmeans(features: torch.Tensor, num_clusters: int, num_iters: int = 20):
    """
    Perform k-means clustering with handling for empty clusters.
    
    Args:
        features: Feature vectors of shape (N, D)
        num_clusters: Number of clusters
        num_iters: Number of iterations to run
    
    Returns:
        centers: Final cluster centers of shape (num_clusters, D)
        labels: Cluster assignments of shape (N,)
    """
    N, D = features.shape
    device = features.device
    features = F.normalize(features, dim=1)

    # Initialize centers
    centers = kmeans_init(features, num_clusters, device)
    
    for _ in range(num_iters):
        # Assign samples to nearest center
        d2 = squared_cosine_cdist(features, centers)
        labels = d2.argmin(dim=1)

        # Handle empty clusters
        counts = torch.bincount(labels, minlength=num_clusters)
        empty = (counts == 0).nonzero(as_tuple=True)[0]
        
        if empty.numel() > 0:
            # Re-initialize empty clusters with farthest points
            assigned_d2 = d2[torch.arange(N, device=device), labels]
            farthest = torch.argsort(assigned_d2, descending=True)
            used = set()
            for k in empty.tolist():
                for idx in farthest.tolist():
                    if idx in used:
                        continue
                    used.add(idx)
                    centers[k] = features[idx]
                    labels[idx] = k
                    break

        # Update centers as mean of assigned points
        for k in range(num_clusters):
            mask = (labels == k)
            if mask.any():
                centers[k] = features[mask].mean(dim=0)
            else:
                centers[k] = features[torch.randint(0, N, (1,), device=device)].squeeze(0)

    # Final assignment
    labels = squared_cosine_cdist(features, centers).argmin(dim=1)
    return centers, labels

# =============================================================================
# STYLE AND RESIDUAL BLOCKS FOR BOTTOM BRANCH
# =============================================================================
# Style , Residual blocks
class StyleBlock(nn.Module):
    """
    Style transfer block for cross-sample style swapping.
    
    Performs style transfer by matching mean and standard deviation
    of feature statistics between content and style features.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, z: torch.Tensor, z_prime: torch.Tensor) -> torch.Tensor:
        """
        Apply style from z_prime to content from z.
        
        Args:
            z: Content features (shape: B x C x H x W or B x C)
            z_prime: Style features (same shape as z)
        
        Returns:
            Styled features with content from z and style from z_prime
        """
        assert z.shape == z_prime.shape, "z and z_prime must have the same shape"
        # mu_z/sigma_z are computed per sample, not pooled across the batch — this already is instance-level (AdaIN-style) normalization, not batch normalization ********************************************************************************************************
        if z.dim() == 4:
            reduce_dims = (2, 3) # Spatial dimensions for Conv features
        elif z.dim() == 2:
            reduce_dims = (1,) # Feature dimension for FC features
        else:
            raise ValueError(f"unsupported tensor dim {z.dim()} for StyleBlock (expected 2 or 4).")

        # Compute statistics for content features
        mu_z = z.mean(dim=reduce_dims, keepdim=True)
        sigma_z = z.std(dim=reduce_dims, keepdim=True) + self.eps

        # Compute statistics for style features
        mu_hat = z_prime.mean(dim=reduce_dims, keepdim=True)
        sigma_hat = z_prime.std(dim=reduce_dims, keepdim=True) + self.eps

        # Apply style transfer
        return sigma_hat * ((z - mu_z) / sigma_z) + mu_hat
        
class ResidualBlock(nn.Module):
    """
    Residual block for the bottom branch of the target model.
    
    Creates a residual path that helps the bottom branch adapt
    to target data while preserving source knowledge.
    """
    def __init__(self, dim: int, dropout_prob: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.alpha = 0.4 # Residual scaling factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply residual transformation to input features.
        
        Args:
            x: Input features of shape (B, D)
        
        Returns:
            Transformed features of shape (B, D)
        """
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.ln2(h)
        h = self.dropout(h)
        return self.alpha * h

# =============================================================================
# PROTOTYPE-BASED FUNCTIONS
# =============================================================================
# Computes pseudo-logits (softmax outputs) by comparing features to class prototypes 
def pseudo_logits_from_prototypes(features: torch.Tensor, class_prototypes: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    Compute pseudo-logits by comparing features to class prototypes.
    
    This is used for distillation where the teacher (bottom branch)
    teaches the student (top branch) about target data. ***
    
    Args:
        features: Feature vectors of shape (N, D)
        class_prototypes: Prototype vectors of shape (K, D)
        tau: Temperature for scaling cosine similarity
    
    Returns:
        Pseudo-logits of shape (N, K)
    """
    tau = max(float(tau), EPS)
    f_norm = F.normalize(features, dim=1, eps=1e-12)
    p_norm = F.normalize(class_prototypes, dim=1, eps=1e-12)
    return (f_norm @ p_norm.t()) / tau
    
@torch.no_grad()
def prototype_distance(
    features: torch.Tensor,
    class_prototypes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distance from features to nearest class prototype.
    
    Args:
        features: Feature vectors of shape (N, D)
        class_prototypes: Prototype vectors of shape (K, D)
    
    Returns:
        min_dist: Minimum cosine distance to any prototype
        argmin: Index of nearest prototype
    """
    f = F.normalize(features, dim=1, eps=1e-12)
    p = F.normalize(class_prototypes, dim=1, eps=1e-12)
    cos = torch.matmul(f, p.t())  
    dist = 1.0 - cos
    min_dist, argmin = dist.min(dim=1)
    return min_dist, argmin
    
def _to_numpy(x):
    """Convert tensor to numpy array if needed."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# =============================================================================
# MAIN TARGET MODEL CLASS
# =============================================================================
# Target model
class TargetModel(nn.Module):
    """
    Main target model for source-free open-set domain adaptation.
    
    Architecture:
    - Two parallel branches (top: student, bottom: teacher)
    - Top branch: Source encoder + source classifier (frozen source knowledge)
    - Bottom branch: Source encoder + residual + style module (adapts to target)
    - Clustering on bottom branch features (2*k clusters)
    - Prototypes for known classes
    - Style transfer for data augmentation
    - Multiple loss terms for adaptation
    
    Key Components:
    1. Encoder: Shared backbone (from source model)
    2. Residual: Adapts features to target domain
    3. StyleBlock: Cross-sample style augmentation
    4. Classifier: Source classifier (frozen for top branch, trainable for bottom)
    5. Cluster centers: 2*k clusters for target data
    6. Class prototypes: Known class prototypes for distillation
    """
    def __init__(
        self,
        encoder: nn.Module,
        source_classifier: nn.Module,
        feature_dim: int,
        num_source_classes: int,
        use_style: bool = True,
        style_eps: float = 1e-5):
        super().__init__()

        # ---------------------------------------------------------------------
        # Basic parameters
        # ---------------------------------------------------------------------
        self.feature_dim = feature_dim
        self.num_source_classes = num_source_classes # number of known classes (classes in source domain) = k
        self.num_clusters = 2 * num_source_classes # number of clusters = 2*k

        # ---------------------------------------------------------------------
        # Buffers for cluster information
        # ---------------------------------------------------------------------
        self.register_buffer("cluster_centers", torch.zeros(self.num_clusters, feature_dim))
        self.register_buffer("cluster_entropy_threshold", torch.tensor(0.0))
        self.register_buffer("proto_dist_threshold", torch.tensor(0.0))
        self.register_buffer("saved_cluster_entropies", torch.full((self.num_clusters,), torch.nan))
        self.register_buffer("saved_cluster_counts", torch.zeros(self.num_clusters))

        # ---------------------------------------------------------------------
        # Buffers for class prototypes
        # ---------------------------------------------------------------------
        self.register_buffer("class_prototypes", torch.zeros(self.num_source_classes, feature_dim))
        self.register_buffer("prototype_counts", torch.zeros(self.num_source_classes))

        # ---------------------------------------------------------------------
        # Buffers for variance anchoring (preserve feature spread)
        # ---------------------------------------------------------------------
        self.register_buffer("target_cluster_variance", torch.zeros(self.num_clusters))
        self.register_buffer("has_target_variance", torch.tensor(False))

        # ---------------------------------------------------------------------
        # Network modules
        # ---------------------------------------------------------------------
        self.encoder = encoder
        
        # Extract last source linear layer (the classifier)
        if isinstance(source_classifier, nn.Linear):
            src_linear = source_classifier
        elif isinstance(source_classifier, nn.Sequential):
            linear_layers = [m for m in source_classifier.modules() if isinstance(m, nn.Linear)]
            if len(linear_layers) == 0:
                raise ValueError("source_classifier Sequential has no Linear layer")
            src_linear = linear_layers[-1]
        else:
            raise TypeError(f"Unsupported source_classifier type: {type(source_classifier)}")
            
        assert src_linear.weight.shape[0] >= num_source_classes
        assert src_linear.weight.shape[1] == feature_dim

        # Use SOURCE classifier (k-way)
        self.classifier = src_linear
        self.num_source_classes = num_source_classes

        # Residual block for bottom branch adaptation
        self.residual = ResidualBlock(feature_dim)
        
        # ---------------------------------------------------------------------
        # Anchor parameters (prevent encoder drift from source)
        # ---------------------------------------------------------------------
        self._anchor_params = {}
        with torch.no_grad():
            for name, p in self.encoder.named_parameters():
                if p.requires_grad:
                    self._anchor_params[f"encoder.{name}"] = p.detach().clone()
            for name, p in self.residual.named_parameters():
                if p.requires_grad:
                    self._anchor_params[f"residual.{name}"] = p.detach().clone()

        # Style module for cross-sample style augmentation
        self.style_module = StyleBlock(eps=style_eps) if use_style else None
        
        self.last_pseudo_labels = None
        self._bootstrapped = False # Track if prototypes are initialized

    # -------------------------------------------------------------------------
    # ANCHOR LOSS (Prevent catastrophic forgetting)
    # -------------------------------------------------------------------------
    def anchor_l2_loss(self) -> torch.Tensor:
        """
        L2 anchor loss to prevent drift from source weights.
        
        Ensures the encoder and residual don't deviate too far from
        the source initialization, preserving source knowledge.
        """
        loss = 0.0
        cnt = 0
        for name, p in self.named_parameters():
            if name in self._anchor_params:
                loss = loss + F.mse_loss(p, self._anchor_params[name])
                cnt += 1
        if cnt == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return loss/cnt

    # -------------------------------------------------------------------------
    # K-MEANS CLUSTER UPDATE
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def update_kmeans(self, target_loader, device, num_iters: int = 20):
        """
        Update cluster centers using k-means on current target features.
        
        This is called periodically during adaptation to refine clusters
        as the model learns better target representations.
        
        Args:
            target_loader: DataLoader for target training data
            device: Device to run on
            num_iters: Number of k-means iterations
        """
        self.eval()
        feats = []

        # Collect features from all target samples
        for batch in target_loader:
            x = batch[0].to(device)
            out = self.forward(x, return_all=True)
            feats.append(out["f_t"].detach().cpu())
            
        feats_all = torch.cat(feats, dim=0).to(device)

        # Run k-means clustering
        centers, _ = run_kmeans(feats_all, num_clusters=self.num_clusters, num_iters=num_iters)
        self.cluster_centers.data.copy_(centers)

    # -------------------------------------------------------------------------
    # SOFT KNOWN WEIGHT (For known/unknown estimation)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _soft_known_weight(self, sample_entropy: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Convert entropy to a continuous known/unknown weight.
        
        Lower entropy = higher known confidence.
        
        Args:
            sample_entropy: Entropy values per sample
            tau: Entropy threshold
        
        Returns:
            Known confidence weights in [0, 1]
        """
        maxH = math.log(self.num_clusters + EPS)
        # temperature for sigmoid: scale with entropy range
        t = max(0.07 * maxH, 1e-3)
        if not math.isfinite(float(tau)):
            w = torch.ones_like(sample_entropy)
        else:
            w = torch.sigmoid((float(tau) - sample_entropy) / t)
        return w.clamp(0.0, 1.0)

    # -------------------------------------------------------------------------
    # BOOTSTRAP PROTOTYPES (Initialize class prototypes)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _bootstrap_prototypes_if_needed(self, f_t: torch.Tensor, logits_st: torch.Tensor):
        """
        Initialize class prototypes if not already done.
        
        Uses high-confidence source predictions to initialize prototypes.
        This is a critical first step for the distillation process.
        """
        if self._bootstrapped:
            return
            
        if self.class_prototypes.norm(dim=1).max().item() > 1e-6:
            self._bootstrapped = True
            return
            
        # Use high-confidence source predictions to bootstrap
        probs_st = F.softmax(logits_st, dim=1)
        conf, pred = probs_st.max(dim=1)
        
        # Lowered threshold from 0.75 to 0.6 for better initialization
        keep = (conf >= 0.6) & (pred < self.num_source_classes)
        if keep.any():
            feats = F.normalize(f_t[keep], dim=1, eps=1e-12)
            labels = pred[keep]
            for c in labels.unique():
                idx = labels == c
                mean_feat = feats[idx].mean(dim=0)
                self.class_prototypes[c] = F.normalize(mean_feat, dim=0, eps=1e-12)
                self.prototype_counts[c] = 1.0

        # If still no prototypes, initialize with random features
        # Fallback: initialize with random features
        if self.class_prototypes.norm(dim=1).max().item() <= 1e-6:
            # Take random samples from f_t
            num_samples = min(self.num_source_classes, f_t.size(0))
            indices = torch.randperm(f_t.size(0))[:num_samples]
            for c in range(min(num_samples, self.num_source_classes)):
                self.class_prototypes[c] = F.normalize(f_t[indices[c]], dim=0, eps=1e-12)
                self.prototype_counts[c] = 1.0

        # Calibrate cluster entropy threshold
        maxH = math.log(self.num_clusters + EPS)
        saved_E = torch.nan_to_num(
            self.saved_cluster_entropies,
            nan=maxH,
            posinf=maxH,
            neginf=0.0
        )
        e = (saved_E / maxH).clamp(0.0, 1.0)
        valid = self.saved_cluster_counts >= 5
        
        if not valid.any():
            self.cluster_entropy_threshold.fill_(0.6)
            self._bootstrapped = True
            return

        e_valid = e[valid]
        # Use lower quantile (30%) for more conservative threshold
        tau = torch.quantile(e_valid, 0.3).item()
        self.cluster_entropy_threshold.fill_(tau)
        
        self._bootstrapped = True

    # -------------------------------------------------------------------------
    # CALIBRATE PROTOTYPE DISTANCE THRESHOLD
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def calibrate_proto_threshold(self, target_loader, device):
        """
        Calibrate prototype distance threshold for open-set recognition.
        
        Samples with ***distance > threshold*** are considered unknown.
        """
        self.eval()
        dists = []
        
        for batch in target_loader:
            x = batch[0].to(device)
            out = self.forward(x, return_all=True)
            f_t = out["f_t"]
            w = out["w_final"]
            min_dist, _ = prototype_distance(f_t, self.class_prototypes)

            # Use high-confidence known samples
            keep = w >= 0.6
            if keep.any():
                dists.append(min_dist[keep].detach().cpu())
                
        if len(dists) == 0:
            self.proto_dist_threshold.fill_(0.3)
            return
            
        d = torch.cat(dists)
        self.proto_dist_threshold.fill_(torch.quantile(d, 0.90))

    # -------------------------------------------------------------------------
    # CALIBRATE TARGET VARIANCE (For variance anchoring)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def calibrate_target_variance(self, target_loader, device):
        """
        Records the CURRENT per-cluster feature variance as the reference
        to preserve. 
        """
        self.eval()
        feats = []

        # Collect features
        for batch in target_loader:
            x = batch[0].to(device)
            out = self.forward(x, return_all=True)
            feats.append(out["f_t"].detach().cpu())
            
        feats_all = torch.cat(feats, dim=0).to(device)
        labels = assign_clusters(feats_all, self.cluster_centers)

        # Compute variance per cluster
        var_per_cluster = torch.zeros(self.num_clusters, device=device)
        for c in range(self.num_clusters):
            mask = labels == c
            if mask.sum() > 1:
                var_per_cluster[c] = feats_all[mask].var(dim=0, unbiased=True).mean()
            else:
                var_per_cluster[c] = 0.0
                
        self.target_cluster_variance.copy_(var_per_cluster)
        self.has_target_variance.fill_(True)

    # -------------------------------------------------------------------------
    # OPEN-SET PREDICTION
    # -------------------------------------------------------------------------
    # entropy > entropy_thresh --> unknown
    # entropy < entropy_thresh --> known
    @torch.no_grad()
    def predict_with_entropy_osr(
        self,
        x: torch.Tensor,
        use_branch: str = "target",
        entropy_thresh: Optional[float] = None,
        proto_dist_thresh: Optional[float] = None,):
        """
        Perform open-set inference with known/unknown classification.
        
        Known samples -> class in [0, k-1]
        Unknown samples -> label = -1 (not discarded)
        
        Args:
            x: Input images
            use_branch: "target" (bottom) or "source" (top) branch
            entropy_thresh: Entropy threshold for OSR 
            proto_dist_thresh: Prototype distance threshold 
        
        Returns:
            final_pred: Class predictions with -1 for unknown
            diagnostics: Dictionary with entropy, energy, w_final, etc.        
        """
        self.eval()
        out = self.forward(x, return_all=True)

        # Use specified branch
        logits = out["logits_t"] if use_branch == "target" else out["logits_st"]
        probs = F.softmax(logits, dim=1)

        # Compute entropy and energy
        entropy = -(probs * torch.log(probs + EPS)).sum(dim=1)
        energy = -torch.logsumexp(logits, dim=1)
        pred_known = probs.argmax(dim=1)

        # Initialize predictions as known
        final_pred = pred_known.clone()
        unknown_mask = torch.zeros_like(pred_known, dtype=torch.bool)
        
        # Apply entropy threshold
        if entropy_thresh is not None:
            unknown_mask |= entropy > float(entropy_thresh)
            
        # Apply prototype distance threshold
        if proto_dist_thresh is not None:
            f_t = out["f_t"]
            min_dist, _ = prototype_distance(f_t, self.class_prototypes)
            unknown_mask |= min_dist > float(proto_dist_thresh)
            
        # Mark unknowns as -1
        final_pred[unknown_mask] = -1 # UNKNOWN KEPT HERE
        
        diagnostics = {
            "entropy": entropy,
            "energy": energy,
            "unknown_mask": unknown_mask,
            "w_final": out["w_final"],
        }
        return final_pred, diagnostics

    # -------------------------------------------------------------------------
    # CATEGORY-AGNOSTIC FUNCTIONS (Eq 3-5 from Cat-agnostic paper) for clustering
    # -------------------------------------------------------------------------
    def _energy_to_rho(self, energy: torch.Tensor, temp: float = 2.0) -> torch.Tensor:
        """Convert energy to confidence weight rho."""
        mean = energy.mean().detach()
        std = energy.std().detach() + EPS
        e = (energy - mean) / std
        rho = torch.sigmoid(-e / temp)
        return rho.clamp(0.1, 0.9)

    @torch.no_grad()
    def cat_agnostic_eq3(
        self,
        f_t: torch.Tensor,
        energy_t: torch.Tensor,
        temp_energy: float = 2.0) -> torch.Tensor:
        """
        Category-agnostic clustering distribution (Eq 3).
        
        Computes soft cluster assignments weighted by energy confidence.
        """
        f_norm = F.normalize(f_t, dim=1, eps=1e-12)
        mu_norm = F.normalize(self.cluster_centers, dim=1, eps=1e-12)
        cos_sim = torch.matmul(f_norm, mu_norm.t())
        rho_t = self._energy_to_rho(energy_t, temp_energy)
        logits = rho_t.unsqueeze(1) * cos_sim
        return F.softmax(logits, dim=1)

    def cat_agnostic_eq4(
        self,
        f_st: torch.Tensor,
        energy_st: torch.Tensor,
        temp_energy: float = 2.0
    ) -> torch.Tensor:
        """
        Category-agnostic student distribution (Eq 4).
        
        Student's version of soft cluster assignments.
        """
        f_norm = F.normalize(f_st, dim=1, eps=1e-12)
        mu_norm = F.normalize(self.cluster_centers, dim=1, eps=1e-12)
        cos_sim = torch.matmul(f_norm, mu_norm.t())
        rho_st = self._energy_to_rho(energy_st, temp_energy)
        logits = rho_st.unsqueeze(1) * cos_sim
        return F.softmax(logits, dim=1)

    def kl_divergence(
        self,
        p_inherent: torch.Tensor,
        p_pred: torch.Tensor,
        eps: float = 1e-8) -> torch.Tensor:
        """
        cat-agnostic paper Eq.(5): KL( P̃_clu || P_clu ).
        kl_div(log(P_pred), P_inherent) = KL(P_inherent || P_pred) aligns teacher and student distributions.
        """
        p_inherent = p_inherent.clamp(min=eps, max=1.0)
        p_pred = p_pred.clamp(min=eps, max=1.0)
        kl_loss = F.kl_div(p_pred.log(), p_inherent, reduction="batchmean")
        return kl_loss

    # -------------------------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------------------------
    def forward(self, x, return_all=True):
        """
        Forward pass through the target model.
        
        Both branches share the encoder but have different paths:
        - Top (student): encoder --> classifier
        - Bottom (teacher): encoder --> residual --> classifier # and style*************************************??
        
        Args:
            x: Input images
            return_all: If True, return all losses and diagnostics
        
        Returns:
            Dictionary containing features, logits, losses, and diagnostics
        """
        # ---------------------------------------------------------------------
        # Extract features from both branches
        # ---------------------------------------------------------------------
        f_st = self.encoder(x) # Top branch (student)
        logits_st = self.classifier(f_st) # Source classifier
        
        f_t = self.residual(f_st) + f_st # Bottom branch (teacher)
        logits_t = self.classifier(f_t) # Source classifier on adapted features
        
        probs_t = F.softmax(logits_t, dim=1) # Probabilities from bottom branch
        
        B = x.size(0)
        device = x.device

        # ---------------------------------------------------------------------
        # Compute known confidence from source classifier
        # ---------------------------------------------------------------------
        with torch.no_grad():
            probs_st = F.softmax(logits_st, dim=1)
            conf_st, _ = probs_st.max(dim=1)
            
        conf_thresh = 0.8
        temp_conf = 0.15
        w_known = torch.sigmoid((conf_st - conf_thresh) / temp_conf)

        # Bootstrap prototypes if not initialized
        self._bootstrap_prototypes_if_needed(f_t, logits_st)
        
        if not torch.isfinite(w_known).all():
            raise RuntimeError("w_known contains NaN/Inf")

        # ---------------------------------------------------------------------
        # Class balance regularization
        # ---------------------------------------------------------------------
        mean_prob = probs_st.mean(dim=0)
        uniform = torch.ones_like(mean_prob) / self.num_source_classes
        loss_balance = F.kl_div(
            torch.log(mean_prob + 1e-8),
            uniform,
            reduction="batchmean"
        ) * 0.5  
                
        # ---------------------------------------------------------------------
        # Logit consistency loss (student learns from teacher)
        # ---------------------------------------------------------------------
        T_cons = 2.0
        with torch.no_grad():
            teacher_prob = F.softmax(logits_t.detach() / T_cons, dim=1)
        student_log_prob = F.log_softmax(logits_st / T_cons, dim=1)
        logit_consistency_loss = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction="batchmean"
        ) * (T_cons * T_cons)

        # ---------------------------------------------------------------------
        # Category-agnostic KL loss (Eq 3-5)
        # ---------------------------------------------------------------------
        # Energies for cat-agnostic Eq3/Eq4
        energy_t = -torch.logsumexp(logits_t, dim=1)
        energy_st = -torch.logsumexp(logits_st, dim=1)
        
        p3_cat = self.cat_agnostic_eq3(f_t, energy_t, temp_energy=5.0)
        p4_cat = self.cat_agnostic_eq4(f_st, energy_st, temp_energy=5.0)
        cat_kl_loss = 0.5 * self.kl_divergence(p3_cat.detach(), p4_cat)

        # ==========================================
        # HYBRID CLUSTER ENTROPY (sample-level + population-level)
        # ==========================================
        maxH = math.log(self.num_clusters + EPS)

        # Sample-level entropy ************************************************************************************************
        q_ij = compute_q_ij(f_t, self.cluster_centers, temp=TEMP)
        sample_entropy = -(q_ij * torch.log(q_ij + EPS)).sum(dim=1)
        sample_entropy_n = (sample_entropy / maxH).clamp(0.0, 1.0)

        # Population-level entropy (stable, whole-dataset estimate)
        cluster_labels = assign_clusters(f_t, self.cluster_centers)
        pop_entropy = torch.nan_to_num(
            self.saved_cluster_entropies, nan=maxH, posinf=maxH, neginf=0.0
        )[cluster_labels]
        pop_entropy_n = (pop_entropy / maxH).clamp(0.0, 1.0)

        # Blend both estimates *********************************************************************************
        cluster_entropy_n = 0.5 * sample_entropy_n + 0.5 * pop_entropy_n

        # Convert to known weight
        tau = float(self.cluster_entropy_threshold.item())
        if not math.isfinite(tau):
            tau = 0.6  # safe fallback
        w_cluster = torch.sigmoid((tau - cluster_entropy_n) / 0.15)

        # Final known weight (mix of source confidence and cluster entropy)
        w_final = 0.7 * w_known + 0.3 * w_cluster
        w_final = w_final.clamp(0.05, 0.95)
        
        # CE only on known
        probs_st = F.softmax(logits_st, dim=1)

        # ---------------------------------------------------------------------
        # CE loss on known samples (top branch)
        # ---------------------------------------------------------------------
        conf_st, pred_st = probs_st.max(dim=1)
        entropy_t = -(probs_t * torch.log(probs_t + EPS)).sum(dim=1)
        entropy_n = entropy_t / math.log(self.num_source_classes + EPS)
        
        loss_known_ce = torch.tensor(0.0, device=x.device)
        mask = (w_final >= 0.75) & (conf_st >= 0.8)
        if mask.any():
            ce = F.cross_entropy(
                logits_st[mask, :self.num_source_classes],
                pred_st[mask].detach()
            )
            loss_known_ce = ce
            
        # ---------------------------------------------------------------------
        # Prototype loss
        # ---------------------------------------------------------------------
        proto_loss = torch.tensor(0.0, device=x.device)
        if self._bootstrapped:
            proto_vec = self.class_prototypes[pred_st]
            valid = proto_vec.norm(dim=1) > 1e-6
            if valid.any():
                f_n = F.normalize(f_t[valid], dim=1)
                p_n = F.normalize(proto_vec[valid], dim=1)
                w = (w_final[valid] * conf_st[valid]).detach()
                proto_loss = ((1.0 - F.cosine_similarity(f_n, p_n)) * w).sum() / (w.sum() + EPS)

        # ---------------------------------------------------------------------
        # Prototype distillation loss (teacher --> student)
        # ---------------------------------------------------------------------
        distill_loss = torch.tensor(0.0, device=x.device)
        valid = (w_final >= 0.7) & (conf_st >= 0.75)
        if valid.any() and self._bootstrapped:
            T = 5.0
            with torch.no_grad():
                f_teacher = F.normalize(f_t[valid].detach(), dim=1)
                proto_teacher = F.normalize(
                    self.class_prototypes.detach(),
                    dim=1)
                teacher_logits = (f_teacher @ proto_teacher.t())
                teacher_prob = F.softmax(teacher_logits / T, dim=1)

            # Normalize student logits
            student_logits = logits_st[valid, :self.num_source_classes]
            student_logits = student_logits - student_logits.mean(dim=1, keepdim=True)
            student_logits = student_logits / (student_logits.std(dim=1, keepdim=True) + 1e-6)
            student_logp = F.log_softmax(student_logits / T, dim=1)
            
            distill_loss = F.kl_div(
                student_logp,
                teacher_prob,
                reduction="batchmean") * (T * T)

        # ---------------------------------------------------------------------
        # Anchor loss (prevent encoder drift)
        # ---------------------------------------------------------------------
        anchor_loss = torch.tensor(0.0, device=x.device)
        if self.training and self._bootstrapped:
            anchor_loss = 0.01 * self.anchor_l2_loss()
            
        # ---------------------------------------------------------------------
        # Style consistency loss (cross-sample style swap)
        # ---------------------------------------------------------------------
        style_loss = torch.tensor(0.0, device=x.device)
        if self.style_module is not None and self.training and B > 1:
            perm = torch.randperm(B, device=device)
            # Guard against identity permutation
            if torch.equal(perm, torch.arange(B, device=device)):
                perm = torch.roll(torch.arange(B, device=device), shifts=1)

            f_content = f_st.detach()          # this sample's content, frozen
            f_style_foreign = f_t.detach()[perm]  # a DIFFERENT sample's style stats **********************************

            # Inject foreign style
            # inject foreign style onto this sample's own content
            f_styled = self.style_module(f_content, f_style_foreign)
            logits_styled = self.classifier(f_styled)

            with torch.no_grad():
                target_probs = F.softmax(logits_st.detach(), dim=1)

            # classifier prediction MUST stay consistent despite the foreign style
            # Consistency loss
            style_loss = F.kl_div(
                F.log_softmax(logits_styled, dim=1),
                target_probs,
                reduction="batchmean"
            )
            
        # ---------------------------------------------------------------------
        # Variance anchor loss (preserve feature spread)
        # ---------------------------------------------------------------------
        variance_anchor_loss = torch.tensor(0.0, device=x.device)
        if self.training and bool(self.has_target_variance.item()):
            with torch.no_grad():
                cluster_labels_va = assign_clusters(f_t, self.cluster_centers)  # centers frozen, no grad path
                
            per_cluster_losses = []
            for c in range(self.num_clusters):
                mask = cluster_labels_va == c
                if mask.sum() > 1:
                    cur_var = f_t[mask].var(dim=0, unbiased=True).mean()#computes variance around the batch's own current mean, not around the fixed cluster_centers[c] ************************************************************************************************************?cluster
                    target_var = self.target_cluster_variance[c]
                    per_cluster_losses.append((cur_var - target_var).pow(2))
                    
            if per_cluster_losses:
                variance_anchor_loss = torch.stack(per_cluster_losses).mean()
                
        # ---------------------------------------------------------------------
        # Assemble outputs
        # ---------------------------------------------------------------------
        outputs = {
            "f_st": f_st,
            "f_t": f_t,
            "logits_st": logits_st,
            "logits_t": logits_t,
            "probs_t": probs_t,
            # losses
            "loss_known_ce": loss_known_ce,
            "proto_loss": proto_loss,
            "distill_loss": distill_loss,
            "cat_kl_loss": cat_kl_loss,
            "logit_consistency_loss": logit_consistency_loss,
            "anchor_loss": anchor_loss,
            "style_loss": style_loss,
            # diagnostics
            "conf_st": conf_st.detach(),
            "energy_t": energy_t.detach(),
            "energy_st": energy_st.detach(),
            "w_final": w_final,
            "loss_balance": loss_balance,
            "variance_anchor_loss": variance_anchor_loss,
        }
        
        if return_all:
            return outputs
        return {"logits_t": logits_t,
            "f_t": f_t,
            "w_final": w_final,}

# =============================================================================
# LAYER PROBING FUNCTIONS (For deciding where to freeze)
# =============================================================================
@torch.no_grad()
def _extract_layer_features(encoder, layer_names, loader, device):
    """
    Extract features from specified layers using forward hooks.
    
    Args:
        encoder: Encoder network
        layer_names: List of layer names to extract features from
        loader: DataLoader for input data
        device: Device to run on
    
    Returns:
        Dictionary mapping layer names to extracted features
    """
    encoder.eval()
    captured = {name: [] for name in layer_names}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            feat = out
            if feat.dim() == 4:  # (B, C, H, W) --> global average pool # CNN features --> global average pool
                feat = feat.mean(dim=(2, 3))
            elif feat.dim() > 2:
                feat = feat.flatten(1)
            captured[name].append(feat.detach().cpu())
        return hook

    # Register hooks
    named = dict(encoder.named_modules())
    for name in layer_names:
        if name not in named:
            raise KeyError(f"Layer '{name}' not found. Available: {list(named.keys())}")
        handles.append(named[name].register_forward_hook(make_hook(name)))

    # Run through loader
    for batch in loader:
        x = batch[0].to(device)
        encoder(x)

    # Remove hooks
    for h in handles:
        h.remove()

    return {name: torch.cat(feats, dim=0) for name, feats in captured.items()}

def _silhouette_cosine(features, labels, num_clusters):
    """
    Lightweight cosine-based silhouette score (no sklearn dependency).
    Compute silhouette score using cosine distance.
    
    A measure of how well-separated clusters are. ********************************
    Higher = better separation
    """
    f = F.normalize(features, dim=1)
    scores = []
    
    for c in range(num_clusters):
        in_mask = labels == c
        if in_mask.sum() < 2:
            continue
            
        f_in = f[in_mask]
        intra = (1.0 - (f_in @ f_in.t())).clamp_min(0.0)
        a_i = intra.sum(dim=1) / (in_mask.sum() - 1)

        b_i = torch.full((in_mask.sum(),), float("inf"))
        for c2 in range(num_clusters):
            if c2 == c:
                continue
            out_mask = labels == c2
            if out_mask.sum() == 0:
                continue
            inter = (1.0 - (f_in @ f[out_mask].t())).clamp_min(0.0).mean(dim=1)
            b_i = torch.minimum(b_i, inter)

        s_i = (b_i - a_i) / torch.maximum(a_i, b_i).clamp_min(1e-8)
        scores.append(s_i)
        
    if not scores:
        return float("nan")
    return torch.cat(scores).mean().item()

""" 
probe_layers: 
take features from candidate near-output layers → cluster known target features → measure separation (silhouette) → check where unknown target features fall relative to those known clusters (distance-based AUROC). *****************************************
"""
def probe_layers(encoder, layer_names, known_loader, unknown_loader, device,
                  num_clusters, num_kmeans_iters=50):
    """
    Probe different layers to find the best freezing boundary.
    
    For each layer, fits k-means on known features and reports:
    - Silhouette: how well-separated KNOWN clusters are
    - auroc_known_vs_unknown: how cleanly unknown samples fall OUTSIDE known clusters
    
    Args:
        encoder: Encoder network
        layer_names: List of layer names to probe
        known_loader: DataLoader for known class samples
        unknown_loader: DataLoader for unknown class samples
        device: Device to run on
        num_clusters: Number of clusters for k-means
        num_kmeans_iters: Number of k-means iterations
    
    Returns:
        Dictionary mapping layer names to probe results
    """
    # Extract features from each layer
    known_feats = _extract_layer_features(encoder, layer_names, known_loader, device)
    unknown_feats = _extract_layer_features(encoder, layer_names, unknown_loader, device)

    results = {}
    for name in layer_names:
        f_known = known_feats[name].to(device)
        f_unknown = unknown_feats[name].to(device)

        # Cluster known features
        centers, labels_known = run_kmeans(f_known, num_clusters=num_clusters, num_iters=num_kmeans_iters)
        silhouette = _silhouette_cosine(f_known, labels_known, num_clusters)
        
        # Compute distance to nearest center for known and unknown
        d_known = squared_cosine_cdist(F.normalize(f_known, dim=1), centers).min(dim=1).values
        d_unknown = squared_cosine_cdist(F.normalize(f_unknown, dim=1), centers).min(dim=1).values

        # Compute AUROC for known vs unknown separation
        scores = torch.cat([-d_known, -d_unknown])  # higher score = more likely to be "known" 
        is_known = torch.cat([torch.ones(len(d_known)), torch.zeros(len(d_unknown))]).bool()

        order = torch.argsort(scores, descending=True)
        lab = is_known[order].to(torch.int)
        P, N = lab.sum().item(), (lab == 0).sum().item()
        
        if P == 0 or N == 0:
            auroc_val = float("nan")
        else:
            tpr = torch.cumsum(lab, dim=0).float() / P
            fpr = torch.cumsum(1 - lab, dim=0).float() / N
            auroc_val = torch.trapz(tpr, fpr).item()

        results[name] = {"silhouette": silhouette, "auroc_known_vs_unknown": auroc_val}

    return results

    