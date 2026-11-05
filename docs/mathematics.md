# GRAIL: mathematical specification

## Task and ordinal assumption

Y ∈ {1,2,3,4} measures increasing **traffic disruption**, conditional on a recorded incident. It is not a crash-occurrence, injury, fatality, or intervention-effect target. The ordered endpoints justify testing ordinal learning, not assuming that differences between successive grades represent equal minutes of delay. Ordinal MAE is a rank-distance diagnostic. Unknown provider-specific coding changes can invalidate comparability even when the numbers remain ordered.

## Encoder and cumulative-link head

A two-layer ReLU MLP maps preprocessed x to h. Set s(x)=wᵀh+b. Learn t₁=a, t₂=a+softplus(d₁)+ε, t₃=t₂+softplus(d₂)+ε, ε=10⁻⁴. Thus thresholds increase at every optimization step, not just at an optimum. qₖ=σ(s−tₖ) is nonincreasing in k. Define p=(1−q₁,q₁−q₂,q₂−q₃,q₃). Every p is nonnegative and its sum telescopes to one.

Let zᵢₖ=1[Yᵢ>k]. The per-incident all-threshold loss is

Lᵢ = (1/3) Σₖ {softplus(sᵢ−tₖ) − zᵢₖ(sᵢ−tₖ)}.

This is stable binary cross-entropy in logits. Its derivative with respect to each logit is (qᵢₖ−zᵢₖ)/3. This objective is a proper binary score at each cumulative threshold, although the shared-score proportional-odds restriction may be misspecified. It is an established cumulative-link/CORAL-family formulation, not a new ordinal head.

The `coral` baseline uses the same shared-score loss with explicitly ordered thresholds, a documented constrained CORAL variant. This is intentionally the same ordinal core used by GRAIL; the tested contribution is geographic regularization. `corn` instead learns conditional logits cₖ for P(Y>k | Y>k−1,x). Eligible training tasks have Y>k−1, and qₖ=∏ⱼ≤ₖ σ(cⱼ). The loss sums eligible binary tasks and divides by their training-set count in expectation. It must not fit all conditional heads on all examples. `ce` uses four unrestricted logits and multiclass cross-entropy.

Thresholds are initialized from training-only cumulative label prevalence: t_k = -logit(mean(1[Y>k])). The ordered gaps are mapped through inverse softplus. This helps represent a dominant middle class without spending a short optimization budget expanding nearly equal initial intervals. All cumulative-link baselines use the same initialization. An earlier narrow-gap diagnostic is retained separately.

## Ordinary group DRO and fixed weights

Let n_g be training group sizes, π_g=n_g/N and L_g the mean training loss in group g. ERM minimizes Σπ_g L_g; fixed uniform geography minimizes ΣL_g/G. DRO approximates min_θ max_{w∈Δ} Σw_g L_g through exponentiated ascent:

w_g⁺ ∝ w_g exp(η L_g).

To obtain an unbiased minibatch gradient under ordinary uniform row sampling, multiply each row's loss by w_g/π_g. Normalizing only over groups present in a minibatch would change the objective. We update the adversary once per epoch after computing every group mean with a frozen model. This is less responsive but more stable for sparse groups. AdamW, early stopping and validation selection remain essential.

## Proposed shrinkage-regularized variant

Define r_g=n_g/(n_g+τ), and a shrunken risk

S_g = r_g L_g + (1−r_g) Σ_h π_h L_h.

r_g is an **estimation-confidence heuristic**, not an estimate of label correctness. Small groups borrow the training average. This may conceal genuine rare-group harms; report those groups rather than declaring them unreliable.

For u_g=1/G solve one mirror-ascent step

argmax_{w∈Δ} ⟨w,S⟩ − λ KL(w||u) − KL(w||w_old)/η.

The Lagrange multiplier for Σw=1 gives

log w_g = [log w_old,g + ηS_g + ηλ log u_g]/(1+ηλ) + constant.

Because u is uniform, its log term cancels in softmax. After this exact step, apply w⁺=(1−α)w+αu. The mixture is a stabilizing floor, not claimed to be the exact solution of an extra constrained optimization problem. It guarantees α/G ≤ w_g⁺ ≤ 1−α+α/G.

The model must differentiate **the same shrunken objective**:

Σ_g w_g S_g = Σ_h v_h L_h,

v_h = w_h r_h + π_h Σ_g w_g(1−r_g).

`effective_weights()` implements v. It is nonnegative and sums to one. During the model update w is treated as fixed. All counts and losses come from training data; no test-state labels inform r, w or v. If τ=0 and λ=α=0, this reduces to ordinary group DRO (up to the epoch update schedule).

These are familiar shrinkage, entropic regularization and mirror-ascent ingredients. Combining them for this task is a research hypothesis, not an established novel algorithm. There is no theorem that optimizing training groups improves unseen-state performance.

## Class imbalance and inference

Balanced class weighting uses a_y=N/(4n_y), fitted on training only. It changes the optimized distribution and can damage population calibration. Controlled inverse-frequency sampling is tested only for ERM and never combined with class weights or DRO, avoiding accidental double correction. The four-label objective never changes to binary.

`predict` returns argmax p+1. For a pure absolute-rank-cost decision, the posterior median would be appropriate; this implementation keeps argmax consistent across baselines and reports ordinal MAE separately. E[Y|x]=Σj·p_j supports descriptive disruption ranking.

Calibration fits T>0 on a separate calibration subset: p'_j=softmax(log(max(p_j,10⁻¹²))/T). Reconstructing cumulative q' from p' preserves ordering. This is class-log-probability temperature scaling, not division of the cumulative logits. It preserves argmax, so it cannot improve recall by itself. Selection validation and calibration have different roles.

Entropy H(p)=−Σp log p ranks review candidates. Risk-coverage curves sort by entropy and report observed classification error among accepted cases. They are empirical diagnostics, not guarantees under shift.
