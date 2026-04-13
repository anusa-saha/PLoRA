# Language Representativeness and Cross-Language Interference in PLoRA

## 1. Why this metric is needed

PLoRA routes language-specific adapters, but the backbone is still shared. That means language-specific behavior can still leak through common residual pathways, attention heads, and latent subspaces. The goal of this note is to define a metric that measures two things at once:

1. **Language representativeness**: how strongly a language is encoded, localized, and causally expressed in the model.
2. **Cross-language interference**: how much one language perturbs another through the shared backbone.

This framing is consistent with the PLoRA idea of depth-dependent language channels, the SPINAL view that important geometric changes can be localized in depth, and the Neural FOXP2 view that language behavior can be probed through selective and causal features. fileciteturn0file0 fileciteturn1file0 fileciteturn1file1

## 2. Notation

Let:

- \(\lambda \in \{1,\dots,m\}\) denote a language
- \(\ell\) denote a layer index
- \(S_\lambda\) denote the sparse set of PLoRA layers selected for language \(\lambda\)
- \(N_\lambda\) denote a set of language-specific features or neurons
- \(\mathcal D_{\mathrm{neutral}}\) denote a set of prompts that do not explicitly specify a target language

We also use:

- \(\alpha_\lambda(\ell)\): layerwise spectral exponent for language \(\lambda\)
- \(L_\lambda(\ell)\): Fisher-Rao belief transport cost at layer \(\ell\)
- \(\mathrm{Sel}_j\): selectivity of feature \(j\)
- \(\mathrm{LiftSlope}_j\): causal lift slope of feature \(j\)
- \(\Delta M_\lambda(x)\): early-step target-language mass advantage on prompt \(x\)

The standardization operator \(z(\cdot)\) means a z-score or any comparable normalization across layers or languages.

## 3. Language representativeness score

We define the representativeness score for language \(\lambda\) as

$$
R_\lambda = w_1 G_\lambda + w_2 C_\lambda + w_3 D_\lambda,
$$

where the three components are:

$$
G_\lambda = \frac{1}{|S_\lambda|}\sum_{\ell \in S_\lambda}
\Big[z\!\left(\alpha_\lambda(\ell)-\bar\alpha(\ell)\right) + z\!\left(-L_\lambda(\ell)\right)\Big],
$$

$$
C_\lambda = \frac{1}{|N_\lambda|}\sum_{j \in N_\lambda}
\Big[z(\mathrm{Sel}_j) + z(\mathrm{LiftSlope}_j)\Big],
$$

$$
D_\lambda = \mathbb E_{x \sim \mathcal D_{\mathrm{neutral}}}[\Delta M_\lambda(x)].
$$

Here:

- \(G_\lambda\) measures **geometry**
- \(C_\lambda\) measures **causal feature strength**
- \(D_\lambda\) measures **behavioral defaultness**

### 3.1 Geometry term \(G_\lambda\)

The geometry term uses the PLoRA and SPINAL style view of language structure. If a language consistently deviates from the multilingual average in selected layers, then it has a distinct depth fingerprint. If it also has low Fisher-Rao transport cost, then the model moves more smoothly toward that language's predictive state. That is exactly the kind of localized geometric pattern reported in SPINAL's terminal calibration analysis. fileciteturn1file0

### 3.2 Causal term \(C_\lambda\)

The causal term is inherited from the Neural FOXP2 logic. A feature should not merely correlate with a language. It should also causally increase the language advantage when intervened on. That is why selectivity alone is not enough, and lift alone is not enough. The combination is more reliable than either statistic by itself. fileciteturn1file1

### 3.3 Behavioral term \(D_\lambda\)

The behavioral term checks whether the model naturally begins in language \(\lambda\) under weak or neutral prompting. This matters because a language can be present in the internal geometry but still fail to become the default generation path. In that case, the language is represented, but not strongly accessible.

## 4. Cross-language interference score

To measure interference from language \(\mu\) into language \(\lambda\), define the directed score

$$
I_{\mu \to \lambda}
=
\eta_1
\mathbb E_{x \sim \mathcal D_\lambda}
\left[
\mathrm{KL}\!\left(
p^{(\lambda)}(\cdot \mid x)\,\|\,p^{(\mu)}(\cdot \mid x)
\right)
\right]
+
\eta_2
\frac{1}{|S_\lambda|}
\sum_{\ell \in S_\lambda}
\frac{
\left\|\Pi_{S_\lambda}\!\left(\Delta W^{(\mu)}_\ell\right)\right\|_F
}{
\left\|\Delta W^{(\mu)}_\ell\right\|_F
}.
$$

This has two parts.

### 4.1 Behavioral leakage

The first term measures how much the output distribution changes when the wrong adapter is used. It captures distributional leakage, not just top-1 token changes. That is important in multilingual generation because interference often appears as subtle probability shifts before it becomes an obvious failure.

### 4.2 Geometry leakage

The second term measures how much of adapter \(\mu\)'s update lives inside the subspace associated with language \(\lambda\). If this projection is large, then the two language routes are sharing directions in representation space. That is a direct sign of interference through the shared backbone.

This directed form is useful because interference can be asymmetric. For example, language \(\mu\) may strongly perturb language \(\lambda\), but the reverse effect may be weaker.

## 5. Final composite score

A single summary score can be defined as

$$
\mathrm{LRIS}_\lambda
=
R_\lambda
-
\gamma \frac{1}{m-1}\sum_{\mu \neq \lambda} I_{\mu \to \lambda}.
$$

Higher values mean:

- stronger representativeness
- better causal support
- stronger defaultness
- lower cross-language interference

So a good multilingual adapter is not only one that improves language \(\lambda\), but one that does so while keeping other languages from leaking into its route.

## 6. Why this is a good fit for your draft

This metric matches the structure of your current PLoRA draft very naturally. PLoRA already uses layerwise language channels, normalized deviation scores, sparse supports, and rank budgeting. SPINAL contributes the language-depth geometry view, especially the layerwise spectral exponent and transport-cost interpretation. Neural FOXP2 contributes the feature-level causal language-control view through selectivity, intervention lift, and stability-aware ranking. Together, these give a coherent framework for measuring both representativeness and interference. fileciteturn0file0 fileciteturn1file0 fileciteturn1file1

## 7. Bayesian extension

A Bayesian version makes the metric more robust because all these quantities are noisy across prompts, layers, and evaluation batches. One simple hierarchical model is

$$
y_{\lambda,\ell,n} \sim \mathcal N(\eta_{\lambda,\ell}, \sigma^2),
$$

with

$$
\eta_{\lambda,\ell} = a_\ell + u_\lambda + u_{\lambda,\ell}.
$$

Here:

- \(a_\ell\) is the global layer effect
- \(u_\lambda\) is the language effect
- \(u_{\lambda,\ell}\) is the language-layer interaction

This lets you estimate posterior means and credible intervals for \(R_\lambda\) and \(I_{\mu \to \lambda}\). It also gives a principled way to compare languages when sample sizes differ or when prompt variance is high.

## 8. Recommended experimental protocol

A strong evaluation should include three checks.

### 8.1 In-language strength

Measure whether each language has high representativeness on its own prompts.

### 8.2 Cross-language leakage

Swap adapters or use soft mixtures and measure how much behavior and geometry move away from the correct language route.

### 8.3 Stability under resampling

Bootstrap prompts and recompute all scores. If the score is unstable, it should not be treated as a strong claim.

## 9. Practical interpretation

The metric says that a language is well represented only when it is:

- internally distinct in geometry
- causally supported by language-selective features
- behaviorally available under neutral prompts
- resistant to interference from other languages

That is the combination you want for a robust multilingual adapter design.

## 10. Final summary

The proposed metric is:

- **representativeness** \(R_\lambda\)
- **interference** \(I_{\mu \to \lambda}\)
- **composite score** \(\mathrm{LRIS}_\lambda\)

This gives you a clean scientific way to ask whether a language is not just improved, but actually represented in a stable and language-specific way inside a shared multilingual backbone.

### One-line takeaway

A language is well represented if it has a stable geometric signature, causal language-specific features, and strong neutral-prompt defaultness, while remaining resistant to leakage from other languages.
