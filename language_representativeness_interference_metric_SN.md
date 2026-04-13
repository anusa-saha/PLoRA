### Quantifying language representativeness and cross-language interference

Our PLoRA routes language-specific adapters, all languages still share the same backbone, so interference can occur through shared residual pathways, attention heads, and latent subspaces. To study this effect, I am defining two complementary quantities: a **language representativeness score** that measures how strongly a language is encoded and causally expressed in the model, and a **cross-language interference score** that measures how much one language perturbs another.

Let $\lambda \in \{1,\dots,m\}$ denote a language, $\ell$ a layer, and $S_\lambda$ the sparse support set selected by PLoRA for language $\lambda$. Let $N_\lambda$ be the set of language-specific features or neurons identified by the causal localization stage, and let $\mathcal D_{\mathrm{neutral}}$ be a set of prompts that do not explicitly specify a target language.

Definition: The language representativeness score

$$
R_\lambda = w_1 G_\lambda + w_2 C_\lambda + w_3 D_\lambda,
$$

where

$$
G_\lambda = \frac{1}{|S_\lambda|} \sum_{\ell \in S_\lambda} \Big[z(\alpha_\lambda(\ell)-\bar\alpha(\ell)) + z(-L_\lambda(\ell))\Big],
$$

$$
C_\lambda = \frac{1}{|N_\lambda|} \sum_{j \in N_\lambda} \Big[z(\mathrm{Sel}_j) + z(\mathrm{LiftSlope}_j)\Big],
$$

and

$$
D_\lambda = \mathbb E_{x \sim \mathcal D_{\mathrm{neutral}}}[\Delta M_\lambda(x)].
$$

Here $\alpha_\lambda(\ell)$ is the layerwise spectral exponent, $L_\lambda(\ell)$ is the Fisher-Rao belief transport cost, $\mathrm{Sel}_j$ measures feature selectivity, $\mathrm{LiftSlope}_j$ measures causal lift under intervention, and $\Delta M_\lambda(x)$ measures early-step target-language mass advantage. The function $z(\cdot)$ denotes a standardized score computed across languages or layers.

To measure interference, defining a directed score from language $\mu$ to language $\lambda$:

\[
I_{\mu \to \lambda}
=
\eta_1 \, \mathbb{E}_{x \sim \mathcal{D}_\lambda}
\left[
\mathrm{KL}\!\left(
p^{(\lambda)}(\cdot \mid x)\,\|\,p^{(\mu)}(\cdot \mid x)
\right)
\right]
+
\eta_2 \, \frac{1}{|S_\lambda|}
\sum_{\ell \in S_\lambda}
\frac{
\left\|\Pi_{S_\lambda}\!\left(\Delta W^{(\mu)}_\ell\right)\right\|_F
}{
\left\|\Delta W^{(\mu)}_\ell\right\|_F
}
\]

The first term measures behavioral leakage, that is, how much routing the wrong language adapter changes the output distribution on language $\lambda$ prompts. The second term here measureing the geometry leakage, namely how much of adapter $\mu$'s update lies in the subspace used by language $\lambda$. A single scalar summary can then be formed like:

$$
\mathrm{LRIS}_\lambda = R_\lambda - \gamma \frac{1}{m-1}\sum_{\mu \neq \lambda} I_{\mu \to \lambda},
$$

where higher values indicate stronger representativeness with lower cross-language interference.

---

# Explanation (From my Point of View)

## 1. The necessity of this metric

The PLoRA draft uses language-specific adapters, but the backbone is still shared. That means the model can learn a language-specific route while still preserving a common representational space underneath. In practice, this is basically creating a real possibility of interference.

Interference can appear in several ways. A Hindi adapter may improve Hindi but slightly distort Spanish or Bengali behavior. A low-resource language adapter can share too much with a high-resource language region, so the model appears to represent the language, yet the representation is unstable or not truly distinct. The reverse can also happen: the language may have an adapter, but the backbone may not give that adapter enough room to express a clean linguistic signature.

So my query: 1. "does the adapter improve the language", also, 2. "how uniquely and cleanly is the language represented, and how much does it borrow from other languages present?"

So a metric should capture both representativeness and interference.

## 2. Language representativeness

Language representativeness is the extent to which the model has a clear internal signature for a language. I believe that a language is well represented if three things happen together.

First, the language should show a distinctive internal geometry at the layers where it matters. In the PLoRA draft, this is already aligned with the depth fingerprint idea based on spectral statistics. If a language consistently deviates from the multilingual average in certain layers, that suggests a real language-specific pathway.

Second, this pathway should not just be statistical noise. It should be supported by causal features. Neural FOXP2 is useful here because it separates mere selectivity from actual causal lift. A feature is not enough if it simply correlates with a language. It should also increase the target-language advantage when intervened on.

Third, the language should be behaviorally expressible. If a language is truly represented, then under neutral prompts it should have a stronger early-step prior and should be easier to activate without heavy prompting.

So representativeness is obviosuly a combined notion: geometry, causality, and behavior.

## 3. Extending

A language might have a high spectral deviation or a strong depth fingerprint, but still be entangled with another language. For example, two languages may share script, tokenization patterns, or syntactic structure. Then a large geometric deviation does not necessarily mean a clean language-specific representation. It may only mean that the model has found a joint representation that serves both languages.

That is why the metric should not rely on geometry alone. Geometry is absoulutely necessary, but it is not sufficient. It should be paired with a causal feature score and a behavioral defaultness score.

## 4. The representativeness scores explaining

Defining:

$$
R_\lambda = w_1 G_\lambda + w_2 C_\lambda + w_3 D_\lambda.
$$

Each term has a different role.

### 4.1 Geometry term $G_\lambda$

This term uses the PLoRA and SPINAL-style view of layerwise language structure. The first component, $\alpha_\lambda(\ell)-\bar\alpha(\ell)$, tells us how much language $\lambda$ deviates from the average language at layer $\ell$. The second component, $-L_\lambda(\ell)$, rewards lower belief-transport cost, which means the model settles more smoothly into a language-consistent predictive state.

The standardization $z(\cdot)$ is important because different layers and different quantities are on different scales here.Standardizing makes the geometry term comparable across languages.

A language with high $G_\lambda$ has a strong and structured depth signature, not just a random spike.

### 4.2 Causal term $C_\lambda$

This term is the mechanistic part. It asks whether the language-specific features actually matter under intervention.

The selectivity term $\mathrm{Sel}_j$ asks whether feature $j$ fires more for the target language than for English or other controls. The lift slope term $\mathrm{LiftSlope}_j$ asks whether increasing that feature causes the target-language logit mass to rise.

This follows the logic of Neural FOXP2, where a feature must be both selective and causally effective. A feature that is only selective can be a passenger. A feature that only lifts but is not selective may be too broad or generic. The product-like structure in the paper logic is powerful because it is favouring features that are both language-specific and intervention-relevant.

### 4.3 Behavioral defaultness term $D_\lambda$

This term is checkimg whether the model naturally starts in language $\lambda$ under weak prompting. That matters because some languages may be representationally present yet not readily accessible. If a language has a strong defaultness advantage, then the model is not only storing the language but also treating it as an active option in generation.

To me this is easy to interpret experimentally. Under neutral prompts, the model should assign more early probability mass to the target language if the representativeness score is truly high.

## 5. The Interference

Interference is not just poor performance. It is specifically the unwanted transfer of one language’s route into another language’s behavior or representation.

There are two kinds of interference worth measuring.

### 5.1 Behavioral leakage

What happens when the wrong adapter is used or when multiple adapters interact. If language $\mu$ is routed into a prompt belonging to language $\lambda$, does the output distribution change sharply? If yes, then the two language systems are entangled.

The KL divergence term is useful because it measures full distributional change, not only top-1 changes. It captures subtle shifts in token preferences, which are important in multilingual generation.

### 5.2 Geometry leakage

Behavioral leakage tells us what changed, but not where it came from. Geometry leakage examines whether the adapter update for language $\mu$ lies in the subspace used by language $\lambda$. If the projected Frobenius norm is large, then the same internal directions are being reused across the languages.

To me this is signalling a shared-backbone multilingual model. It is basically defining that the model is not fully disentangled, even if each adapter is different.

## 6. The directed interference score

Interference is often asymmetric (not symmetrical). A high-resource language can influence a low-resource language more than the reverse. Or therefore a language with a larger adapter budget may spill into others more easily.

So better to define

$$
I_{\mu \to \lambda}
$$

rather than a symmetric distance. So from this we can say that Hindi-to-Bengali interference is larger than Bengali-to-Hindi interference.

That I think is much informative than a single undirected overlap number.

## 7. The main composite score

Composite Score:

$$
\mathrm{LRIS}_\lambda = R_\lambda - \gamma \frac{1}{m-1} \sum_{\mu \neq \lambda} I_{\mu \to \lambda}
$$

is designed so that higher the score (Composite Score) the better it is.

It is determining that a language is getting well represented only if it is both strong internally and cleanly separated from other languages. This is useful because sometimes a language score looks impressive on its own, but here the interference matrix reveals that the model is borrowing too much from neighboring languages.

## 8. Bayesian version (extension)

What if we treat all these scores as noisy observations of latent language quality?

For example, say $y_{\lambda,\ell,n}$ denote a layerwise measurement, such as spectral exponent, lift, or token-mass advantage on prompt $n$. Then model

$$
y_{\lambda,\ell,n} \sim \mathcal N(\eta_{\lambda,\ell}, \sigma^2),
$$

with

$$
\eta_{\lambda,\ell} = a_\ell + u_\lambda + u_{\lambda,\ell}.
$$

Here:
- $a_\ell$ is the global layer effect,
- $u_\lambda$ is the language effect,
- $u_{\lambda,\ell}$ is the language-layer interaction.

This is a clean hierarchical model to estimate not just point scores but uncertainty intervals. Then we can report posterior means and credible intervals for $R_\lambda$ and $I_{\mu \to \lambda}$.

This is valuable because multilingual measurements are noisy, prompt-sensitive, and layer-sensitive. A Bayesian model lets us determine if an observed interference effect is robust or just sample noise.

## 9. Experimentally evaluation strategy

A good evaluation protocol should include three tests.

### 9.1 In-language strength

Checking if each language has high representativeness on its own prompts. That means strong geometry, strong causal lift, and strong early-step defaultness.

### 9.2 Cross-language leakage

Swap adapters or apply soft mixtures, then measure how much the output distribution and internal geometry move away from the correct language path.

### 9.3 Stability under resampling

Bootstrap prompts, recompute all scores, and measure variance. If the score changes too much, it is not reliable enough to use as a scientific claim for us.

This part is where I beleive Bayesian credible intervals help a lot.

## 10. The connection I found from PLoRA draft, SPINAL paper and NeuralFOXP2


PLoRA gives us the language-routed, layer-sparse adapter structure and the depth fingerprint view. SPINAL gives us the geometric diagnostic backbone, especially the layerwise spectral and belief-transport perspective. Neural FOXP2 give us the causal and feature-based language steering logic. Together, they can support a metric that is not only descriptive but mechanistic.

## 11. The main takeaway

A good multilingual adapter should not only improve a language. It should make that language internally distinct, causally accessible, and behaviorally default, while minimizing leakage into other languages.

That is what the proposed metric captures I think.

---

# Notation summary

- $R_\lambda$: language representativeness
- $I_{\mu \to \lambda}$: directed interference from language $\mu$ to $\lambda$
- $\mathrm{LRIS}_\lambda$: final composite score
- $G_\lambda$: geometry component
- $C_\lambda$: causal component
- $D_\lambda$: defaultness component

---

# In short:

A language is well represented if it has a stable geometric signature, causal language-specific features, and strong neutral-prompt defaultness, while remaining resistant to leakage from other languages.

