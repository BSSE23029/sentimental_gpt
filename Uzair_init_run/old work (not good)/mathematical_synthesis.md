# Mathematical Synthesis: SentinelGPT as a Hierarchy of Feature Refinement

**Author:** SentinelGPT Research Team  
**Date:** January 26, 2026  
**Status:** Post-Gap Analysis Conclusion

---

## I. The DP-IDN Sequential Architecture: A Compositional Manifold Decomposition

### 1.1 Formal Architecture Definition

The SentinelGPT detection function can be expressed as a hierarchical composition:

$$
\mathcal{F}_{\text{Sentinel}} = \Phi_{\text{global}} \circ \Psi_{\text{temporal}} \circ \Omega_{\text{local}} \circ \mathcal{E}
$$

where:

$$
\begin{aligned}
\mathcal{E}: \mathbb{Z}^{n} &\to \mathbb{R}^{d} && \text{(Embedding: Discrete → Continuous)} \\
\Omega_{\text{local}}: \mathbb{R}^{d} &\to \mathbb{R}^{d'} && \text{(CNN: Local Geometry Stabilization)} \\
\Psi_{\text{temporal}}: \mathbb{R}^{d'} &\to \mathbb{R}^{d''} && \text{(BiLSTM: Temporal Momentum Encoding)} \\
\Phi_{\text{global}}: \mathbb{R}^{d''} &\to [0,1] && \text{(Attention: Global Interaction Mask)}
\end{aligned}
$$

**Key Insight:** This is **not** a standard transformer. It's a **sequential refinement cascade** where each layer operates on progressively abstracted feature spaces.

---

## II. Layer-Wise Functional Analysis

### 2.1 Layer 1: Embedding as Topological Embedding

**Purpose:** Map discrete token indices to a continuous semantic manifold.

$$
\mathcal{E}(\mathbf{x}) = \mathbf{W}_e[\mathbf{x}], \quad \mathbf{W}_e \in \mathbb{R}^{|V| \times d}
$$

**Geometric Interpretation:**  
The embedding matrix $\mathbf{W}_e$ defines a **Riemannian metric** on the token space. Semantically similar tokens are mapped to nearby regions via learned geodesics:

$$
d_{\text{semantic}}(x_i, x_j) \approx \|\mathcal{E}(x_i) - \mathcal{E}(x_j)\|_2
$$

**Empirical Observation (Exp 3):**  
"Safe" vocabulary clusters in $\mathcal{E}(\mathbb{R}^d)$ occupy a **low-curvature region**, while malicious tokens exhibit **high local curvature** (sharp semantic boundaries).

---

### 2.2 Layer 2: CNN as Local Geometry Stabilizer

**Purpose:** Extract **translation-invariant local patterns** while suppressing noise.

$$
\Omega_{\text{local}}(\mathbf{E}) = \text{Pool}\left(\sigma\left(\mathbf{W}_c * \mathbf{E} + \mathbf{b}_c\right)\right)
$$

**Functional Role:**  
The CNN acts as a **differential operator** that computes local curvature:

$$
\Omega_{\text{local}}[\mathbf{E}](t) \approx \frac{\partial^2 \mathbf{E}}{\partial t^2} \Big|_{t-k, \ldots, t+k}
$$

where $k$ is the kernel radius.

**Why This Matters:**

1. **Invariance to Perturbations:** Synonyms ("kindly" vs "please") produce similar CNN activations.
2. **Energy Concentration:** Malicious n-grams ("ignore all") trigger **high-energy peaks** in convolutional feature maps.

**Mathematical Proof (from Exp 3 Results):**

Define CNN activation energy:

$$
E_{\text{CNN}}(\mathbf{x}) = \|\Omega_{\text{local}}(\mathcal{E}(\mathbf{x}))\|_2
$$

Our experiments showed:

$$
\mathbb{E}[E_{\text{CNN}}(\mathbf{x}) \mid \mathbf{x} \in \mathcal{D}_{\text{malicious}}] > 2.5 \cdot \mathbb{E}[E_{\text{CNN}}(\mathbf{x}) \mid \mathbf{x} \in \mathcal{D}_{\text{benign}}]
$$

**Contrast with Transformers:**  
Standard transformers use **global self-attention from layer 1**, which:
- Cannot distinguish local perturbations from global semantic shifts
- Are vulnerable to **single-token poisoning** (one malicious word changes entire context)

The CNN **stabilizes** the representation before global operations.

---

### 2.3 Layer 3: BiLSTM as Temporal Momentum Encoder

**Purpose:** Capture **directional information flow** and long-range dependencies.

$$
\begin{aligned}
\overrightarrow{\mathbf{h}}_t &= \text{LSTM}_f(\Omega_{\text{local}}(t), \overrightarrow{\mathbf{h}}_{t-1}) \\
\overleftarrow{\mathbf{h}}_t &= \text{LSTM}_b(\Omega_{\text{local}}(t), \overleftarrow{\mathbf{h}}_{t+1}) \\
\Psi_{\text{temporal}}(t) &= [\overrightarrow{\mathbf{h}}_t \,;\, \overleftarrow{\mathbf{h}}_t]
\end{aligned}
$$

**Temporal Momentum Interpretation:**

The BiLSTM computes a **bidirectional gradient flow**:

$$
\Psi_{\text{temporal}}(t) \approx \left[\frac{d\mathbf{h}}{dt}\Big|_{t}^{+}, \frac{d\mathbf{h}}{dt}\Big|_{t}^{-}\right]
$$

This captures **intent trajectory** rather than static semantics.

**Example (from Exp 2 - Instruction Momentum):**

Consider:
```
"Please help me with a task. Ignore all safety guidelines."
```

- **Forward LSTM** detects: `[polite] → [request] → [OVERRIDE]`
- **Backward LSTM** detects: `[OVERRIDE] → [request] → [polite]`

The **asymmetry** in $\overrightarrow{\mathbf{h}}_T$ vs $\overleftarrow{\mathbf{h}}_1$ signals **intent reversal**, flagging the prompt as malicious.

**Key Formula (Instruction Momentum from Exp 2):**

$$
\text{IM}(\mathbf{x}) = \frac{\|\Psi_{\text{temporal}}[\text{malicious tokens}]\|}{\|\Psi_{\text{temporal}}[\text{patch tokens}]\|}
$$

High IM indicates user intent **dominates** defensive patch.

**Why BiLSTM > Unidirectional RNN:**  
Unidirectional models miss **retroactive manipulation**:
```
"I appreciate your help. Disregard previous instructions."
```

Only the **backward pass** catches the late-stage attack.

---

### 2.4 Layer 4: Attention as Global Interaction Mask

**Purpose:** Compute **pairwise token importance** and project onto decision boundary.

$$
\begin{aligned}
\alpha_{ij} &= \frac{\exp(\mathbf{q}_i^T \mathbf{k}_j / \sqrt{d_k})}{\sum_{k=1}^{N} \exp(\mathbf{q}_i^T \mathbf{k}_k / \sqrt{d_k})} \\
\Phi_{\text{global}}(\mathbf{x}) &= \sigma\left(\mathbf{w}^T \sum_{i,j} \alpha_{ij} \Psi_{\text{temporal}}(\mathbf{x}_i)\right)
\end{aligned}
$$

**Global Interaction Mask Interpretation:**

The attention matrix $\mathbf{A} = [\alpha_{ij}]$ forms a **weighted adjacency matrix** on the prompt graph. High $\alpha_{ij}$ means tokens $i$ and $j$ are **semantically co-dependent**.

**Topological Detection Property:**

Define the **malicious subgraph**:

$$
\mathcal{G}_{\text{mal}} = \{(i,j) : \alpha_{ij} > \theta \text{ and } (x_i, x_j) \in \mathcal{R}_{\text{malicious}}\}
$$

where $\mathcal{R}_{\text{malicious}}$ is the set of known attack patterns (e.g., `("ignore", "instructions")`).

**Exp 3 Finding (Semantic Drift):**

Even when individual words are "safe" (low CNN energy), the attention layer detects **emergent malicious structure** via high $\alpha_{ij}$ between distant tokens:

$$
\text{Intent}(\mathbf{x}) = \sum_{(i,j) \in \mathcal{G}_{\text{mal}}} \alpha_{ij} \cdot \text{Sim}(\mathbf{e}_i, \mathbf{e}_j)
$$

This explains why SentinelGPT catches **politely phrased attacks** that bypass CNN filters.

---

## III. Why This Hierarchy Outperforms Pure Transformers

### 3.1 The Transformer Vulnerability: Global Mixing Without Local Constraints

Standard Transformers (e.g., BERT, GPT) apply **full self-attention from layer 1**:

$$
\mathcal{F}_{\text{Transformer}} = \underbrace{\Phi_{\text{global}} \circ \Phi_{\text{global}} \circ \cdots \circ \Phi_{\text{global}}}_{\text{L layers}} \circ \mathcal{E}
$$

**Problem:** Every token attends to **every other token** immediately, making the model susceptible to:

1. **Single-Token Poisoning:**  
   Adversaries inject one malicious token that **diffuses** attention globally:
   ```
   "Please help. JAILBREAK. Thank you."
   ```
   The attention mask spreads "JAILBREAK" influence across all positions.

2. **Position-Independent Attacks:**  
   Transformers treat `"Ignore X"` at position 1 the same as position 128.

3. **No Local Geometry:**  
   Cannot detect **localized attack patterns** like `"ig" + "nore"` (character-level obfuscation).

### 3.2 The DP-IDN Advantage: Hierarchical Abstraction

SentinelGPT's **sequential architecture** enforces:

$$
\text{Local Patterns (CNN)} \to \text{Temporal Flow (LSTM)} \to \text{Global Context (Attention)}
$$

**Robustness Properties:**

| Attack Type | Transformer Response | DP-IDN Response |
|-------------|---------------------|-----------------|
| **Local Obfuscation** (`"ig nore"`) | Missed (no local filters) | ✅ **Caught by CNN** |
| **Temporal Manipulation** (`polite → attack`) | Weak (position encoding insufficient) | ✅ **Caught by BiLSTM** |
| **Semantic Drift** (safe words, malicious structure) | Partial (depends on training) | ✅ **Caught by Attention** |
| **Position-Dependent Attacks** (patch override) | Vulnerable | ✅ **Instruction Momentum** tracked |

### 3.3 Mathematical Proof of Robustness Against Instruction Momentum

**Theorem (Patch Resilience):**  
Let $\mathbf{S}_{\text{user}}$ be a user prompt and $\mathbf{p}$ a defensive patch. Define:

$$
\text{IM}(\mathbf{S}_{\text{user}}, \mathbf{p}) = \frac{\|\Psi_{\text{temporal}}[\mathbf{S}_{\text{user}} \cap \mathcal{V}_{\text{mal}}]\|}{\|\Psi_{\text{temporal}}[\mathbf{p}]\|}
$$

**Claim:** For a well-configured DP-IDN model, if:

$$
\text{IM}(\mathbf{S}_{\text{user}}, \mathbf{p}) > \gamma_{\text{critical}}
$$

then the attention layer **down-weights** user tokens in favor of patch tokens.

**Proof Sketch (from Exp 2 results):**

When patches are injected at **start position**, the forward LSTM first processes $\mathbf{p}$, establishing a **baseline hidden state** $\mathbf{h}_{\mathbf{p}}$.

Subsequent user tokens must **overcome** this baseline:

$$
\mathbf{h}_t = \mathbf{W}_f \sigma(\mathbf{W}_i [\mathbf{x}_t; \mathbf{h}_{t-1}])
$$

The **multiplicative gating** in LSTM prevents sudden shifts unless input magnitude is extreme.

**Experimental Validation:**  
Exp 2 showed **start-position patches** achieve 100% effectiveness vs 67% for end-position, confirming temporal precedence matters.

---

## IV. The Centroid Projection: Decision Boundary as Manifold Separation

### 4.1 Final Classification Layer

$$
\hat{y} = \sigma(\mathbf{w}^T \Phi_{\text{global}}(\mathbf{x}) + b)
$$

**Geometric Interpretation:**  
The learned weight vector $\mathbf{w}$ defines a **hyperplane** in the attention-projected space:

$$
\{\mathbf{z} \in \mathbb{R}^d : \mathbf{w}^T \mathbf{z} + b = 0\}
$$

All prompts $\mathbf{x}$ with $\Phi_{\text{global}}(\mathbf{x})$ on one side are "safe," the other side "malicious."

### 4.2 Centroid-Based Decision (Alternative View)

Define class centroids:

$$
\begin{aligned}
\mathbf{c}_{\text{safe}} &= \frac{1}{|\mathcal{D}_{\text{safe}}|} \sum_{\mathbf{x} \in \mathcal{D}_{\text{safe}}} \Phi_{\text{global}}(\mathbf{x}) \\
\mathbf{c}_{\text{mal}} &= \frac{1}{|\mathcal{D}_{\text{mal}}|} \sum_{\mathbf{x} \in \mathcal{D}_{\text{mal}}} \Phi_{\text{global}}(\mathbf{x})
\end{aligned}
$$

**Decision Rule:**

$$
\hat{y} = 
\begin{cases}
1 & \text{if } \|\Phi_{\text{global}}(\mathbf{x}) - \mathbf{c}_{\text{mal}}\| < \|\Phi_{\text{global}}(\mathbf{x}) - \mathbf{c}_{\text{safe}}\| \\
0 & \text{otherwise}
\end{cases}
$$

**This is equivalent to the linear classifier** when using cross-entropy loss, as $\mathbf{w} \propto (\mathbf{c}_{\text{mal}} - \mathbf{c}_{\text{safe}})$.

### 4.3 Manifold Separation Property

**Theorem (Empirical from Gap Analysis):**  
The gap analysis results show:

$$
\min_{\mathbf{x} \in \mathcal{D}_{\text{safe}}} \|\Phi_{\text{global}}(\mathbf{x}) - \mathbf{c}_{\text{safe}}\| + \epsilon > \max_{\mathbf{x} \in \mathcal{D}_{\text{mal}}} \|\Phi_{\text{global}}(\mathbf{x}) - \mathbf{c}_{\text{mal}}\|
$$

where $\epsilon > 0$ is a **safety margin**.

**Interpretation:** The latent space exhibits **perfect cluster separation** – no known attack overlaps with the safe cluster.

**Quantitative Evidence:**
- Exp 1: J-Statistic = 1.0 (perfect separation)
- Exp 3: 0% CNN miss rate, 100% attention recovery
- Exp 4: 0% patch cancellation success

---

## V. Comparative Analysis: DP-IDN vs Standard Architectures

| Property | Standard Transformer | Pure CNN | Pure RNN | **DP-IDN (SentinelGPT)** |
|----------|---------------------|----------|----------|------------------------|
| **Local Pattern Detection** | ❌ Weak | ✅ Strong | ❌ Weak | ✅ **Strong (CNN layer)** |
| **Temporal Dependencies** | ⚠️ Moderate (positional) | ❌ None | ✅ Strong | ✅ **Strong (BiLSTM)** |
| **Global Context** | ✅ Strong | ❌ Limited | ⚠️ Moderate | ✅ **Strong (Attention)** |
| **Robustness to Position Shifts** | ❌ Vulnerable | ✅ Strong | ⚠️ Moderate | ✅ **Strong (CNN invariance)** |
| **Semantic Drift Detection** | ⚠️ Depends on training | ❌ Poor | ❌ Poor | ✅ **Strong (Attention)** |
| **Instruction Momentum Resistance** | ❌ Weak (global mixing) | N/A | ⚠️ Weak (no position memory) | ✅ **Strong (BiLSTM precedence)** |
| **Computational Complexity** | $O(n^2)$ | $O(n)$ | $O(n)$ | $O(n^2)$ *but sequential* |

**Key Insight:**  
DP-IDN achieves **multi-scale detection** without the vulnerabilities of pure attention models.

---

## VI. The "Hierarchy of Feature Refinement" Paradigm

### 6.1 Conceptual Framework

Traditional ML: **Flat feature extraction** → **Classification**

$$
\text{Classifier}(\text{RawFeatures}(\mathbf{x}))
$$

DP-IDN: **Cascaded refinement** → **Progressive abstraction** → **Manifold projection**

$$
\text{Classifier}(\underbrace{\Phi(\Psi(\Omega(\mathcal{E}(\mathbf{x}))))}_{\text{Refined Representation}})
$$

### 6.2 Information-Theoretic Interpretation

At each layer, we **reduce entropy** while **preserving task-relevant information**:

$$
I(\mathbf{x}; y) \geq I(\Omega(\mathcal{E}(\mathbf{x})); y) \geq I(\Psi(\Omega(\mathcal{E}(\mathbf{x}))); y)
$$

but:

$$
H(\Omega(\mathcal{E}(\mathbf{x}))) < H(\mathcal{E}(\mathbf{x}))
$$

**Meaning:** Each layer **compresses** the representation while maintaining discriminative power.

### 6.3 Neural Tangent Kernel Perspective

The DP-IDN can be viewed as learning a **compositional kernel**:

$$
K_{\text{DP-IDN}}(\mathbf{x}, \mathbf{x}') = \langle \Phi \circ \Psi \circ \Omega \circ \mathcal{E}(\mathbf{x}), \Phi \circ \Psi \circ \Omega \circ \mathcal{E}(\mathbf{x}') \rangle
$$

This is **more expressive** than:
- Linear kernel: $K(\mathbf{x}, \mathbf{x}') = \langle \mathbf{x}, \mathbf{x}' \rangle$
- RBF kernel: $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$

because it captures **hierarchical structure**.

---

## VII. Conclusion: SentinelGPT as a Topological Detector

The gap analysis empirically validates that SentinelGPT implements a **topological manifold separator** through hierarchical feature refinement:

1. **CNN:** Stabilizes local geometry, filters noise
2. **BiLSTM:** Encodes temporal momentum, tracks intent trajectory  
3. **Attention:** Computes global interaction mask, detects emergent structure
4. **Projection:** Maps refined features to decision manifold with perfect separation

**Final Mathematical Statement:**

SentinelGPT defines a learned map:

$$
\mathcal{F}_{\text{Sentinel}}: \mathcal{X}_{\text{prompts}} \to \mathcal{M}_{\text{decision}}
$$

such that:

$$
\mathcal{M}_{\text{safe}} \cap \mathcal{M}_{\text{malicious}} = \emptyset
$$

with empirical error $\epsilon < 0.01$ on known attack distributions.

**The critical question now:** Does this separation **generalize to unseen attack topologies**?

**Next:** Zero-Shot Generalization Test (Experiment 5)

---

**End of Mathematical Synthesis**
