
# Model Architecture Description

## 1. Input Space

Let the input sequence be:

x = (x1, x2, ..., x128), where xi ∈ Z

So:

x ∈ Z^128

Each xi is a discrete token index.

---

## 2. Embedding Layer

**Embedding matrix:**

We ∈ R^(30522 × 128)

**Embedding lookup:**

ei = We[xi], ei ∈ R^128

**Stacked embeddings:**

E =
[
e1
e2
⋮
e128
]

E ∈ R^(128 × 128)

So the embedding function is:

E(x) = E

---

## 3. 1D Convolution (Local Feature Extraction)

**Convolution kernel:**

Wc ∈ R^(128 × 128 × 3)
bc ∈ R^128

For position t and output channel k:

C[t, k] = σ( Σ(i=1→128) Σ(j=−1→1) Wc[k, i, j] · E[t+j, i] + bc[k] )

**Resulting feature map:**

C ∈ R^(128 × 128)

---

## 4. BiLSTM (Sequential Dependence)

**Forward LSTM:**

→h_t = LSTM_f(C_t, →h_(t−1))

**Backward LSTM:**

←h_t = LSTM_b(C_t, ←h_(t+1))

**Concatenation:**

h_t = [→h_t ; ←h_t]

If each direction has hidden size 128:

h_t ∈ R^256

**Weight example:**

W_ih ∈ R^(512 × 128)

---

## 5. Linear Projection

**Projection matrix:**

Wp ∈ R^(768 × 256)
bp ∈ R^768

**Projection:**

p_t = Wp · h_t + bp

---

## 6. Tensor Fusion (Concatenation)

Let:

z = ⊕

{ E, C, p }

Explicitly:

z = [ e_t ; C_t ; p_t ]

**Resulting fused feature:**

z ∈ R^256

---

## 7. Dimension Reduction

**Dense layer:**

Wd ∈ R^(64 × 256)
bd ∈ R^64

a = ReLU(Wd · z + bd)

So:

a ∈ R^64

---

## 8. Output Layer (Probability)

**Final classifier:**

ŷ = σ(wᵀ · a + b)

where:

w ∈ R^64
b ∈ R

and:

ŷ ∈ [0, 1]

---

## Compact End-to-End Form

ŷ = σ(
wᵀ · ReLU(
Wd · [ E(x) ; C(E(x)) ; P(L(C(E(x)))) ] + bd
) + b
)

It’s ugly, but that’s what real models look like when you stop pretending they’re elegant.
