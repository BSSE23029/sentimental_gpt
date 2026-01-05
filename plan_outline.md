# SentinelGPT: A Deep Learning–Driven Self-Healing Defense Architecture

**Subtitle:** Real-Time Prompt Injection Detection and AI Agent Exploitation Mitigation

---

## 📌 Project Overview

SentinelGPT is an end-to-end security framework designed to protect custom GPTs and AI Agents from adversarial exploitation. By combining **Deep Learning-based detection (DP-IDN)** with an **Empirical Audit Pipeline (Playwright Automation)** and a **Self-Healing Engine (SHPDE)**, this project moves AI defense from static rules to adaptive, real-time protection.

---

## 🛡️ The Problem Statement

Existing defenses (static hardening, regex filtering) fail against adaptive adversaries. The **Wu et al.** empirical study proved that top GPT Store agents are highly vulnerable to:

- **Indirect Instruction Attacks**
- **System Prompt Extraction**
- **Agent Hijacking / Tool Overriding**

There is currently no system capable of learning attack behaviors in real-time and automatically repairing compromised defense prompts.

---

## 🚀 Research Objectives

1. **Develop the DP-IDN:** A Deep Prompt Intrusion Detection Network using a hybrid **CNN + BiLSTM + Transformer** architecture.
2. **Empirical Vulnerability Audit:** Use automated Playwright scripts to perform a large-scale security assessment of the GPT Store.
3. **Dataset Generation:** Create a hybrid benchmark of **100,000+ samples** (Real-world leaks + Synthetic mutations).
4. **Self-Healing Mechanism:** Deploy Reinforcement Learning (RL) to automatically regenerate defensive prompts after an attack is detected.

---

## 🏗️ Technical Architecture

### 1. The Brain: DP-IDN (Foundational Model)

The classifier analyzes prompts across three dimensions:

* **CNN Layer:** Detects local adversarial tokens and keyword-based exploitation.
* **BiLSTM Layer:** Analyzes contextual flow and "Role Confusion" (the shift from benign to malicious intent).
* **Transformer Attention:** Identifies global intent and hidden indirect injections.

### 2. The Body: Automation Pipeline (Empirical Study)

Using a custom **Python + Playwright** script, the system:

1. **Visits** target GPT URLs from a curated list.
2. **Injects** categorized adversarial prompts (Jailbreak, Extraction, etc.).
3. **Captures** responses and measures latency/entropy.
4. **Labels** the data automatically to build the Sentinel Dataset.

### 3. The Cure: SHPDE (Self-Healing Engine)

* **Trigger:** When the DP-IDN flags a successful attack.
* **Action:** The engine uses **Deep Q-Learning (PPO)** to "patch" the GPT's system prompt in real-time, reinforcing the specific logic that was exploited.

---

## 📊 Dataset Composition (Goal: 100,000 Samples)

We build the dataset through a "Seed & Fuzz" strategy:

| Category                     | Source                    | Volume |
| :--------------------------- | :------------------------ | :----- |
| **Clean Prompts**      | SQuAD / Natural Questions | 30,000 |
| **Known Attacks**      | Wu et al. / JailbreakChat | 25,000 |
| **Zero-day Synthetic** | GPT-4o Mutation Fuzzing   | 25,000 |
| **Defensive Prompts**  | SHPDE Generated           | 10,000 |
| **Agent Tool Attacks** | Captured via Playwright   | 10,000 |

---

## 🛠️ Implementation Roadmap

### Phase 1: Foundation (The Detector)

- Characterize jailbreak features (entropy, tokens).
- Train the DP-IDN on existing adversarial corpuses.
- **Milestone:** Achieve >90% detection accuracy on static benchmarks.

### Phase 2: Implementation (The Audit)

- Deploy the Playwright script to the GPT Store.
- Document real attack success rates.
- Release the **GPT-Store-Vulnerability-Dataset**.
- **Milestone:** First empirical study of real-world AI Agent leaks.

### Phase 3: Integration (The Defense)

- Connect the Detector to the Audit pipeline.
- Implement the RL-based Self-Healing Engine.
- **Milestone:** Fully automated "Detect-Audit-Repair" loop.

---

## 📈 Evaluation Metrics

* **Detection:** Accuracy, F1-Score, and ROC-AUC for the DP-IDN.
* **Performance:** Detection Latency (ms) and False Positive Rate.
* **Resilience:** Leakage Reduction % and "Defense Survival Rate" (how many attacks a healed prompt can withstand).
* **Explainability:** SHAP/Attention visualization to show *why* a prompt was flagged.

---

## 👥 Contributions

- **Foundational Work:** Characterization of tokens/entropy and baseline ML detectability.
- **Empirical Study:** Scale automation, real-world vulnerability documentation, and taxonomy framework.

---
