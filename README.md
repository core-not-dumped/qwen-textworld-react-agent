# 🤖 Qwen TextWorld ReAct Agent 🤖

This repository trains a **Qwen3-based LLM agent** to solve **TextWorld** environments.<br>
using an **InstructGPT-style pipeline** with **SFT and GRPO**.<br>
**LoRA** is applied for memory-efficient fine-tuning, and **ReAct prompting** is used to evaluate the effect of explicit reasoning on agent performance.

---

## Training Pipeline

### 1. Teacher Forcing (Recommended)

Teacher forcing is used to initialize the model with valid interaction patterns and action formats.

This stage helps to:
- Reduce invalid actions
- Improve early exploration
- Stabilize reinforcement learning

python train_teacher_forcing_main.py

---

### 2. GRPO Reinforcement Learning

The agent is trained using **GRPO**, a PPO-style objective that:

- Samples multiple rollouts per prompt
- Normalizes rewards within each group
- Updates the policy using log-probability ratios
- Does not rely on advantage estimation or a value network

python train_GRPO_main.py

---

## Evaluation

After training, the agent can be evaluated using:

python test_main.py

The evaluation:
- Runs the trained agent in TextWorld environments
- Logs interaction trajectories
- Saves example outputs in `test_result.txt`

---

## Dependencies

- Python 3.13.11
- torch 2.10.0+cu126
- transformers 4.57.6
- textworld 1.6.2

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.