# Roadmap: AgenticSciML for Brain-FWI

Status: **Active**
Owner: Morgan Hough
Focus: Leveraging an all-Opus AgenticSciML Swarm ($100 budget/20 generations) to solve the Brain-FWI Phase 4 bottleneck and push toward Phase 5 physics.

## Current State & The Bottleneck
The `brain-fwi` pipeline successfully generated Phase 0 data and built a solid JAX-based optimization loop. However, Phase 4 (Neural Operator Surrogate) is stalled at the `<1% trace-fidelity gate`. The baseline Classic FNO architecture is plateauing around an 8% relative-L2 error because it lacks the capacity to capture local wave features and sharp interfaces (the skull).

Instead of manual trial-and-error, we are pivoting to an **AgenticSciML-driven discovery process**. The orchestrator, now fully powered by **Claude 3 Opus** and backed by remote execution on **Modal (H100)**, will automatically propose, debate, implement, and validate new architectural designs.

---

## Stage 1: Surrogate Architecture Evolution (Phase 4.5)
**Objective:** Pass the 1% trace-fidelity gate.
**Mechanism:** AgenticSciML evolutionary search over the surrogate architecture space.

1.  **Baseline Stabilization:**
    *   *Done:* Batch averaging/gradient accumulation implemented to smooth optimization.
    *   *Done:* Agentic API Bridge (`prepare.py`) exposed to the swarm.
2.  **Swarm Objectives:**
    *   **U-FNO (UNO):** The swarm will implement and evaluate U-shaped architectures with skip connections to better resolve multi-scale features (skull vs brain).
    *   **Source-Conditioning:** The Critic agent will enforce that the architecture does not just globally pool the wavefield, but conditions explicitly on the transducer geometry (e.g., via learned embeddings or positional encoding).
    *   **Training Dynamics:** The Engineer agent will tune the `lambda_spec` (spectral loss weight) and Adam learning rate schedules (Cosine vs. Exponential decay).
3.  **Success Condition:** The ResultAnalyst reports `brain_rmse` converging and trace relative L2 dropping below 1% on the Modal H100 sandbox.

---

## Stage 2: Physics-Informed Regularization
**Objective:** Improve out-of-distribution robustness (e.g., handling skulls not perfectly represented in the Phase 0 MIDA dataset).
**Mechanism:** AgenticSciML proposing loss function modifications.

1.  **Swarm Objectives:**
    *   **P-FNO (Physics-Informed FNO):** The Proposer agent will draft modifications to `surrogate_loss` that include the physical acoustic wave equation residual.
    *   **Weighting:** The Swarm will sweep the weighting between the data-driven loss (`rel_l2`) and the physics residual to find the optimal trade-off that maintains trace fidelity while preventing non-physical artifacts.

---

## Stage 3: Phase 5 Physics (Density and Attenuation)
**Objective:** Simultaneous inversion of Sound Speed ($c$), Density ($\rho$), and Attenuation ($\alpha$).
**Mechanism:** Expanding the `FWIConfig` search space once the surrogate speedup (100x+) is unlocked.

1.  **The FWI Expansion:**
    *   Currently, FWI only updates $c$. The swarm will implement the inversion of $\rho$ and $\alpha$.
    *   The Critic will ensure that cross-talk between the three parameters is minimized.
2.  **Swarm Objectives:**
    *   **Envelope Loss:** The swarm will deploy Envelope Loss during the early inversion stages to recover large-scale density structure before focusing on sound speed.
    *   **Multi-parameter SIREN:** The swarm will evaluate if a single SIREN with 3 output channels ($c, \rho, \alpha$) outperforms three separate voxel grids.

---

## Operational Mechanics
*   **Orchestration:** `agentsciml/run_discovery.sh` (running locally on macOS).
*   **Budget:** $100 per run.
*   **Swarm Config:** 8 Specialized Agents, all running **Claude 3 Opus** for maximum reasoning depth.
*   **Sandbox:** Seamless dispatch to `Modal (H100)` for all 3D JAX execution.
*   **Metrics:** Evolution favors the minimum `brain_rmse`.