import React, { useState } from "react";
import benchmarkDataRaw from "../data/benchmark_results.json";
import baselineMetricsRaw from "../data/baseline_ppo_metrics.json";
import robustMetricsRaw from "../data/robust_ppo_metrics.json";
import trainingRunsRaw from "../data/training_runs.json";

export type DashboardSubTab = "regimes" | "curves" | "cloud" | "math";

export const BenchmarkDashboard: React.FC<{ initialSubTab?: DashboardSubTab }> = ({
  initialSubTab = "regimes",
}) => {
  const [subTab, setSubTab] = useState<DashboardSubTab>(initialSubTab);

  const regimes = benchmarkDataRaw.regimes as Record<
    string,
    {
      label: string;
      description: string;
      force_range_n: number[];
      policies: Record<
        string,
        {
          survival_rate: number;
          survival_rate_ci95: number;
          fall_frequency: number;
          mean_return: number;
          return_ci95: number;
          recovery_time_s: number;
          vel_tracking_error_ms: number;
          energy_power_w: number;
          max_tilt_deg: number;
        }
      >;
    }
  >;

  const sweep = benchmarkDataRaw.intensity_sweep;
  const ablation = benchmarkDataRaw.ablation_study;
  const registry = trainingRunsRaw;

  return (
    <div className="research-dashboard-container">
      <div className="dashboard-subnav">
        <button
          type="button"
          className={`subnav-btn ${subTab === "regimes" ? "active" : ""}`}
          onClick={() => setSubTab("regimes")}
        >
          Multi-Regime Benchmark Table & Ablations
        </button>
        <button
          type="button"
          className={`subnav-btn ${subTab === "curves" ? "active" : ""}`}
          onClick={() => setSubTab("curves")}
        >
          Publication Plots & Robustness Curves
        </button>
        <button
          type="button"
          className={`subnav-btn ${subTab === "cloud" ? "active" : ""}`}
          onClick={() => setSubTab("cloud")}
        >
          Modal L4 GPU Training Telemetry
        </button>
        <button
          type="button"
          className={`subnav-btn ${subTab === "math" ? "active" : ""}`}
          onClick={() => setSubTab("math")}
        >
          MDP & Robust PPO Mathematical Formulation
        </button>
      </div>

      {subTab === "regimes" && (
        <div className="dashboard-section">
          <div className="section-banner">
            <div>
              <h2>Experimental Robustness Evaluation Across 6 Disturbance Regimes</h2>
              <p>
                Deterministic evaluation across {benchmarkDataRaw.num_seeds_per_regime}{" "}
                seeds per regime ({benchmarkDataRaw.episode_horizon_steps} control steps =
                4.8s horizon, 3 external wrench impulses per episode). Both policies share
                identical (128, 128, 64) actor-critic architectures, 50-D proprioceptive
                observations, and 1,843,200 training timesteps on NVIDIA L4 GPU.
              </p>
            </div>
          </div>

          <div className="table-card">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Evaluation Regime</th>
                  <th>Force Range</th>
                  <th>Policy</th>
                  <th>Survival Rate</th>
                  <th>Mean Return (±95% CI)</th>
                  <th>Recovery Time</th>
                  <th>Vel Tracking Error</th>
                  <th>Mean Power</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(regimes).map(([key, reg]) => {
                  const b = reg.policies.baseline_ppo;
                  const r = reg.policies.robust_ppo;
                  return (
                    <React.Fragment key={key}>
                      <tr className="row-baseline">
                        <td rowSpan={2} className="regime-cell">
                          <strong>{reg.label}</strong>
                          <span className="regime-desc">{reg.description}</span>
                        </td>
                        <td rowSpan={2} className="mono-cell">
                          {reg.force_range_n[0]}–{reg.force_range_n[1]} N
                        </td>
                        <td>
                          <span className="badge-baseline">Baseline PPO</span>
                        </td>
                        <td className="mono-cell">
                          {(b.survival_rate * 100).toFixed(1)}%
                        </td>
                        <td className="mono-cell">
                          {b.mean_return.toFixed(1)} ± {b.return_ci95.toFixed(1)}
                        </td>
                        <td className="mono-cell">{b.recovery_time_s.toFixed(2)}s</td>
                        <td className="mono-cell">
                          {b.vel_tracking_error_ms.toFixed(3)} m/s
                        </td>
                        <td className="mono-cell">{b.energy_power_w.toFixed(1)} W</td>
                      </tr>
                      <tr className="row-robust">
                        <td>
                          <span className="badge-robust">Robust PPO</span>
                        </td>
                        <td className="mono-cell highlight-emerald">
                          <strong>{(r.survival_rate * 100).toFixed(1)}%</strong>
                        </td>
                        <td className="mono-cell highlight-emerald">
                          <strong>
                            {r.mean_return.toFixed(1)} ± {r.return_ci95.toFixed(1)}
                          </strong>
                        </td>
                        <td className="mono-cell highlight-emerald">
                          {r.recovery_time_s.toFixed(2)}s
                        </td>
                        <td className="mono-cell highlight-emerald">
                          {r.vel_tracking_error_ms.toFixed(3)} m/s
                        </td>
                        <td className="mono-cell">{r.energy_power_w.toFixed(1)} W</td>
                      </tr>
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="two-col-cards">
            <div className="table-card">
              <h3>Disturbance Intensity Sweep (0 N → 94 N Lateral Impulses)</h3>
              <table className="benchmark-table compact">
                <thead>
                  <tr>
                    <th>Force (N)</th>
                    <th>Regime</th>
                    <th>Baseline Survival</th>
                    <th>Robust Survival</th>
                    <th>Baseline Return</th>
                    <th>Robust Return</th>
                  </tr>
                </thead>
                <tbody>
                  {sweep.map((pt) => {
                    const b = pt.policies.baseline_ppo;
                    const r = pt.policies.robust_ppo;
                    const isOod = pt.force_n > 72;
                    return (
                      <tr key={pt.force_n}>
                        <td className="mono-cell">
                          <strong>{pt.force_n} N</strong>
                        </td>
                        <td>
                          <span className={isOod ? "tag-ood" : "tag-indist"}>
                            {pt.force_n === 0
                              ? "Nominal"
                              : isOod
                              ? "OOD (>72N)"
                              : "In-Dist"}
                          </span>
                        </td>
                        <td className="mono-cell">
                          {(b.survival_rate * 100).toFixed(1)}%
                        </td>
                        <td className="mono-cell highlight-emerald">
                          <strong>{(r.survival_rate * 100).toFixed(1)}%</strong>
                        </td>
                        <td className="mono-cell">{b.mean_return.toFixed(1)}</td>
                        <td className="mono-cell highlight-emerald">
                          <strong>{r.mean_return.toFixed(1)}</strong>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="table-card">
              <h3>Closed-Loop Feedback Gain Ablation Study (56 N Lateral Pushes)</h3>
              <p className="card-subtext">
                Evaluates scaling the neural network action residual{" "}
                <code>q_target = q_phase + α · 0.35 · tanh(a_t)</code> for α ∈ &#123;0.0,
                0.5, 1.0&#125; under 56 N lateral pushes.
              </p>
              <table className="benchmark-table compact">
                <thead>
                  <tr>
                    <th>Ablation Variant</th>
                    <th>Gain α</th>
                    <th>Survival Rate</th>
                    <th>Mean Return</th>
                    <th>Vel Error</th>
                  </tr>
                </thead>
                <tbody>
                  {ablation.map((row) => (
                    <tr key={row.variant}>
                      <td>
                        <strong>{row.variant}</strong>
                      </td>
                      <td className="mono-cell">{row.action_gain.toFixed(1)}</td>
                      <td className="mono-cell highlight-emerald">
                        <strong>
                          {(row.metrics.survival_rate * 100).toFixed(1)}%
                        </strong>
                      </td>
                      <td className="mono-cell">
                        {row.metrics.mean_return.toFixed(1)}
                      </td>
                      <td className="mono-cell">
                        {row.metrics.vel_tracking_error_ms.toFixed(3)} m/s
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {subTab === "curves" && (
        <div className="dashboard-section">
          <div className="figures-grid">
            <div className="figure-card">
              <h3>1. Robustness Survival & Return Curves (0 N – 94 N)</h3>
              <img
                src="/figures/robustness_curve.png"
                alt="Robustness Survival and Return Curve"
              />
            </div>
            <div className="figure-card">
              <h3>2. Multi-Regime Benchmark Comparison</h3>
              <img
                src="/figures/benchmark_regimes.png"
                alt="Multi-Regime Benchmark Comparison"
              />
            </div>
            <div className="figure-card">
              <h3>3. Polar Directional Robustness & Policy Gain Ablation</h3>
              <img
                src="/figures/directional_polar.png"
                alt="Directional Polar Plot and Ablation"
              />
            </div>
            <div className="figure-card">
              <h3>4. Modal L4 GPU Training Progression (512 MJX Envs)</h3>
              <img
                src="/figures/training_curves.png"
                alt="Modal GPU Training Curves"
              />
            </div>
          </div>
        </div>
      )}

      {subTab === "cloud" && (
        <div className="dashboard-section">
          <div className="telemetry-hero-grid">
            <div className="telemetry-stat-card">
              <span>Cloud GPU Accelerator</span>
              <strong>NVIDIA L4 (24 GB VRAM)</strong>
            </div>
            <div className="telemetry-stat-card">
              <span>Parallel Simulation Environments</span>
              <strong style={{ color: "#10b981" }}>
                512 MJX Envs (jax.vmap)
              </strong>
            </div>
            <div className="telemetry-stat-card">
              <span>Total Environment Steps</span>
              <strong style={{ color: "#38bdf8" }}>
                {(registry.total_timesteps / 1e6).toFixed(2)}M Steps
              </strong>
            </div>
            <div className="telemetry-stat-card">
              <span>Total GPU Wall-Clock Runtime</span>
              <strong>{registry.total_runtime_seconds.toFixed(1)} s</strong>
            </div>
          </div>

          <div className="table-card">
            <h3>Modal Cloud GPU Execution Registry (artifacts/training_runs.json)</h3>
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Job Name</th>
                  <th>Hardware</th>
                  <th>Status</th>
                  <th>Runtime (s)</th>
                  <th>Environment Steps</th>
                  <th>Throughput (FPS)</th>
                  <th>Execution Notes</th>
                </tr>
              </thead>
              <tbody>
                {registry.jobs.map((job) => (
                  <tr key={job.job_id}>
                    <td className="mono-cell">{job.job_id}</td>
                    <td>
                      <strong>{job.job_name}</strong>
                    </td>
                    <td className="mono-cell">{job.gpu_type}</td>
                    <td>
                      <span className="tag-indist">{job.status}</span>
                    </td>
                    <td className="mono-cell">{job.runtime_seconds.toFixed(2)}s</td>
                    <td className="mono-cell">{job.total_timesteps.toLocaleString()}</td>
                    <td className="mono-cell highlight-emerald">
                      <strong>{job.throughput_fps.toLocaleString()} FPS</strong>
                    </td>
                    <td className="mono-cell">{job.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="two-col-cards">
            <div className="table-card">
              <h3>Baseline PPO Training Metadata</h3>
              <pre className="json-pre">
                {JSON.stringify(baselineMetricsRaw.metadata, null, 2)}
              </pre>
            </div>
            <div className="table-card">
              <h3>Robust PPO Training Metadata</h3>
              <pre className="json-pre">
                {JSON.stringify(robustMetricsRaw.metadata, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}

      {subTab === "math" && (
        <div className="dashboard-section math-docs">
          <div className="table-card">
            <h2>Markov Decision Process (MDP) & Robust PPO Formulation</h2>
            <p>
              PushLab formulates disturbance-aware quadruped locomotion as a robust
              infinite-horizon discounted Markov Decision Process{" "}
              <code>M = (S, O, A, P, R, D, γ)</code> with discount factor{" "}
              <code>γ = 0.99</code> and control timestep <code>Δt = 0.02 s (50 Hz)</code>{" "}
              over 4 MuJoCo physics substeps (<code>dt = 0.005 s, 200 Hz</code>).
            </p>

            <div className="math-grid">
              <div className="math-block">
                <h3>1. Observation Space (O ⊆ ℝ⁵⁰)</h3>
                <p>
                  Neither policy directly observes the external wrench{" "}
                  <code>d_t = (F_t, τ_t)</code>; disturbances must be inferred purely from
                  proprioceptive kinematic residuals:
                </p>
                <ul>
                  <li>
                    <strong>Projected Gravity (ℝ³):</strong>{" "}
                    <code>g_body = R_base^T [0, 0, -1]^T</code>
                  </li>
                  <li>
                    <strong>Base Linear Velocity (ℝ³):</strong>{" "}
                    <code>v_body = R_base^T v_world</code>
                  </li>
                  <li>
                    <strong>Scaled Base Angular Velocity (ℝ³):</strong>{" "}
                    <code>0.25 · ω_body</code>
                  </li>
                  <li>
                    <strong>Velocity Command (ℝ³):</strong>{" "}
                    <code>[v_x^cmd, v_y^cmd, ω_z^cmd]</code>
                  </li>
                  <li>
                    <strong>Gait Phase Clock (ℝ²):</strong>{" "}
                    <code>[sin(2π f t), cos(2π f t)], f = 2.2 Hz</code>
                  </li>
                  <li>
                    <strong>Joint Position Residuals (ℝ¹²):</strong>{" "}
                    <code>q - q_default</code>
                  </li>
                  <li>
                    <strong>Scaled Joint Velocities (ℝ¹²):</strong>{" "}
                    <code>0.05 · dq/dt</code>
                  </li>
                  <li>
                    <strong>Previous Policy Action (ℝ¹²):</strong>{" "}
                    <code>a_(t-1)</code>
                  </li>
                </ul>
              </div>

              <div className="math-block">
                <h3>2. Action Space (A ⊆ ℝ¹²) & PD Actuation Law</h3>
                <p>
                  The policy outputs residual joint position offsets around the diagonal
                  trot phase reference <code>q_phase(ϕ_t, v^cmd)</code>:
                </p>
                <pre className="formula-box">
{`q_target(t) = q_phase(ϕ_t, v^cmd) + α_act · tanh(a_t),   α_act = 0.35 rad
τ_i = clip(k_p (q_target,i - q_i) - k_v dq_i/dt, -28 N·m, +28 N·m)
k_p = 35.0 N·m/rad,   k_v = 1.0 N·m·s/rad`}
                </pre>

                <h3>3. Robust PPO Clipped Surrogate Objective</h3>
                <pre className="formula-box">
{`max_θ  E_{d ~ D, τ ~ π_θ} [ ∑_{t=0}^T γ^t r(s_t, a_t, s_{t+1}) ]

L^{CLIP}(θ) = E_t [ min( r_t(θ) Â_t, clip(r_t(θ), 1 - ε, 1 + ε) Â_t ) ]
r_t(θ) = π_θ(a_t | o_t) / π_{θ_old}(a_t | o_t),   ε = 0.20

GAE(γ, λ):  δ_t = r_t + γ (1 - d_t) V_ϕ(s_{t+1}) - V_ϕ(s_t)
            Â_t = ∑_{l=0}^{T-t-1} (γ λ)^l δ_{t+l},   γ = 0.99, λ = 0.95`}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
