import React from "react";
import { Cpu, Gauge, Footprints, BarChart2 } from "lucide-react";
import { RobotRenderState } from "../physics/quadrupedSimulator";

interface TelemetryPanelProps {
  primaryState: RobotRenderState;
  secondaryState: RobotRenderState | null;
  duelMode: boolean;
  recentHistory: RobotRenderState[];
}

function renderSparkline(
  values: number[],
  minVal: number,
  maxVal: number,
  stroke: string,
  zeroLine = false
): React.ReactNode {
  const w = 250;
  const h = 48;
  if (values.length < 2) {
    return <svg width={w} height={h} className="spark-svg" />;
  }
  const span = Math.max(1e-4, maxVal - minVal);
  const pts = values
    .map((v, idx) => {
      const x = (idx / Math.max(1, values.length - 1)) * w;
      const clamped = Math.max(minVal, Math.min(maxVal, v));
      const y = h - 4 - ((clamped - minVal) / span) * (h - 8);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const zeroY = h - 4 - ((0 - minVal) / span) * (h - 8);

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h}`} className="spark-svg">
      {zeroLine && zeroY >= 0 && zeroY <= h && (
        <line
          x1={0}
          y1={zeroY}
          x2={w}
          y2={zeroY}
          stroke="#334155"
          strokeDasharray="3 3"
          strokeWidth={1}
        />
      )}
      <polyline
        fill="none"
        stroke={stroke}
        strokeWidth={2}
        points={pts}
      />
    </svg>
  );
}

const JOINT_LABELS = [
  "FL Hip",
  "FL Thigh",
  "FL Calf",
  "FR Hip",
  "FR Thigh",
  "FR Calf",
  "RL Hip",
  "RL Thigh",
  "RL Calf",
  "RR Hip",
  "RR Thigh",
  "RR Calf",
];

export const TelemetryPanel: React.FC<TelemetryPanelProps> = ({
  primaryState,
  secondaryState,
  duelMode,
  recentHistory,
}) => {
  const windowSlice = recentHistory.slice(-80);
  const rollSeries = windowSlice.map((s) => s.roll_deg);
  const velErrSeries = windowSlice.map((s) => s.vel_tracking_error);
  const criticSeries = windowSlice.map((s) => s.critic_value);

  const accentColor =
    primaryState.policy_name === "robust_ppo" ? "#10b981" : "#38bdf8";

  return (
    <aside className="telemetry-sidebar">
      {/* 1. Real-Time State & Critic Metrics */}
      <section className="control-card">
        <div className="card-header">
          <Gauge size={15} className="icon-accent" />
          <h3>LIVE RL & DYNAMICS TELEMETRY</h3>
        </div>

        <div className="kpi-grid">
          <div className="kpi-box">
            <span className="kpi-label">Critic Value V(s)</span>
            <strong className="kpi-value" style={{ color: accentColor }}>
              {primaryState.critic_value.toFixed(2)}
            </strong>
          </div>
          <div className="kpi-box">
            <span className="kpi-label">Episode Return</span>
            <strong className="kpi-value">
              {primaryState.cumulative_return.toFixed(1)}
            </strong>
          </div>
          <div className="kpi-box">
            <span className="kpi-label">Vel Error ||v - v*||</span>
            <strong className="kpi-value">
              {primaryState.vel_tracking_error.toFixed(3)} m/s
            </strong>
          </div>
          <div className="kpi-box">
            <span className="kpi-label">Mech. Power</span>
            <strong className="kpi-value">
              {primaryState.mechanical_power_w.toFixed(1)} W
            </strong>
          </div>
          <div className="kpi-box">
            <span className="kpi-label">Upright cos(θ)</span>
            <strong
              className="kpi-value"
              style={{
                color: primaryState.upright_cos < 0.65 ? "#f43f5e" : "#f3f4f6",
              }}
            >
              {primaryState.upright_cos.toFixed(3)}
            </strong>
          </div>
          <div className="kpi-box">
            <span className="kpi-label">Falls Recorded</span>
            <strong
              className="kpi-value"
              style={{
                color: primaryState.fall_count > 0 ? "#f43f5e" : "#10b981",
              }}
            >
              {primaryState.fall_count}
            </strong>
          </div>
        </div>

        {duelMode && secondaryState && (
          <div className="duel-kpi-compare">
            <div className="duel-compare-title">
              LIVE HEAD-TO-HEAD POLICY COMPARISON
            </div>
            <div className="duel-compare-table">
              <div className="duel-col">
                <span className="duel-col-head">Metric</span>
                <span>Status</span>
                <span>Return</span>
                <span>Vel Error</span>
                <span>Roll Tilt</span>
                <span>Falls</span>
              </div>
              <div className="duel-col">
                <span className="duel-col-head" style={{ color: "#10b981" }}>
                  Robust PPO
                </span>
                {(() => {
                  const r =
                    primaryState.policy_name === "robust_ppo"
                      ? primaryState
                      : secondaryState;
                  return (
                    <>
                      <strong
                        style={{
                          color: r.terminated ? "#f43f5e" : "#10b981",
                        }}
                      >
                        {r.terminated ? "FALLEN" : "STABLE"}
                      </strong>
                      <span>{r.cumulative_return.toFixed(1)}</span>
                      <span>{r.vel_tracking_error.toFixed(3)} m/s</span>
                      <span>{r.roll_deg.toFixed(1)}°</span>
                      <span>{r.fall_count}</span>
                    </>
                  );
                })()}
              </div>
              <div className="duel-col">
                <span className="duel-col-head" style={{ color: "#38bdf8" }}>
                  Baseline PPO
                </span>
                {(() => {
                  const b =
                    primaryState.policy_name === "baseline_ppo"
                      ? primaryState
                      : secondaryState;
                  return (
                    <>
                      <strong
                        style={{
                          color: b.terminated ? "#f43f5e" : "#38bdf8",
                        }}
                      >
                        {b.terminated ? "FALLEN" : "STABLE"}
                      </strong>
                      <span>{b.cumulative_return.toFixed(1)}</span>
                      <span>{b.vel_tracking_error.toFixed(3)} m/s</span>
                      <span>{b.roll_deg.toFixed(1)}°</span>
                      <span>{b.fall_count}</span>
                    </>
                  );
                })()}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* 2. Streaming Telemetry Sparklines */}
      <section className="control-card">
        <div className="card-header">
          <BarChart2 size={15} className="icon-accent" />
          <h3>STREAMING TIME-SERIES DIAGNOSTICS</h3>
        </div>

        <div className="spark-block">
          <div className="spark-label-row">
            <span>Torso Roll Tilt Angle (°)</span>
            <strong>{primaryState.roll_deg.toFixed(1)}°</strong>
          </div>
          {renderSparkline(rollSeries, -45, 45, "#f59e0b", true)}
        </div>

        <div className="spark-block">
          <div className="spark-label-row">
            <span>Velocity Tracking Error (m/s)</span>
            <strong>{primaryState.vel_tracking_error.toFixed(3)} m/s</strong>
          </div>
          {renderSparkline(velErrSeries, 0, 0.8, accentColor, false)}
        </div>

        <div className="spark-block">
          <div className="spark-label-row">
            <span>Critic State Value Estimate V(s)</span>
            <strong>{primaryState.critic_value.toFixed(2)}</strong>
          </div>
          {renderSparkline(criticSeries, -10, 35, "#a855f7", true)}
        </div>
      </section>

      {/* 3. Foot Contact Matrix & 12-DOF Actuator Torques */}
      <section className="control-card">
        <div className="card-header">
          <Footprints size={15} className="icon-accent" />
          <h3>GAIT PHASE & 12-DOF ACTUATOR TORQUES</h3>
        </div>

        <div className="foot-contact-grid">
          {(["FL", "FR", "RL", "RR"] as const).map((leg, i) => {
            const active = primaryState.foot_contacts[i];
            return (
              <div
                key={leg}
                className={`foot-pill ${active ? "stance" : "swing"}`}
              >
                <span className="foot-name">{leg}</span>
                <span className="foot-phase">
                  {active ? "STANCE" : "SWING"}
                </span>
              </div>
            );
          })}
        </div>

        <div className="joint-bars-header">
          <Cpu size={13} />
          <span>PD Motor Torques τ (N·m) & Policy Action a_t</span>
        </div>

        <div className="joint-bars-list">
          {JOINT_LABELS.map((label, idx) => {
            const tau = primaryState.torques[idx] ?? 0;
            const pct = Math.min(100, (Math.abs(tau) / 28.0) * 100);
            const isHip = idx % 3 === 0;
            return (
              <div key={label} className="joint-bar-row">
                <span className="joint-bar-name">{label}</span>
                <div className="joint-bar-track">
                  <div
                    className="joint-bar-fill"
                    style={{
                      width: `${pct}%`,
                      backgroundColor: isHip ? "#f59e0b" : accentColor,
                    }}
                  />
                </div>
                <span className="joint-bar-val">
                  {tau >= 0 ? `+${tau.toFixed(1)}` : tau.toFixed(1)}
                </span>
              </div>
            );
          })}
        </div>
      </section>
    </aside>
  );
};
