import React from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Zap,
  StepForward,
  Shield,
  Activity,
  Compass,
  Sliders,
} from "lucide-react";
import { PolicyName } from "../policies/policyInference";

interface DisturbanceControlsProps {
  activePolicy: PolicyName;
  onSelectPolicy: (policy: PolicyName) => void;
  duelMode: boolean;
  onToggleDuelMode: () => void;
  isRunning: boolean;
  onToggleRun: () => void;
  onSingleStep: () => void;
  onReset: () => void;
  simSpeed: number;
  onChangeSimSpeed: (speed: number) => void;
  pushForceN: number;
  onChangePushForce: (force: number) => void;
  pushAzimuthDeg: number;
  onChangePushAzimuth: (deg: number) => void;
  pushDurationSteps: number;
  onChangePushDuration: (steps: number) => void;
  autoPushEnabled: boolean;
  onToggleAutoPush: () => void;
  onTriggerPush: () => void;
  vxCmd: number;
  vyCmd: number;
  wzCmd: number;
  onChangeVelCmd: (vx: number, vy: number, wz: number) => void;
  historyLength: number;
  replayIndex: number | null;
  onScrubReplay: (idx: number | null) => void;
}

export const DisturbanceControls: React.FC<DisturbanceControlsProps> = ({
  activePolicy,
  onSelectPolicy,
  duelMode,
  onToggleDuelMode,
  isRunning,
  onToggleRun,
  onSingleStep,
  onReset,
  simSpeed,
  onChangeSimSpeed,
  pushForceN,
  onChangePushForce,
  pushAzimuthDeg,
  onChangePushAzimuth,
  pushDurationSteps,
  onChangePushDuration,
  autoPushEnabled,
  onToggleAutoPush,
  onTriggerPush,
  vxCmd,
  vyCmd,
  wzCmd,
  onChangeVelCmd,
  historyLength,
  replayIndex,
  onScrubReplay,
}) => {
  const regimeLabel =
    pushForceN <= 32
      ? "Weak (In-Dist)"
      : pushForceN <= 48
      ? "Moderate (In-Dist)"
      : pushForceN <= 72
      ? "Strong (In-Dist)"
      : "Extreme OOD (>72 N)";

  const regimeColor =
    pushForceN <= 32
      ? "#10b981"
      : pushForceN <= 48
      ? "#38bdf8"
      : pushForceN <= 72
      ? "#f59e0b"
      : "#f43f5e";

  const rad = (pushAzimuthDeg * Math.PI) / 180;
  const cx = 60;
  const cy = 60;
  const r = 42;
  const tipX = cx + r * Math.cos(rad - Math.PI / 2);
  const tipY = cy + r * Math.sin(rad - Math.PI / 2);

  const handleCompassClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const dx = e.clientX - rect.left - cx;
    const dy = e.clientY - rect.top - cy;
    let deg = Math.round(((Math.atan2(dy, dx) + Math.PI / 2) * 180) / Math.PI);
    if (deg < 0) deg += 360;
    onChangePushAzimuth(deg);
  };

  return (
    <aside className="control-sidebar">
      {/* 1. Policy Selector */}
      <section className="control-card">
        <div className="card-header">
          <Shield size={15} className="icon-accent" />
          <h3>POLICY CONTROLLER</h3>
        </div>

        <div className="policy-selector-grid">
          <button
            type="button"
            className={`policy-card-btn ${
              activePolicy === "robust_ppo" ? "selected-robust" : ""
            }`}
            onClick={() => onSelectPolicy("robust_ppo")}
          >
            <div className="policy-btn-top">
              <span className="dot-emerald" />
              <strong>Robust PPO</strong>
            </div>
            <span className="policy-sub">
              Trained w/ 20–72 N Wrenches + Domain Rand
            </span>
          </button>

          <button
            type="button"
            className={`policy-card-btn ${
              activePolicy === "baseline_ppo" ? "selected-baseline" : ""
            }`}
            onClick={() => onSelectPolicy("baseline_ppo")}
          >
            <div className="policy-btn-top">
              <span className="dot-cyan" />
              <strong>Baseline PPO</strong>
            </div>
            <span className="policy-sub">
              Trained w/ Nominal 0 N Locomotion Only
            </span>
          </button>
        </div>

        <label className="duel-toggle-row">
          <input
            type="checkbox"
            checked={duelMode}
            onChange={onToggleDuelMode}
          />
          <span>
            <strong>Synchronized Dual-Policy Arena</strong> (Compare Both Live)
          </span>
        </label>
      </section>

      {/* 2. External Disturbance Wrench Generator */}
      <section className="control-card">
        <div className="card-header">
          <Compass size={15} className="icon-rose" />
          <h3>EXTERNAL DISTURBANCE WRENCH</h3>
        </div>

        <div className="compass-and-sliders">
          <div className="compass-box">
            <svg
              width={120}
              height={120}
              className="compass-svg"
              onClick={handleCompassClick}
            >
              <circle
                cx={cx}
                cy={cy}
                r={48}
                fill="#0b1120"
                stroke="#1f2937"
                strokeWidth={2}
              />
              <circle
                cx={cx}
                cy={cy}
                r={30}
                fill="none"
                stroke="#1e293b"
                strokeDasharray="3 3"
              />
              <line x1={cx} y1={12} x2={cx} y2={108} stroke="#1f2937" />
              <line x1={12} y1={cy} x2={108} y2={cy} stroke="#1f2937" />
              <text x={cx} y={11} textAnchor="middle" fill="#9ca3af" fontSize={9}>
                0° F
              </text>
              <text x={112} y={cy + 3} textAnchor="middle" fill="#9ca3af" fontSize={9}>
                90°
              </text>
              <text x={cx} y={117} textAnchor="middle" fill="#9ca3af" fontSize={9}>
                180°
              </text>
              <text x={9} y={cy + 3} textAnchor="middle" fill="#9ca3af" fontSize={9}>
                270°
              </text>
              <line
                x1={cx}
                y1={cy}
                x2={tipX}
                y2={tipY}
                stroke="#f43f5e"
                strokeWidth={3}
              />
              <circle cx={tipX} cy={tipY} r={5} fill="#f43f5e" />
              <circle cx={cx} cy={cy} r={3.5} fill="#e2e8f0" />
            </svg>
            <div className="compass-caption">Azimuth: {pushAzimuthDeg}°</div>
          </div>

          <div className="angle-presets">
            {[
              { label: "Front (0°)", deg: 0 },
              { label: "Left (90°)", deg: 90 },
              { label: "Rear (180°)", deg: 180 },
              { label: "Right (270°)", deg: 270 },
              { label: "Oblique (45°)", deg: 45 },
              { label: "Shear (135°)", deg: 135 },
            ].map((p) => (
              <button
                key={p.deg}
                type="button"
                className={`mini-angle-btn ${
                  pushAzimuthDeg === p.deg ? "active" : ""
                }`}
                onClick={() => onChangePushAzimuth(p.deg)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="slider-group">
          <div className="slider-label-row">
            <span>Force Magnitude</span>
            <strong style={{ color: regimeColor }}>
              {pushForceN} N ({regimeLabel})
            </strong>
          </div>
          <input
            type="range"
            min={15}
            max={95}
            step={1}
            value={pushForceN}
            onChange={(e) => onChangePushForce(Number(e.target.value))}
          />
        </div>

        <div className="slider-group">
          <div className="slider-label-row">
            <span>Impulse Duration</span>
            <strong>{(pushDurationSteps * 20).toFixed(0)} ms ({pushDurationSteps} steps)</strong>
          </div>
          <input
            type="range"
            min={4}
            max={16}
            step={1}
            value={pushDurationSteps}
            onChange={(e) => onChangePushDuration(Number(e.target.value))}
          />
        </div>

        <button
          type="button"
          className="apply-push-cta"
          onClick={onTriggerPush}
        >
          <Zap size={16} />
          <span>APPLY {pushForceN} N PUSH WRENCH</span>
        </button>

        <label className="duel-toggle-row">
          <input
            type="checkbox"
            checked={autoPushEnabled}
            onChange={onToggleAutoPush}
          />
          <span>Periodic Stochastic Disturbance Mode (Every 1.5s)</span>
        </label>
      </section>

      {/* 3. Velocity Command & Playback Controls */}
      <section className="control-card">
        <div className="card-header">
          <Sliders size={15} className="icon-accent" />
          <h3>VELOCITY COMMAND & REPLAY</h3>
        </div>

        <div className="slider-group">
          <div className="slider-label-row">
            <span>Forward Velocity (vx)</span>
            <strong>{vxCmd.toFixed(2)} m/s</strong>
          </div>
          <input
            type="range"
            min={-0.3}
            max={0.85}
            step={0.05}
            value={vxCmd}
            onChange={(e) => onChangeVelCmd(Number(e.target.value), vyCmd, wzCmd)}
          />
        </div>

        <div className="slider-group">
          <div className="slider-label-row">
            <span>Lateral Velocity (vy)</span>
            <strong>{vyCmd.toFixed(2)} m/s</strong>
          </div>
          <input
            type="range"
            min={-0.25}
            max={0.25}
            step={0.05}
            value={vyCmd}
            onChange={(e) => onChangeVelCmd(vxCmd, Number(e.target.value), wzCmd)}
          />
        </div>

        <div className="slider-group">
          <div className="slider-label-row">
            <span>Yaw Rate (wz)</span>
            <strong>{wzCmd.toFixed(2)} rad/s</strong>
          </div>
          <input
            type="range"
            min={-0.45}
            max={0.45}
            step={0.05}
            value={wzCmd}
            onChange={(e) => onChangeVelCmd(vxCmd, vyCmd, Number(e.target.value))}
          />
        </div>

        <div className="transport-bar">
          <button
            type="button"
            className={`transport-btn ${isRunning ? "active" : ""}`}
            onClick={onToggleRun}
          >
            {isRunning ? <Pause size={15} /> : <Play size={15} />}
            <span>{isRunning ? "Pause" : "Resume"}</span>
          </button>
          <button
            type="button"
            className="transport-btn"
            onClick={onSingleStep}
            disabled={isRunning}
            title="Step 20ms forward"
          >
            <StepForward size={15} />
            <span>Step</span>
          </button>
          <button type="button" className="transport-btn" onClick={onReset}>
            <RotateCcw size={15} />
            <span>Reset</span>
          </button>
        </div>

        <div className="speed-row">
          <span>Sim Rate:</span>
          {[0.25, 0.5, 1.0].map((sp) => (
            <button
              key={sp}
              type="button"
              className={`speed-pill ${simSpeed === sp ? "active" : ""}`}
              onClick={() => onChangeSimSpeed(sp)}
            >
              {sp}x
            </button>
          ))}
        </div>

        <div className="slider-group">
          <div className="slider-label-row">
            <span>
              <Activity size={12} style={{ display: "inline", marginRight: 4 }} />
              Replay Scrubber
            </span>
            <strong>
              {replayIndex !== null
                ? `Frame ${replayIndex + 1} / ${historyLength}`
                : `Live (${historyLength} frames)`}
            </strong>
          </div>
          <input
            type="range"
            min={0}
            max={Math.max(0, historyLength - 1)}
            value={replayIndex !== null ? replayIndex : Math.max(0, historyLength - 1)}
            onChange={(e) => onScrubReplay(Number(e.target.value))}
          />
          {replayIndex !== null && (
            <button
              type="button"
              className="return-live-btn"
              onClick={() => onScrubReplay(null)}
            >
              Return to Live Simulation
            </button>
          )}
        </div>
      </section>
    </aside>
  );
};
