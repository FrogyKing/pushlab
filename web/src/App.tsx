import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Box,
  BarChart3,
  Cloud,
  BookOpen,
  Cpu,
  CheckCircle2,
} from "lucide-react";
import { PolicyName } from "./policies/policyInference";
import {
  BrowserQuadrupedInstance,
  RobotRenderState,
} from "./physics/quadrupedSimulator";
import { RobotViewport3D } from "./components/RobotViewport3D";
import { DisturbanceControls } from "./components/DisturbanceControls";
import { TelemetryPanel } from "./components/TelemetryPanel";
import {
  BenchmarkDashboard,
  DashboardSubTab,
} from "./components/BenchmarkDashboard";
import trainingRunsRaw from "./data/training_runs.json";

type MainView = "simulator" | "benchmarks" | "cloud" | "math";

interface HistoryFrame {
  robust: RobotRenderState;
  baseline: RobotRenderState;
}

export const App: React.FC = () => {
  const [mainView, setMainView] = useState<MainView>("simulator");
  const [activePolicy, setActivePolicy] = useState<PolicyName>("robust_ppo");
  const [duelMode, setDuelMode] = useState<boolean>(true);
  const [isRunning, setIsRunning] = useState<boolean>(true);
  const [simSpeed, setSimSpeed] = useState<number>(1.0);

  // Disturbance parameters
  const [pushForceN, setPushForceN] = useState<number>(56);
  const [pushAzimuthDeg, setPushAzimuthDeg] = useState<number>(90);
  const [pushDurationSteps, setPushDurationSteps] = useState<number>(8);
  const [autoPushEnabled, setAutoPushEnabled] = useState<boolean>(false);

  // Velocity command
  const [vxCmd, setVxCmd] = useState<number>(0.55);
  const [vyCmd, setVyCmd] = useState<number>(0.0);
  const [wzCmd, setWzCmd] = useState<number>(0.0);

  // Simulator instances
  const robustSimRef = useRef<BrowserQuadrupedInstance>(
    new BrowserQuadrupedInstance("robust_ppo")
  );
  const baselineSimRef = useRef<BrowserQuadrupedInstance>(
    new BrowserQuadrupedInstance("baseline_ppo")
  );

  const [robustState, setRobustState] = useState<RobotRenderState>(() =>
    robustSimRef.current.getRenderState()
  );
  const [baselineState, setBaselineState] = useState<RobotRenderState>(() =>
    baselineSimRef.current.getRenderState()
  );

  const historyRef = useRef<HistoryFrame[]>([]);
  const [historyLen, setHistoryLen] = useState<number>(1);
  const [replayIndex, setReplayIndex] = useState<number | null>(null);
  const [backendConnected, setBackendConnected] = useState<boolean>(false);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => (r.ok ? setBackendConnected(true) : setBackendConnected(false)))
      .catch(() => setBackendConnected(false));
  }, []);

  const isFetchingRef = useRef<boolean>(false);

  const stepBothSimulators = useCallback(() => {
    if (backendConnected) {
      if (isFetchingRef.current) return;
      isFetchingRef.current = true;
      fetch("/api/sim/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          policy_name: "robust_ppo",
          duel_mode: true,
          vx_cmd: vxCmd,
          vy_cmd: vyCmd,
          wz_cmd: wzCmd,
          num_steps: 1,
        }),
      })
        .then((res) => (res.ok ? res.json() : Promise.reject()))
        .then((data) => {
          const rState: RobotRenderState = data.primary;
          const bState: RobotRenderState = data.secondary;
          if (autoPushEnabled && rState.step > 0 && rState.step % 75 === 0) {
            const angles = [90, -90, 60, -60, 120, -120];
            const ang = angles[Math.floor(rState.step / 75) % angles.length];
            fetch("/api/sim/push", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                force_magnitude_n: pushForceN,
                azimuth_deg: ang,
                duration_steps: pushDurationSteps,
              }),
            }).catch(() => {});
          }
          historyRef.current.push({ robust: rState, baseline: bState });
          if (historyRef.current.length > 350) {
            historyRef.current.shift();
          }
          setHistoryLen(historyRef.current.length);
          setRobustState(rState);
          setBaselineState(bState);
        })
        .catch(() => setBackendConnected(false))
        .finally(() => {
          isFetchingRef.current = false;
        });
      return;
    }

    const rState = robustSimRef.current.step();
    const bState = baselineSimRef.current.step();

    // Auto-push check every 75 steps if enabled
    if (autoPushEnabled && rState.step > 0 && rState.step % 75 === 0) {
      const angles = [90, -90, 60, -60, 120, -120];
      const ang = angles[(rState.step / 75) % angles.length];
      robustSimRef.current.triggerPush(pushForceN, ang, pushDurationSteps);
      baselineSimRef.current.triggerPush(pushForceN, ang, pushDurationSteps);
    }

    historyRef.current.push({ robust: rState, baseline: bState });
    if (historyRef.current.length > 350) {
      historyRef.current.shift();
    }
    setHistoryLen(historyRef.current.length);
    setRobustState(rState);
    setBaselineState(bState);
  }, [backendConnected, vxCmd, vyCmd, wzCmd, autoPushEnabled, pushForceN, pushDurationSteps]);

  useEffect(() => {
    if (!isRunning || replayIndex !== null || mainView !== "simulator") {
      return;
    }
    const intervalMs = Math.round(20 / simSpeed);
    const timer = window.setInterval(stepBothSimulators, intervalMs);
    return () => window.clearInterval(timer);
  }, [isRunning, replayIndex, simSpeed, mainView, stepBothSimulators]);

  const handleReset = useCallback(() => {
    if (backendConnected) {
      fetch("/api/sim/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          policy_name: "robust_ppo",
          seed: 42,
          vx_cmd: vxCmd,
          vy_cmd: vyCmd,
          wz_cmd: wzCmd,
        }),
      })
        .then((res) => (res.ok ? res.json() : Promise.reject()))
        .then((data) => {
          const r: RobotRenderState = data.primary;
          const b: RobotRenderState = data.secondary;
          historyRef.current = [{ robust: r, baseline: b }];
          setHistoryLen(1);
          setReplayIndex(null);
          setRobustState(r);
          setBaselineState(b);
        })
        .catch(() => {});
      return;
    }
    const r = robustSimRef.current.reset([vxCmd, vyCmd, wzCmd]);
    const b = baselineSimRef.current.reset([vxCmd, vyCmd, wzCmd]);
    historyRef.current = [{ robust: r, baseline: b }];
    setHistoryLen(1);
    setReplayIndex(null);
    setRobustState(r);
    setBaselineState(b);
  }, [backendConnected, vxCmd, vyCmd, wzCmd]);

  const handleTriggerPush = useCallback(
    (overrideForce?: number, overrideAzimuth?: number) => {
      const f = overrideForce ?? pushForceN;
      const az = overrideAzimuth ?? pushAzimuthDeg;
      if (backendConnected) {
        fetch("/api/sim/push", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            force_magnitude_n: f,
            azimuth_deg: az,
            duration_steps: pushDurationSteps,
          }),
        }).then(() => {
          if (replayIndex !== null) {
            setReplayIndex(null);
          }
          if (!isRunning) {
            stepBothSimulators();
          }
        });
        return;
      }
      robustSimRef.current.triggerPush(f, az, pushDurationSteps);
      baselineSimRef.current.triggerPush(f, az, pushDurationSteps);
      if (replayIndex !== null) {
        setReplayIndex(null);
      }
      if (!isRunning) {
        stepBothSimulators();
      }
    },
    [
      backendConnected,
      pushForceN,
      pushAzimuthDeg,
      pushDurationSteps,
      replayIndex,
      isRunning,
      stepBothSimulators,
    ]
  );

  const handleChangeVelCmd = useCallback(
    (vx: number, vy: number, wz: number) => {
      setVxCmd(vx);
      setVyCmd(vy);
      setWzCmd(wz);
      robustSimRef.current.setVelocityCommand(vx, vy, wz);
      baselineSimRef.current.setVelocityCommand(vx, vy, wz);
    },
    []
  );

  const handleScrubReplay = useCallback((idx: number | null) => {
    if (idx === null) {
      setReplayIndex(null);
      return;
    }
    setIsRunning(false);
    const clamped = Math.max(0, Math.min(historyRef.current.length - 1, idx));
    setReplayIndex(clamped);
  }, []);

  const displayedRobust =
    replayIndex !== null && historyRef.current[replayIndex]
      ? historyRef.current[replayIndex].robust
      : robustState;

  const displayedBaseline =
    replayIndex !== null && historyRef.current[replayIndex]
      ? historyRef.current[replayIndex].baseline
      : baselineState;

  const primaryState =
    activePolicy === "robust_ppo" ? displayedRobust : displayedBaseline;
  const secondaryState =
    activePolicy === "robust_ppo" ? displayedBaseline : displayedRobust;

  const recentHistory = historyRef.current.map((h) =>
    activePolicy === "robust_ppo" ? h.robust : h.baseline
  );

  return (
    <div className="app-shell">
      {/* Top Navigation Bar */}
      <header className="top-navbar">
        <div className="brand-cluster">
          <div className="brand-logo">
            <Cpu size={18} />
          </div>
          <div>
            <div className="brand-title">
              PUSHLAB <span className="brand-tag">RESEARCH v0.1</span>
            </div>
            <div className="brand-subtitle">
              GPU-Accelerated Robust Reinforcement Learning for Quadruped Disturbance Rejection
            </div>
          </div>
        </div>

        <nav className="main-nav-tabs">
          <button
            type="button"
            className={`nav-tab ${mainView === "simulator" ? "active" : ""}`}
            onClick={() => setMainView("simulator")}
          >
            <Box size={15} />
            <span>3D Interactive Simulator</span>
          </button>
          <button
            type="button"
            className={`nav-tab ${mainView === "benchmarks" ? "active" : ""}`}
            onClick={() => setMainView("benchmarks")}
          >
            <BarChart3 size={15} />
            <span>Robustness Benchmarks</span>
          </button>
          <button
            type="button"
            className={`nav-tab ${mainView === "cloud" ? "active" : ""}`}
            onClick={() => setMainView("cloud")}
          >
            <Cloud size={15} />
            <span>Cloud GPU Telemetry</span>
          </button>
          <button
            type="button"
            className={`nav-tab ${mainView === "math" ? "active" : ""}`}
            onClick={() => setMainView("math")}
          >
            <BookOpen size={15} />
            <span>MDP & PPO Mathematics</span>
          </button>
        </nav>

        <div className="header-status-cluster">
          <div className="engine-pill">
            <CheckCircle2 size={13} className="icon-emerald" />
            <span>
              {backendConnected
                ? "MuJoCo + Browser NN Engine (50Hz)"
                : "Browser Physics + NN Engine (50Hz)"}
            </span>
          </div>
          <div className="telemetry-pill">
            <span>Training Hardware:</span>
            <strong>
              NVIDIA L4 ({(trainingRunsRaw.total_timesteps / 1e6).toFixed(2)}M Steps)
            </strong>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      {mainView === "simulator" ? (
        <main className="simulator-workspace">
          <DisturbanceControls
            activePolicy={activePolicy}
            onSelectPolicy={setActivePolicy}
            duelMode={duelMode}
            onToggleDuelMode={() => setDuelMode((v) => !v)}
            isRunning={isRunning}
            onToggleRun={() => {
              setReplayIndex(null);
              setIsRunning((v) => !v);
            }}
            onSingleStep={() => {
              setReplayIndex(null);
              stepBothSimulators();
            }}
            onReset={handleReset}
            simSpeed={simSpeed}
            onChangeSimSpeed={setSimSpeed}
            pushForceN={pushForceN}
            onChangePushForce={setPushForceN}
            pushAzimuthDeg={pushAzimuthDeg}
            onChangePushAzimuth={setPushAzimuthDeg}
            pushDurationSteps={pushDurationSteps}
            onChangePushDuration={setPushDurationSteps}
            autoPushEnabled={autoPushEnabled}
            onToggleAutoPush={() => setAutoPushEnabled((v) => !v)}
            onTriggerPush={() => handleTriggerPush()}
            vxCmd={vxCmd}
            vyCmd={vyCmd}
            wzCmd={wzCmd}
            onChangeVelCmd={handleChangeVelCmd}
            historyLength={historyLen}
            replayIndex={replayIndex}
            onScrubReplay={handleScrubReplay}
          />

          <RobotViewport3D
            primaryState={primaryState}
            secondaryState={secondaryState}
            duelMode={duelMode}
            previewPushAzimuthDeg={pushAzimuthDeg}
            previewPushForceN={pushForceN}
            onQuickPush={(force, az) => {
              setPushForceN(force);
              setPushAzimuthDeg(az);
              handleTriggerPush(force, az);
            }}
          />

          <TelemetryPanel
            primaryState={primaryState}
            secondaryState={secondaryState}
            duelMode={duelMode}
            recentHistory={recentHistory}
          />
        </main>
      ) : (
        <main className="dashboard-workspace">
          <BenchmarkDashboard
            key={mainView}
            initialSubTab={
              (mainView === "benchmarks"
                ? "regimes"
                : mainView === "cloud"
                ? "cloud"
                : "math") as DashboardSubTab
            }
          />
        </main>
      )}
    </div>
  );
};
