import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { RobotRenderState } from "../physics/quadrupedSimulator";

interface RobotViewport3DProps {
  primaryState: RobotRenderState;
  secondaryState: RobotRenderState | null;
  duelMode: boolean;
  previewPushAzimuthDeg: number;
  previewPushForceN: number;
  onQuickPush: (forceN: number, azimuthDeg: number) => void;
}

interface ArticulatedQuadrupedRig {
  root: THREE.Group;
  visorMat: THREE.MeshStandardMaterial;
  hipPivots: THREE.Group[];
  thighPivots: THREE.Group[];
  calfPivots: THREE.Group[];
  footSpheres: THREE.Mesh[];
  contactRings: THREE.Mesh[];
  pushArrow: THREE.ArrowHelper;
  velArrow: THREE.ArrowHelper;
  trailLine: THREE.Line;
  trailPoints: THREE.Vector3[];
}

const HIP_COORDS: [number, number, number][] = [
  [0.188, 0.0, -0.065],  // FL in Three.js (x=forward, y=up, z=-left)
  [0.188, 0.0, 0.065],   // FR
  [-0.188, 0.0, -0.065], // RL
  [-0.188, 0.0, 0.065],  // RR
];
const LAT_SIGNS = [-1, 1, -1, 1]; // Left is -Z in Three.js right-handed (X=forward, Y=up, Z=right)

function createQuadrupedRig(
  accentHex: number,
  isGhost = false
): ArticulatedQuadrupedRig {
  const root = new THREE.Group();
  const opacity = isGhost ? 0.82 : 1.0;

  const chassisMat = new THREE.MeshStandardMaterial({
    color: isGhost ? 0x1e293b : 0x1f2937,
    metalness: 0.65,
    roughness: 0.32,
    transparent: isGhost,
    opacity,
  });

  const darkMetalMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    metalness: 0.75,
    roughness: 0.28,
    transparent: isGhost,
    opacity,
  });

  const visorMat = new THREE.MeshStandardMaterial({
    color: accentHex,
    emissive: accentHex,
    emissiveIntensity: 0.65,
    metalness: 0.3,
    roughness: 0.2,
    transparent: isGhost,
    opacity,
  });

  // Main trunk body (length 0.376m, height 0.11m, width 0.13m)
  const trunkGeom = new THREE.BoxGeometry(0.376, 0.11, 0.135);
  const trunkMesh = new THREE.Mesh(trunkGeom, chassisMat);
  trunkMesh.castShadow = true;
  trunkMesh.receiveShadow = true;
  root.add(trunkMesh);

  // Top spine rail & IMU dome
  const railGeom = new THREE.BoxGeometry(0.28, 0.018, 0.09);
  const railMesh = new THREE.Mesh(railGeom, darkMetalMat);
  railMesh.position.set(-0.01, 0.062, 0);
  root.add(railMesh);

  const imuGeom = new THREE.CylinderGeometry(0.026, 0.03, 0.024, 20);
  const imuMesh = new THREE.Mesh(imuGeom, visorMat);
  imuMesh.position.set(0.08, 0.072, 0);
  root.add(imuMesh);

  // Front optical visor
  const visorGeom = new THREE.BoxGeometry(0.045, 0.055, 0.11);
  const visorMesh = new THREE.Mesh(visorGeom, visorMat);
  visorMesh.position.set(0.175, 0.012, 0);
  root.add(visorMesh);

  // Side accent stripes
  const stripeGeom = new THREE.BoxGeometry(0.26, 0.016, 0.14);
  const stripeMesh = new THREE.Mesh(stripeGeom, visorMat);
  stripeMesh.position.set(-0.01, 0.0, 0);
  root.add(stripeMesh);

  const hipPivots: THREE.Group[] = [];
  const thighPivots: THREE.Group[] = [];
  const calfPivots: THREE.Group[] = [];
  const footSpheres: THREE.Mesh[] = [];
  const contactRings: THREE.Mesh[] = [];

  for (let i = 0; i < 4; i++) {
    const hipPivot = new THREE.Group();
    hipPivot.position.set(HIP_COORDS[i][0], HIP_COORDS[i][1], HIP_COORDS[i][2]);
    root.add(hipPivot);
    hipPivots.push(hipPivot);

    // Hip actuator drum
    const drumGeom = new THREE.CylinderGeometry(0.036, 0.036, 0.058, 20);
    const drumMesh = new THREE.Mesh(drumGeom, visorMat);
    drumMesh.position.set(0, 0, LAT_SIGNS[i] * 0.035);
    hipPivot.add(drumMesh);

    const thighPivot = new THREE.Group();
    thighPivot.position.set(0, 0, LAT_SIGNS[i] * 0.078);
    hipPivot.add(thighPivot);
    thighPivots.push(thighPivot);

    // Shoulder joint cap
    const capGeom = new THREE.CylinderGeometry(0.032, 0.032, 0.038, 18);
    capGeom.rotateX(Math.PI / 2);
    const capMesh = new THREE.Mesh(capGeom, darkMetalMat);
    thighPivot.add(capMesh);

    // Thigh link (0.213m downward along -Y)
    const thighGeom = new THREE.CapsuleGeometry(0.022, 0.175, 8, 16);
    const thighMesh = new THREE.Mesh(thighGeom, chassisMat);
    thighMesh.position.set(0, -0.1065, 0);
    thighMesh.castShadow = true;
    thighPivot.add(thighMesh);

    const calfPivot = new THREE.Group();
    calfPivot.position.set(0, -0.213, 0);
    thighPivot.add(calfPivot);
    calfPivots.push(calfPivot);

    // Calf link (0.213m downward along -Y)
    const calfGeom = new THREE.CapsuleGeometry(0.016, 0.18, 8, 16);
    const calfMesh = new THREE.Mesh(calfGeom, darkMetalMat);
    calfMesh.position.set(0, -0.1065, 0);
    calfMesh.castShadow = true;
    calfPivot.add(calfMesh);

    // Foot sphere
    const footGeom = new THREE.SphereGeometry(0.024, 16, 16);
    const footMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xf59e0b,
      emissiveIntensity: 0.35,
      roughness: 0.4,
    });
    const footMesh = new THREE.Mesh(footGeom, footMat);
    footMesh.position.set(0, -0.213, 0);
    footMesh.castShadow = true;
    calfPivot.add(footMesh);
    footSpheres.push(footMesh);

    // Ground contact ring
    const ringGeom = new THREE.RingGeometry(0.022, 0.044, 24);
    ringGeom.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: accentHex,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.65,
    });
    const ringMesh = new THREE.Mesh(ringGeom, ringMat);
    ringMesh.position.set(0, 0.003, 0);
    contactRings.push(ringMesh);
  }

  const pushArrow = new THREE.ArrowHelper(
    new THREE.Vector3(0, 0, -1),
    new THREE.Vector3(0, 0.3, 0.5),
    0.5,
    0xf43f5e,
    0.14,
    0.09
  );
  pushArrow.visible = false;

  const velArrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 0.02, 0),
    0.4,
    accentHex,
    0.1,
    0.06
  );

  const trailGeom = new THREE.BufferGeometry();
  const trailMat = new THREE.LineBasicMaterial({
    color: accentHex,
    transparent: true,
    opacity: 0.6,
  });
  const trailLine = new THREE.Line(trailGeom, trailMat);

  return {
    root,
    visorMat,
    hipPivots,
    thighPivots,
    calfPivots,
    footSpheres,
    contactRings,
    pushArrow,
    velArrow,
    trailLine,
    trailPoints: [],
  };
}

function updateRigFromState(
  rig: ArticulatedQuadrupedRig,
  state: RobotRenderState,
  zLateralOffset: number,
  accentHex: number
): void {
  // Convert MuJoCo [x, y, z] (Z-up) -> Three.js [x, z, -y] (Y-up)
  const tx = state.base_pos[0];
  const ty = state.base_pos[2];
  const tz = -state.base_pos[1] + zLateralOffset;

  rig.root.position.set(tx, ty, tz);

  // Convert MuJoCo quaternion (w, x, y, z) in Z-up to Three.js Y-up
  const [qw, qx, qy, qz] = state.base_quat_wxyz;
  rig.root.quaternion.set(qx, qz, -qy, qw);

  const colorHex = state.terminated ? 0xf43f5e : accentHex;
  rig.visorMat.color.setHex(colorHex);
  rig.visorMat.emissive.setHex(colorHex);

  for (let leg = 0; leg < 4; leg++) {
    const qHip = state.joint_pos[leg * 3 + 0] ?? 0;
    const qThigh = state.joint_pos[leg * 3 + 1] ?? 0.8;
    const qCalf = state.joint_pos[leg * 3 + 2] ?? -1.55;

    // MuJoCo +X rotation corresponds to +X in Three.js
    rig.hipPivots[leg].rotation.set(qHip, 0, 0);
    // MuJoCo +Y rotation (pitch downward/backward) corresponds to -Z rotation in Three.js
    rig.thighPivots[leg].rotation.set(0, 0, -qThigh);
    rig.calfPivots[leg].rotation.set(0, 0, -qCalf);

    const footMat = rig.footSpheres[leg].material as THREE.MeshStandardMaterial;
    const inContact = state.foot_contacts[leg];
    footMat.emissiveIntensity = inContact ? 0.85 : 0.15;

    // Update contact ring position on floor using world transform of footSphere
    const footWorld = new THREE.Vector3();
    rig.footSpheres[leg].getWorldPosition(footWorld);
    rig.contactRings[leg].position.set(footWorld.x, 0.004, footWorld.z);
    rig.contactRings[leg].visible = inContact && !state.terminated;
  }

  // External push force vector arrow
  const [fx, fy, fz] = state.external_force;
  const fMag = Math.hypot(fx, fy, fz);
  if (fMag > 1.0) {
    const dirThree = new THREE.Vector3(fx, fz, -fy).normalize();
    const arrowLen = Math.min(1.1, 0.25 + fMag * 0.009);
    const origin = new THREE.Vector3(tx, ty + 0.04, tz).sub(
      dirThree.clone().multiplyScalar(arrowLen + 0.16)
    );
    rig.pushArrow.position.copy(origin);
    rig.pushArrow.setDirection(dirThree);
    rig.pushArrow.setLength(arrowLen, 0.16, 0.1);
    rig.pushArrow.visible = true;
  } else {
    rig.pushArrow.visible = false;
  }

  // Velocity vector arrow on ground
  const vx = state.base_lin_vel[0];
  const vy = state.base_lin_vel[1];
  const speed = Math.hypot(vx, vy);
  if (speed > 0.05 && !state.terminated) {
    const vDir = new THREE.Vector3(vx, 0, -vy).normalize();
    rig.velArrow.position.set(tx, 0.015, tz);
    rig.velArrow.setDirection(vDir);
    rig.velArrow.setLength(Math.min(0.65, 0.15 + speed * 0.45), 0.1, 0.06);
    rig.velArrow.visible = true;
  } else {
    rig.velArrow.visible = false;
  }

  // Trajectory ribbon
  if (state.step <= 1) {
    rig.trailPoints = [];
  }
  const lastPt = rig.trailPoints[rig.trailPoints.length - 1];
  const currPt = new THREE.Vector3(tx, 0.006, tz);
  if (!lastPt || lastPt.distanceTo(currPt) > 0.02) {
    rig.trailPoints.push(currPt);
    if (rig.trailPoints.length > 160) {
      rig.trailPoints.shift();
    }
    rig.trailLine.geometry.setFromPoints(rig.trailPoints);
  }
}

export const RobotViewport3D: React.FC<RobotViewport3DProps> = ({
  primaryState,
  secondaryState,
  duelMode,
  onQuickPush,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [followCam, setFollowCam] = useState(true);
  const followCamRef = useRef(true);

  const cameraSpherical = useRef({
    radius: 2.15,
    theta: 0.72,
    phi: 1.08,
    targetOffset: new THREE.Vector3(0, 0.24, 0),
  });

  const rigsRef = useRef<{
    primary: ArticulatedQuadrupedRig;
    secondary: ArticulatedQuadrupedRig;
    grid: THREE.GridHelper;
  } | null>(null);

  useEffect(() => {
    followCamRef.current = followCam;
  }, [followCam]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b14);
    scene.fog = new THREE.FogExp2(0x070b14, 0.065);

    const camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / Math.max(1, container.clientHeight),
      0.05,
      80
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    // Lighting
    const ambient = new THREE.AmbientLight(0xdbeafe, 0.65);
    scene.add(ambient);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.35);
    dirLight.position.set(4, 7, 4);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    scene.add(dirLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 0.45);
    rimLight.position.set(-4, 3, -4);
    scene.add(rimLight);

    // Arena floor & grid
    const floorGeom = new THREE.PlaneGeometry(120, 120);
    floorGeom.rotateX(-Math.PI / 2);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x0b1120,
      roughness: 0.85,
      metalness: 0.15,
    });
    const floorMesh = new THREE.Mesh(floorGeom, floorMat);
    floorMesh.receiveShadow = true;
    scene.add(floorMesh);

    const grid = new THREE.GridHelper(60, 120, 0x1e293b, 0x111827);
    grid.position.y = 0.002;
    scene.add(grid);

    // Primary and secondary quadruped rigs
    const primaryRig = createQuadrupedRig(0x10b981, false);
    scene.add(primaryRig.root);
    scene.add(primaryRig.pushArrow);
    scene.add(primaryRig.velArrow);
    scene.add(primaryRig.trailLine);
    primaryRig.contactRings.forEach((r) => scene.add(r));

    const secondaryRig = createQuadrupedRig(0x38bdf8, true);
    scene.add(secondaryRig.root);
    scene.add(secondaryRig.pushArrow);
    scene.add(secondaryRig.velArrow);
    scene.add(secondaryRig.trailLine);
    secondaryRig.contactRings.forEach((r) => scene.add(r));

    rigsRef.current = { primary: primaryRig, secondary: secondaryRig, grid };

    // Orbit mouse controls
    let isDragging = false;
    let prevX = 0;
    let prevY = 0;

    const onPointerDown = (e: PointerEvent) => {
      isDragging = true;
      prevX = e.clientX;
      prevY = e.clientY;
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - prevX;
      const dy = e.clientY - prevY;
      prevX = e.clientX;
      prevY = e.clientY;
      cameraSpherical.current.theta -= dx * 0.0075;
      cameraSpherical.current.phi = Math.max(
        0.22,
        Math.min(1.48, cameraSpherical.current.phi - dy * 0.0065)
      );
    };
    const onPointerUp = () => {
      isDragging = false;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      cameraSpherical.current.radius = Math.max(
        0.95,
        Math.min(5.5, cameraSpherical.current.radius + e.deltaY * 0.0015)
      );
    };

    const dom = renderer.domElement;
    dom.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    dom.addEventListener("wheel", onWheel, { passive: false });

    const onResize = () => {
      if (!container) return;
      camera.aspect = container.clientWidth / Math.max(1, container.clientHeight);
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener("resize", onResize);

    let animId = 0;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      const sph = cameraSpherical.current;
      const targetPos = followCamRef.current
        ? new THREE.Vector3(primaryRig.root.position.x, 0.24, primaryRig.root.position.z)
        : sph.targetOffset;

      const cx =
        targetPos.x + sph.radius * Math.sin(sph.phi) * Math.cos(sph.theta);
      const cy = targetPos.y + sph.radius * Math.cos(sph.phi);
      const cz =
        targetPos.z + sph.radius * Math.sin(sph.phi) * Math.sin(sph.theta);

      camera.position.set(cx, cy, cz);
      camera.lookAt(targetPos);

      // Keep grid centered near robot in long walks
      grid.position.x = Math.floor(primaryRig.root.position.x / 2) * 2;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      dom.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      dom.removeEventListener("wheel", onWheel);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    const rigs = rigsRef.current;
    if (!rigs) return;

    const primaryColor =
      primaryState.policy_name === "robust_ppo" ? 0x10b981 : 0x38bdf8;
    const zOffsetPrimary = duelMode ? -0.34 : 0.0;
    updateRigFromState(rigs.primary, primaryState, zOffsetPrimary, primaryColor);

    if (duelMode && secondaryState) {
      rigs.secondary.root.visible = true;
      rigs.secondary.trailLine.visible = true;
      const secColor =
        secondaryState.policy_name === "robust_ppo" ? 0x10b981 : 0x38bdf8;
      updateRigFromState(rigs.secondary, secondaryState, 0.34, secColor);
    } else {
      rigs.secondary.root.visible = false;
      rigs.secondary.pushArrow.visible = false;
      rigs.secondary.velArrow.visible = false;
      rigs.secondary.trailLine.visible = false;
      rigs.secondary.contactRings.forEach((r) => {
        r.visible = false;
      });
    }
  }, [primaryState, secondaryState, duelMode]);

  const setCameraPreset = (theta: number, phi: number, radius: number) => {
    cameraSpherical.current.theta = theta;
    cameraSpherical.current.phi = phi;
    cameraSpherical.current.radius = radius;
  };

  const extMag = Math.hypot(
    primaryState.external_force[0],
    primaryState.external_force[1],
    primaryState.external_force[2]
  );

  return (
    <div className="viewport-wrapper">
      <div ref={mountRef} className="viewport-canvas" />

      {/* Top-Left Live Arena Status Overlay */}
      <div className="viewport-hud-top-left">
        <div className="hud-badge">
          <span
            className="status-dot"
            style={{
              backgroundColor: primaryState.terminated
                ? "#f43f5e"
                : primaryState.policy_name === "robust_ppo"
                ? "#10b981"
                : "#38bdf8",
            }}
          />
          <span>
            {primaryState.policy_name === "robust_ppo"
              ? "ROBUST PPO (Disturbance-Aware)"
              : "BASELINE PPO (Nominal)"}
          </span>
          {duelMode && <span className="duel-tag">HEAD-TO-HEAD DUEL</span>}
        </div>

        <div className="hud-telemetry-strip">
          <div>
            <span className="hud-label">TIME</span>
            <span className="hud-val">{primaryState.time_s.toFixed(2)}s</span>
          </div>
          <div>
            <span className="hud-label">VX / CMD</span>
            <span className="hud-val">
              {primaryState.base_lin_vel[0].toFixed(2)} /{" "}
              {primaryState.vel_cmd[0].toFixed(2)} m/s
            </span>
          </div>
          <div>
            <span className="hud-label">ROLL / PITCH</span>
            <span className="hud-val">
              {primaryState.roll_deg.toFixed(1)}° /{" "}
              {primaryState.pitch_deg.toFixed(1)}°
            </span>
          </div>
          <div>
            <span className="hud-label">RECOVERY</span>
            <span className="hud-val">
              {primaryState.recovery_time_s !== null
                ? `${primaryState.recovery_time_s.toFixed(2)}s`
                : "Nominal"}
            </span>
          </div>
        </div>
      </div>

      {/* Active Push Alert Banner */}
      {extMag > 1.0 && (
        <div className="push-active-banner">
          EXTERNAL WRENCH ACTIVE: {extMag.toFixed(0)} N IMPULSE
        </div>
      )}

      {/* Fall Alert Banner */}
      {primaryState.terminated && (
        <div className="fall-alert-banner">
          ROBOT DESTABILIZED (TILT / HEIGHT TERMINATION THRESHOLD EXCEEDED)
        </div>
      )}

      {/* Duel Mode Legend */}
      {duelMode && secondaryState && (
        <div className="duel-overlay-card">
          <div className="duel-row">
            <span className="dot-emerald" />
            <strong>Robust PPO:</strong>
            <span>
              {primaryState.policy_name === "robust_ppo"
                ? primaryState.terminated
                  ? "FALLEN"
                  : `STABLE (Ret ${primaryState.cumulative_return.toFixed(0)})`
                : secondaryState.terminated
                ? "FALLEN"
                : `STABLE (Ret ${secondaryState.cumulative_return.toFixed(0)})`}
            </span>
          </div>
          <div className="duel-row">
            <span className="dot-cyan" />
            <strong>Baseline PPO:</strong>
            <span>
              {primaryState.policy_name === "baseline_ppo"
                ? primaryState.terminated
                  ? "FALLEN"
                  : `STABLE (Ret ${primaryState.cumulative_return.toFixed(0)})`
                : secondaryState.terminated
                ? "FALLEN"
                : `STABLE (Ret ${secondaryState.cumulative_return.toFixed(0)})`}
            </span>
          </div>
        </div>
      )}

      {/* Quick Shove Bar & Camera Presets at Bottom of Viewport */}
      <div className="viewport-hud-bottom">
        <div className="quick-push-group">
          <span className="quick-group-title">INSTANT SHOVE:</span>
          <button
            type="button"
            className="hud-btn push-weak"
            onClick={() => onQuickPush(28, 90)}
          >
            Weak (28 N)
          </button>
          <button
            type="button"
            className="hud-btn push-mod"
            onClick={() => onQuickPush(46, 90)}
          >
            Moderate (46 N)
          </button>
          <button
            type="button"
            className="hud-btn push-strong"
            onClick={() => onQuickPush(60, -90)}
          >
            Strong (60 N)
          </button>
          <button
            type="button"
            className="hud-btn push-ood"
            onClick={() => onQuickPush(78, 90)}
          >
            OOD Shock (78 N)
          </button>
        </div>

        <div className="camera-preset-group">
          <button
            type="button"
            className={`hud-btn ${followCam ? "active" : ""}`}
            onClick={() => setFollowCam((v) => !v)}
          >
            {followCam ? "Follow Cam: ON" : "Follow Cam: OFF"}
          </button>
          <button
            type="button"
            className="hud-btn"
            onClick={() => setCameraPreset(0.72, 1.08, 2.15)}
          >
            Iso
          </button>
          <button
            type="button"
            className="hud-btn"
            onClick={() => setCameraPreset(1.57, 1.25, 1.95)}
          >
            Side
          </button>
          <button
            type="button"
            className="hud-btn"
            onClick={() => setCameraPreset(0.0, 1.18, 1.85)}
          >
            Front
          </button>
          <button
            type="button"
            className="hud-btn"
            onClick={() => setCameraPreset(0.0, 0.25, 2.8)}
          >
            Top
          </button>
        </div>
      </div>
    </div>
  );
};
