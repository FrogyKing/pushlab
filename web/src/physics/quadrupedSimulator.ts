import { POLICIES, PolicyName } from "../policies/policyInference";

export interface RobotRenderState {
  step: number;
  time_s: number;
  policy_name: PolicyName;
  base_pos: [number, number, number];
  base_quat_wxyz: [number, number, number, number];
  roll_deg: number;
  pitch_deg: number;
  yaw_deg: number;
  upright_cos: number;
  base_lin_vel: [number, number, number];
  base_ang_vel: [number, number, number];
  vel_cmd: [number, number, number];
  joint_pos: number[];
  joint_vel: number[];
  torques: number[];
  action: number[];
  foot_positions: [number, number, number][];
  foot_contacts: boolean[];
  external_force: [number, number, number];
  external_torque: [number, number, number];
  mechanical_power_w: number;
  vel_tracking_error: number;
  step_reward: number;
  cumulative_return: number;
  critic_value: number;
  recovery_time_s: number | null;
  terminated: boolean;
  fall_count: number;
}

const DEFAULT_JOINT_POS = [
  0.0, 0.8, -1.55,
  0.0, 0.8, -1.55,
  0.0, 0.8, -1.55,
  0.0, 0.8, -1.55,
];

const HIP_OFFSETS: [number, number, number][] = [
  [0.188, 0.065, 0.0],   // FL
  [0.188, -0.065, 0.0],  // FR
  [-0.188, 0.065, 0.0],  // RL
  [-0.188, -0.065, 0.0], // RR
];

const HIP_LAT_OFFSETS = [0.08, -0.08, 0.08, -0.08];
const LINK_LEN = 0.213;

function eulerToQuatWxyz(roll: number, pitch: number, yaw: number): [number, number, number, number] {
  const cr = Math.cos(roll * 0.5);
  const sr = Math.sin(roll * 0.5);
  const cp = Math.cos(pitch * 0.5);
  const sp = Math.sin(pitch * 0.5);
  const cy = Math.cos(yaw * 0.5);
  const sy = Math.sin(yaw * 0.5);

  const w = cr * cp * cy + sr * sp * sy;
  const x = sr * cp * cy - cr * sp * sy;
  const y = cr * sp * cy + sr * cp * sy;
  const z = cr * cp * sy - sr * sp * cy;
  return [w, x, y, z];
}

function quatRotateInverse(
  q: [number, number, number, number],
  v: [number, number, number]
): [number, number, number] {
  const [w, x, y, z] = q;
  const [vx, vy, vz] = v;
  const s = 2.0 * w * w - 1.0;
  const dot = x * vx + y * vy + z * vz;
  const cx = y * vz - z * vy;
  const cy = z * vx - x * vz;
  const cz = x * vy - y * vx;

  return [
    vx * s - 2.0 * w * cx + 2.0 * dot * x,
    vy * s - 2.0 * w * cy + 2.0 * dot * y,
    vz * s - 2.0 * w * cz + 2.0 * dot * z,
  ];
}

function quatRotate(
  q: [number, number, number, number],
  v: [number, number, number]
): [number, number, number] {
  const [w, x, y, z] = q;
  return quatRotateInverse([w, -x, -y, -z], v);
}

export class BrowserQuadrupedInstance {
  public readonly policyName: PolicyName;
  private stepCount = 0;
  private phaseRad = 0.0;
  private basePos: [number, number, number] = [0.0, 0.0, 0.305];
  private rpy: [number, number, number] = [0.0, 0.0, 0.0];
  private baseLinVelBody: [number, number, number] = [0.0, 0.0, 0.0];
  private baseAngVelBody: [number, number, number] = [0.0, 0.0, 0.0];
  private velCmd: [number, number, number] = [0.55, 0.0, 0.0];
  private jointPos: Float32Array = new Float32Array(DEFAULT_JOINT_POS);
  private jointVel: Float32Array = new Float32Array(12);
  private prevAction: Float32Array = new Float32Array(12);
  private lastAction: Float32Array = new Float32Array(12);
  private lastTorques: Float32Array = new Float32Array(12);
  private lastCriticValue = 0.0;
  private lastStepReward = 0.0;
  private cumulativeReturn = 0.0;
  private terminated = false;
  private fallCount = 0;

  // Disturbance state
  private activeForce: [number, number, number] = [0.0, 0.0, 0.0];
  private activeTorque: [number, number, number] = [0.0, 0.0, 0.0];
  private pushEndStep = -1;
  private postPushPerturbed = false;
  private lastRecoveryTimeS: number | null = null;

  constructor(policyName: PolicyName) {
    this.policyName = policyName;
    this.reset();
  }

  public reset(velCmd?: [number, number, number]): RobotRenderState {
    this.stepCount = 0;
    this.phaseRad = 0.35;
    this.basePos = [0.0, 0.0, 0.305];
    this.rpy = [0.0, 0.0, 0.0];
    this.baseLinVelBody = [0.0, 0.0, 0.0];
    this.baseAngVelBody = [0.0, 0.0, 0.0];
    if (velCmd) {
      this.velCmd = [...velCmd];
    }
    this.jointPos = new Float32Array(DEFAULT_JOINT_POS);
    this.jointVel = new Float32Array(12);
    this.prevAction = new Float32Array(12);
    this.lastAction = new Float32Array(12);
    this.lastTorques = new Float32Array(12);
    this.lastStepReward = 0.0;
    this.cumulativeReturn = 0.0;
    this.terminated = false;
    this.activeForce = [0.0, 0.0, 0.0];
    this.activeTorque = [0.0, 0.0, 0.0];
    this.pushEndStep = -1;
    this.postPushPerturbed = false;
    this.lastRecoveryTimeS = null;

    const obs = this.buildObservation();
    const { action, value } = POLICIES[this.policyName].predict(obs);
    this.lastAction = action;
    this.lastCriticValue = value;
    return this.getRenderState();
  }

  public setVelocityCommand(vx: number, vy: number, wz: number): void {
    this.velCmd = [vx, vy, wz];
  }

  public triggerPush(
    forceMagN: number,
    azimuthDeg: number,
    durationSteps = 8,
    verticalForceN = 0.0
  ): void {
    const azRad = (azimuthDeg * Math.PI) / 180.0;
    const fx = forceMagN * Math.cos(azRad);
    const fy = forceMagN * Math.sin(azRad);
    this.activeForce = [fx, fy, verticalForceN];
    this.activeTorque = [-fy * 0.055, fx * 0.055, 0.0];
    this.pushEndStep = this.stepCount + Math.max(1, durationSteps);
    this.postPushPerturbed = true;
    this.lastRecoveryTimeS = null;
  }

  private computeGaitReference(): Float32Array {
    const speedScale = Math.max(
      0.25,
      Math.min(
        1.35,
        Math.hypot(this.velCmd[0], this.velCmd[1]) / 0.55 + Math.abs(this.velCmd[2]) * 0.35
      )
    );
    const dirSign = this.velCmd[0] >= 0 ? 1.0 : -1.0;
    const s1 = Math.sin(this.phaseRad);
    const c1 = Math.cos(this.phaseRad) * dirSign;
    const s2 = -s1;
    const c2 = -c1;

    const ampT = 0.22 * speedScale;
    const ampC = 0.30 * speedScale;
    const hipLat = Math.max(-0.15, Math.min(0.15, 0.18 * this.velCmd[1]));
    const hipYaw = Math.max(-0.12, Math.min(0.12, 0.10 * this.velCmd[2]));

    const ref = new Float32Array(DEFAULT_JOINT_POS);
    // FL
    ref[0] += hipLat - hipYaw * s1;
    ref[1] += -ampT * c1;
    ref[2] += ampC * Math.max(0, s1);
    // FR
    ref[3] += hipLat + hipYaw * s2;
    ref[4] += -ampT * c2;
    ref[5] += ampC * Math.max(0, s2);
    // RL
    ref[6] += hipLat + hipYaw * s2;
    ref[7] += -ampT * c2;
    ref[8] += ampC * Math.max(0, s2);
    // RR
    ref[9] += hipLat - hipYaw * s1;
    ref[10] += -ampT * c1;
    ref[11] += ampC * Math.max(0, s1);
    return ref;
  }

  private buildObservation(): Float32Array {
    const quat = eulerToQuatWxyz(this.rpy[0], this.rpy[1], this.rpy[2]);
    const gravBody = quatRotateInverse(quat, [0.0, 0.0, -1.0]);

    const obs = new Float32Array(50);
    obs[0] = gravBody[0];
    obs[1] = gravBody[1];
    obs[2] = gravBody[2];
    obs[3] = this.baseLinVelBody[0];
    obs[4] = this.baseLinVelBody[1];
    obs[5] = this.baseLinVelBody[2];
    obs[6] = this.baseAngVelBody[0] * 0.25;
    obs[7] = this.baseAngVelBody[1] * 0.25;
    obs[8] = this.baseAngVelBody[2] * 0.25;
    obs[9] = this.velCmd[0];
    obs[10] = this.velCmd[1];
    obs[11] = this.velCmd[2];
    obs[12] = Math.sin(this.phaseRad);
    obs[13] = Math.cos(this.phaseRad);

    for (let i = 0; i < 12; i++) {
      obs[14 + i] = this.jointPos[i] - DEFAULT_JOINT_POS[i];
      obs[26 + i] = this.jointVel[i] * 0.05;
      obs[38 + i] = this.prevAction[i];
    }
    return obs;
  }

  public step(): RobotRenderState {
    if (this.terminated) {
      return this.getRenderState();
    }

    const obs = this.buildObservation();
    const { action, value } = POLICIES[this.policyName].predict(obs);
    this.lastCriticValue = value;

    const clippedAction = new Float32Array(12);
    const actionOffset = new Float32Array(12);
    for (let i = 0; i < 12; i++) {
      clippedAction[i] = Math.max(-2.5, Math.min(2.5, action[i]));
      actionOffset[i] = 0.35 * Math.tanh(clippedAction[i]);
    }
    this.lastAction = clippedAction;

    const qPhaseRef = this.computeGaitReference();
    const qTarget = new Float32Array(12);
    for (let i = 0; i < 12; i++) {
      qTarget[i] = qPhaseRef[i] + actionOffset[i];
    }

    const isPushActive = this.stepCount < this.pushEndStep;
    const extForce: [number, number, number] = isPushActive
      ? this.activeForce
      : [0.0, 0.0, 0.0];
    const extTorque: [number, number, number] = isPushActive
      ? this.activeTorque
      : [0.0, 0.0, 0.0];

    const mass = 6.06;
    const subDt = 0.005;
    const decimation = 4;
    const kp = 35.0;
    const kv = 1.0;

    // Evaluate policy's learned knee brace and hip abduction reflex authority
    const meanHipAction =
      0.25 * (actionOffset[0] + actionOffset[3] + actionOffset[6] + actionOffset[9]);
    const kneeAsymAction =
      0.5 * (actionOffset[2] - actionOffset[5] + actionOffset[8] - actionOffset[11]);
    const meanThighAction =
      0.25 * (actionOffset[1] + actionOffset[4] + actionOffset[7] + actionOffset[10]);

    // Reflex authority measures active capture-point bracing from the trained neural network
    const rawReflex =
      Math.max(0.0, Math.abs(kneeAsymAction) - 0.12) * 4.2 +
      Math.max(0.0, Math.abs(meanHipAction) - 0.04) * 2.0;
    const reflexAuthority = Math.min(1.0, rawReflex);

    for (let sub = 0; sub < decimation; sub++) {
      for (let i = 0; i < 12; i++) {
        const tau = Math.max(
          -28.0,
          Math.min(28.0, kp * (qTarget[i] - this.jointPos[i]) - kv * this.jointVel[i])
        );
        this.lastTorques[i] = tau;
        const qAcc = (tau - 1.2 * this.jointVel[i]) / 0.065;
        this.jointVel[i] += qAcc * subDt;
        this.jointPos[i] += this.jointVel[i] * subDt;
      }

      // Rotate world external force into body frame
      const quat = eulerToQuatWxyz(this.rpy[0], this.rpy[1], this.rpy[2]);
      const extForceBody = quatRotateInverse(quat, extForce);

      // Ground reaction & locomotion propulsion with capture-point lateral damping
      const vxDrive = (6.8 + 3.0 * reflexAuthority) * (this.velCmd[0] - this.baseLinVelBody[0]);
      const vyDrive = (6.2 + 5.5 * reflexAuthority) * (this.velCmd[1] - this.baseLinVelBody[1]);
      const wzDrive = 6.5 * (this.velCmd[2] - this.baseAngVelBody[2]);

      const ax = vxDrive + (extForceBody[0] / mass) * (1.0 - 0.35 * reflexAuthority);
      const ay = vyDrive + (extForceBody[1] / mass) * (1.0 - 0.38 * reflexAuthority);
      const az =
        48.0 * (0.305 - this.basePos[2]) -
        8.5 * this.baseLinVelBody[2] +
        extForceBody[2] / mass;

      this.baseLinVelBody[0] += ax * subDt;
      this.baseLinVelBody[1] += ay * subDt;
      this.baseLinVelBody[2] += az * subDt;

      // Roll/pitch rotational dynamics with capture-point stabilization vs overturning moment
      const rollCaptureLimit = 0.38 + 0.30 * reflexAuthority;
      const pitchCaptureLimit = 0.42 + 0.26 * reflexAuthority;

      const gravityOverturnRoll =
        Math.abs(this.rpy[0]) > rollCaptureLimit ? 28.0 * Math.sin(this.rpy[0]) : 0.0;
      const gravityOverturnPitch =
        Math.abs(this.rpy[1]) > pitchCaptureLimit ? 24.0 * Math.sin(this.rpy[1]) : 0.0;

      const rollStiffness = 24.0 + 18.0 * reflexAuthority;
      const rollDamping = 5.8 + 4.6 * reflexAuthority;
      const pushRollSensitivity = 0.46 * (1.0 - 0.48 * reflexAuthority);

      const pitchStiffness = 25.0 + 14.0 * reflexAuthority + 4.0 * Math.abs(meanThighAction);
      const pitchDamping = 6.0 + 4.0 * reflexAuthority;
      const pushPitchSensitivity = 0.34 * (1.0 - 0.45 * reflexAuthority);

      const rollAlpha =
        -rollStiffness * this.rpy[0] -
        rollDamping * this.baseAngVelBody[0] +
        pushRollSensitivity * extForceBody[1] +
        0.25 * extTorque[0] +
        gravityOverturnRoll;
      const pitchAlpha =
        -pitchStiffness * this.rpy[1] -
        pitchDamping * this.baseAngVelBody[1] -
        pushPitchSensitivity * extForceBody[0] +
        0.25 * extTorque[1] +
        gravityOverturnPitch;

      this.baseAngVelBody[0] += rollAlpha * subDt;
      this.baseAngVelBody[1] += pitchAlpha * subDt;
      this.baseAngVelBody[2] += wzDrive * subDt;

      this.rpy[0] += this.baseAngVelBody[0] * subDt;
      this.rpy[1] += this.baseAngVelBody[1] * subDt;
      this.rpy[2] += this.baseAngVelBody[2] * subDt;

      const worldVel = quatRotate(
        eulerToQuatWxyz(this.rpy[0], this.rpy[1], this.rpy[2]),
        this.baseLinVelBody
      );
      this.basePos[0] += worldVel[0] * subDt;
      this.basePos[1] += worldVel[1] * subDt;
      this.basePos[2] = Math.max(0.08, this.basePos[2] + worldVel[2] * subDt);
    }

    this.stepCount += 1;
    this.phaseRad = (this.phaseRad + 2.0 * Math.PI * 2.2 * 0.02) % (2.0 * Math.PI);
    this.prevAction.set(clippedAction);

    const quat = eulerToQuatWxyz(this.rpy[0], this.rpy[1], this.rpy[2]);
    const gravBody = quatRotateInverse(quat, [0.0, 0.0, -1.0]);
    const uprightCos = -gravBody[2];

    if (this.basePos[2] < 0.16 || uprightCos < 0.38) {
      this.terminated = true;
      this.fallCount += 1;
      this.basePos[2] = 0.11;
    }

    if (
      this.postPushPerturbed &&
      !this.terminated &&
      this.stepCount >= this.pushEndStep
    ) {
      const velErr = Math.hypot(
        this.baseLinVelBody[0] - this.velCmd[0],
        this.baseLinVelBody[1] - this.velCmd[1]
      );
      const tiltMag = Math.hypot(gravBody[0], gravBody[1]);
      if (velErr < 0.25 && tiltMag < 0.16) {
        this.lastRecoveryTimeS = Number(
          ((this.stepCount - this.pushEndStep + 1) * 0.02).toFixed(2)
        );
        this.postPushPerturbed = false;
      }
    }

    // Compute step reward
    const velErrSq =
      Math.pow(this.baseLinVelBody[0] - this.velCmd[0], 2) +
      Math.pow(this.baseLinVelBody[1] - this.velCmd[1], 2);
    const rLin = Math.exp(-velErrSq / 0.12);
    const rUp = Math.exp(-(gravBody[0] * gravBody[0] + gravBody[1] * gravBody[1]) / 0.08);
    this.lastStepReward = this.terminated
      ? -1.6
      : (2.4 * rLin + 1.6 * rUp + 2.0) * 0.2;
    this.cumulativeReturn += this.lastStepReward;

    return this.getRenderState();
  }

  public getRenderState(): RobotRenderState {
    const quat = eulerToQuatWxyz(this.rpy[0], this.rpy[1], this.rpy[2]);
    const gravBody = quatRotateInverse(quat, [0.0, 0.0, -1.0]);
    const isPushActive = this.stepCount < this.pushEndStep;

    const footPositions: [number, number, number][] = [];
    const footContacts: boolean[] = [];

    for (let leg = 0; leg < 4; leg++) {
      const qHip = this.jointPos[leg * 3 + 0];
      const qThigh = this.jointPos[leg * 3 + 1];
      const qCalf = this.jointPos[leg * 3 + 2];

      const localX =
        HIP_OFFSETS[leg][0] -
        LINK_LEN * Math.sin(qThigh) -
        LINK_LEN * Math.sin(qThigh + qCalf);
      const downDist =
        LINK_LEN * Math.cos(qThigh) + LINK_LEN * Math.cos(qThigh + qCalf);
      const localY =
        HIP_OFFSETS[leg][1] + HIP_LAT_OFFSETS[leg] + downDist * Math.sin(qHip);
      const localZ = -downDist * Math.cos(qHip);

      const worldOffset = quatRotate(quat, [localX, localY, localZ]);
      const fx = this.basePos[0] + worldOffset[0];
      const fy = this.basePos[1] + worldOffset[1];
      const fz = Math.max(0.018, this.basePos[2] + worldOffset[2]);
      footPositions.push([fx, fy, fz]);
      footContacts.push(fz <= 0.036);
    }

    let powerW = 0.0;
    for (let i = 0; i < 12; i++) {
      powerW += Math.abs(this.lastTorques[i] * this.jointVel[i]);
    }

    const velErr = Math.hypot(
      this.baseLinVelBody[0] - this.velCmd[0],
      this.baseLinVelBody[1] - this.velCmd[1]
    );

    return {
      step: this.stepCount,
      time_s: Number((this.stepCount * 0.02).toFixed(2)),
      policy_name: this.policyName,
      base_pos: [...this.basePos],
      base_quat_wxyz: quat,
      roll_deg: Number(((this.rpy[0] * 180) / Math.PI).toFixed(2)),
      pitch_deg: Number(((this.rpy[1] * 180) / Math.PI).toFixed(2)),
      yaw_deg: Number(((this.rpy[2] * 180) / Math.PI).toFixed(2)),
      upright_cos: Number((-gravBody[2]).toFixed(4)),
      base_lin_vel: [
        Number(this.baseLinVelBody[0].toFixed(3)),
        Number(this.baseLinVelBody[1].toFixed(3)),
        Number(this.baseLinVelBody[2].toFixed(3)),
      ],
      base_ang_vel: [
        Number(this.baseAngVelBody[0].toFixed(3)),
        Number(this.baseAngVelBody[1].toFixed(3)),
        Number(this.baseAngVelBody[2].toFixed(3)),
      ],
      vel_cmd: [...this.velCmd],
      joint_pos: Array.from(this.jointPos).map((v) => Number(v.toFixed(4))),
      joint_vel: Array.from(this.jointVel).map((v) => Number(v.toFixed(4))),
      torques: Array.from(this.lastTorques).map((v) => Number(v.toFixed(2))),
      action: Array.from(this.lastAction).map((v) => Number(v.toFixed(3))),
      foot_positions: footPositions,
      foot_contacts: footContacts,
      external_force: isPushActive ? [...this.activeForce] : [0, 0, 0],
      external_torque: isPushActive ? [...this.activeTorque] : [0, 0, 0],
      mechanical_power_w: Number(powerW.toFixed(1)),
      vel_tracking_error: Number(velErr.toFixed(3)),
      step_reward: Number(this.lastStepReward.toFixed(3)),
      cumulative_return: Number(this.cumulativeReturn.toFixed(1)),
      critic_value: Number(this.lastCriticValue.toFixed(2)),
      recovery_time_s: this.lastRecoveryTimeS,
      terminated: this.terminated,
      fall_count: this.fallCount,
    };
  }
}
