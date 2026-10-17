import baselineCheckpoint from "../data/baseline_ppo.json";
import robustCheckpoint from "../data/robust_ppo.json";

export type PolicyName = "baseline_ppo" | "robust_ppo";

export interface LayerWeights {
  w: number[][];
  b: number[];
}

export interface PolicyCheckpointData {
  policy_name: string;
  metadata: Record<string, unknown>;
  obs_mean: number[];
  obs_var: number[];
  actor_layers: LayerWeights[];
  actor_head: LayerWeights;
  actor_log_std: number[];
  critic_layers: LayerWeights[];
  critic_head: LayerWeights;
}

function elu(x: number): number {
  return x > 0 ? x : Math.exp(Math.max(-20, x)) - 1;
}

function matVecMul(vec: Float32Array, w: number[][], b: number[], applyElu: boolean): Float32Array {
  const inDim = vec.length;
  const outDim = b.length;
  const out = new Float32Array(outDim);
  for (let j = 0; j < outDim; j++) {
    let sum = b[j];
    for (let i = 0; i < inDim; i++) {
      sum += vec[i] * w[i][j];
    }
    out[j] = applyElu ? elu(sum) : sum;
  }
  return out;
}

export class BrowserPolicyEngine {
  public readonly policyName: PolicyName;
  public readonly checkpoint: PolicyCheckpointData;
  private readonly obsMean: Float32Array;
  private readonly obsStdInv: Float32Array;

  constructor(policyName: PolicyName, data: PolicyCheckpointData) {
    this.policyName = policyName;
    this.checkpoint = data;
    this.obsMean = new Float32Array(data.obs_mean);
    this.obsStdInv = new Float32Array(
      data.obs_var.map((v) => 1.0 / Math.sqrt(Math.max(v, 0) + 1e-8))
    );
  }

  public normalizeObs(obs: Float32Array | number[]): Float32Array {
    const norm = new Float32Array(obs.length);
    for (let i = 0; i < obs.length; i++) {
      const z = (obs[i] - this.obsMean[i]) * this.obsStdInv[i];
      norm[i] = Math.max(-10.0, Math.min(10.0, z));
    }
    return norm;
  }

  public predict(obs: Float32Array | number[]): { action: Float32Array; value: number } {
    const norm = this.normalizeObs(obs);

    let xa = norm;
    for (const layer of this.checkpoint.actor_layers) {
      xa = matVecMul(xa, layer.w, layer.b, true);
    }
    const action = matVecMul(xa, this.checkpoint.actor_head.w, this.checkpoint.actor_head.b, false);

    let xc = norm;
    for (const layer of this.checkpoint.critic_layers) {
      xc = matVecMul(xc, layer.w, layer.b, true);
    }
    const valVec = matVecMul(xc, this.checkpoint.critic_head.w, this.checkpoint.critic_head.b, false);

    return {
      action,
      value: valVec[0] ?? 0,
    };
  }
}

export const POLICIES: Record<PolicyName, BrowserPolicyEngine> = {
  baseline_ppo: new BrowserPolicyEngine(
    "baseline_ppo",
    baselineCheckpoint as unknown as PolicyCheckpointData
  ),
  robust_ppo: new BrowserPolicyEngine(
    "robust_ppo",
    robustCheckpoint as unknown as PolicyCheckpointData
  ),
};
