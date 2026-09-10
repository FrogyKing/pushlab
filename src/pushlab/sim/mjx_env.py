"""GPU-accelerated JAX/MJX vectorized quadruped environment for parallel PPO training."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from pushlab.config import ExperimentConfig
from pushlab.sim.disturbance import sample_disturbance_wrench_jax
from pushlab.sim.env import MODEL_XML_PATH
from pushlab.sim.rewards import compute_gait_reference_jax, compute_reward_jax


def quat_rotate_inverse_jax(quat_wxyz: jnp.ndarray, vec_xyz: jnp.ndarray) -> jnp.ndarray:
    """Rotate a 3D world vector into the body frame defined by unit quaternion (w, x, y, z)."""
    w = quat_wxyz[0]
    q_vec = quat_wxyz[1:4]
    a = vec_xyz * (2.0 * w * w - 1.0)
    b = 2.0 * w * jnp.cross(q_vec, vec_xyz)
    c = 2.0 * jnp.dot(q_vec, vec_xyz) * q_vec
    return a - b + c


class MJXEnvState(NamedTuple):
    """Immutable PyTree state for vectorized MJX quadruped simulation."""

    mjx_data: mjx.Data
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    terminated: jnp.ndarray
    step_count: jnp.ndarray
    phase_rad: jnp.ndarray
    vel_cmd: jnp.ndarray
    prev_action: jnp.ndarray
    kp_scale: jnp.ndarray
    active_wrench: jnp.ndarray
    push_end_step: jnp.ndarray
    next_push_step: jnp.ndarray
    episode_return: jnp.ndarray
    episode_length: jnp.ndarray
    completed_return: jnp.ndarray
    completed_length: jnp.ndarray
    rng: jax.Array


class MJXQuadrupedEnv:
    """Vectorized MuJoCo MJX environment running thousands of quadrupeds in parallel on GPU."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.sim_cfg = config.sim
        self.dist_cfg = config.disturbance
        self.reward_weights = config.rewards

        self.mj_model = mujoco.MjModel.from_xml_path(str(MODEL_XML_PATH))
        self.mj_model.opt.timestep = self.sim_cfg.physics_dt
        self.mj_data_init = mujoco.MjData(self.mj_model)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data_init, 0)
        mujoco.mj_forward(self.mj_model, self.mj_data_init)

        self.trunk_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        self.mjx_model = mjx.put_model(self.mj_model)
        self._template_data = mjx.put_data(self.mj_model, self.mj_data_init)
        self._qpos_home = jnp.array(self.mj_data_init.qpos, dtype=jnp.float32)
        self._q_default = jnp.array(self.sim_cfg.default_joint_pos, dtype=jnp.float32)

        self.vmap_reset = jax.jit(jax.vmap(self._reset_single))
        self.vmap_step = jax.jit(jax.vmap(self._step_single))

    @property
    def obs_dim(self) -> int:
        return self.sim_cfg.obs_dim

    @property
    def action_dim(self) -> int:
        return self.sim_cfg.action_dim

    def _compute_obs(
        self,
        data: mjx.Data,
        phase_rad: jnp.ndarray,
        vel_cmd: jnp.ndarray,
        prev_action: jnp.ndarray,
        rng: jax.Array,
    ) -> jnp.ndarray:
        quat = data.qpos[3:7]
        gravity_world = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
        gravity_body = quat_rotate_inverse_jax(quat, gravity_world)
        base_lin_vel_body = quat_rotate_inverse_jax(quat, data.qvel[0:3])
        base_ang_vel_body = quat_rotate_inverse_jax(quat, data.qvel[3:6])

        joint_pos_err = data.qpos[7:] - self._q_default
        joint_vel_scaled = data.qvel[6:] * 0.05
        clock = jnp.array([jnp.sin(phase_rad), jnp.cos(phase_rad)], dtype=jnp.float32)

        obs = jnp.concatenate(
            [
                gravity_body,  # 3
                base_lin_vel_body,  # 3
                base_ang_vel_body * 0.25,  # 3
                vel_cmd,  # 3
                clock,  # 2
                joint_pos_err,  # 12
                joint_vel_scaled,  # 12
                prev_action,  # 12
            ]
        )
        if self.dist_cfg.obs_noise_std > 0:
            obs = obs + jax.random.normal(rng, obs.shape) * self.dist_cfg.obs_noise_std
        return jnp.clip(obs, -10.0, 10.0)

    def _reset_single(self, rng: jax.Array) -> MJXEnvState:
        k_q, k_v, k_cmd, k_phase, k_kp, k_int, k_obs, k_next = jax.random.split(rng, 8)

        q_jitter = jax.random.uniform(
            k_q, (self.sim_cfg.action_dim,), minval=-0.025, maxval=0.025
        )
        qpos = self._qpos_home.at[7:].add(q_jitter)
        qvel = jax.random.uniform(
            k_v, (self.mj_model.nv,), minval=-0.03, maxval=0.03
        )
        xfrc_zero = jnp.zeros_like(self._template_data.xfrc_applied)
        data = self._template_data.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros((self.sim_cfg.action_dim,), dtype=jnp.float32),
            xfrc_applied=xfrc_zero,
        )

        vx = jax.random.uniform(
            k_cmd, (), minval=self.sim_cfg.vx_cmd_range[0], maxval=self.sim_cfg.vx_cmd_range[1]
        )
        k_vy, k_wz = jax.random.split(k_cmd)
        vy = jax.random.uniform(
            k_vy, (), minval=self.sim_cfg.vy_cmd_range[0], maxval=self.sim_cfg.vy_cmd_range[1]
        )
        wz = jax.random.uniform(
            k_wz, (), minval=self.sim_cfg.wz_cmd_range[0], maxval=self.sim_cfg.wz_cmd_range[1]
        )
        vel_cmd = jnp.array([vx, vy, wz], dtype=jnp.float32)

        phase_rad = jax.random.uniform(k_phase, (), minval=0.0, maxval=2.0 * jnp.pi)
        prev_action = jnp.zeros((self.sim_cfg.action_dim,), dtype=jnp.float32)

        if self.dist_cfg.domain_randomization:
            kp_scale = jax.random.uniform(
                k_kp,
                (),
                minval=self.dist_cfg.kp_scale_range[0],
                maxval=self.dist_cfg.kp_scale_range[1],
            )
        else:
            kp_scale = jnp.array(1.0, dtype=jnp.float32)

        base_int = max(15, self.dist_cfg.push_interval_steps)
        first_push = jax.random.randint(
            k_int, (), minval=base_int // 2, maxval=base_int + 10
        )

        obs = self._compute_obs(data, phase_rad, vel_cmd, prev_action, k_obs)

        return MJXEnvState(
            mjx_data=data,
            obs=obs,
            reward=jnp.array(0.0, dtype=jnp.float32),
            done=jnp.array(False),
            terminated=jnp.array(False),
            step_count=jnp.array(0, dtype=jnp.int32),
            phase_rad=phase_rad,
            vel_cmd=vel_cmd,
            prev_action=prev_action,
            kp_scale=kp_scale,
            active_wrench=jnp.zeros((6,), dtype=jnp.float32),
            push_end_step=jnp.array(-1, dtype=jnp.int32),
            next_push_step=first_push.astype(jnp.int32),
            episode_return=jnp.array(0.0, dtype=jnp.float32),
            episode_length=jnp.array(0, dtype=jnp.int32),
            completed_return=jnp.array(0.0, dtype=jnp.float32),
            completed_length=jnp.array(0, dtype=jnp.int32),
            rng=k_next,
        )

    def _step_single(self, state: MJXEnvState, action: jnp.ndarray) -> MJXEnvState:
        k_push, k_int, k_obs, k_reset, k_next = jax.random.split(state.rng, 5)

        clipped_action = jnp.clip(action, -2.5, 2.5)
        action_offset = self.sim_cfg.action_scale * jnp.tanh(clipped_action)
        q_phase_ref = compute_gait_reference_jax(
            state.phase_rad, state.vel_cmd, self.sim_cfg
        )
        q_target = q_phase_ref + action_offset

        # Disturbance wrench schedule
        start_new_push = jnp.logical_and(
            jnp.array(self.dist_cfg.enabled),
            state.step_count >= state.next_push_step,
        )
        sampled_wrench = sample_disturbance_wrench_jax(k_push, self.dist_cfg)
        active_wrench = jnp.where(start_new_push, sampled_wrench, state.active_wrench)
        push_end_step = jnp.where(
            start_new_push,
            state.step_count + self.dist_cfg.push_duration_steps,
            state.push_end_step,
        )
        base_int = max(15, self.dist_cfg.push_interval_steps)
        jitter = jax.random.randint(k_int, (), minval=-base_int // 5, maxval=base_int // 5 + 1)
        next_push_step = jnp.where(
            start_new_push,
            state.step_count + base_int + jitter,
            state.next_push_step,
        )

        is_push_active = jnp.logical_and(
            jnp.array(self.dist_cfg.enabled),
            state.step_count < push_end_step,
        )
        applied_wrench = jnp.where(
            is_push_active, active_wrench, jnp.zeros((6,), dtype=jnp.float32)
        )

        xfrc = jnp.zeros_like(state.mjx_data.xfrc_applied)
        xfrc = xfrc.at[self.trunk_body_id].set(applied_wrench)
        data_in = state.mjx_data.replace(xfrc_applied=xfrc)

        kp_eff = self.sim_cfg.kp * state.kp_scale
        kv_eff = self.sim_cfg.kv

        def physics_substep(
            carry: tuple[mjx.Data, jnp.ndarray], _: None
        ) -> tuple[tuple[mjx.Data, jnp.ndarray], None]:
            d, tau_sum = carry
            q_cur = d.qpos[7:]
            dq_cur = d.qvel[6:]
            tau = jnp.clip(kp_eff * (q_target - q_cur) - kv_eff * dq_cur, -28.0, 28.0)
            d_ctrl = d.replace(ctrl=tau)
            d_next = mjx.step(self.mjx_model, d_ctrl)
            return (d_next, tau_sum + tau), None

        (data_out, tau_total), _ = jax.lax.scan(
            physics_substep,
            (data_in, jnp.zeros((self.sim_cfg.action_dim,), dtype=jnp.float32)),
            None,
            length=self.sim_cfg.decimation,
        )
        mean_torques = tau_total / float(self.sim_cfg.decimation)

        new_step = state.step_count + 1
        new_phase = jnp.mod(
            state.phase_rad + 2.0 * jnp.pi * self.sim_cfg.gait_freq_hz * self.sim_cfg.ctrl_dt,
            2.0 * jnp.pi,
        )

        quat = data_out.qpos[3:7]
        gravity_world = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32)
        gravity_body = quat_rotate_inverse_jax(quat, gravity_world)
        base_lin_vel_body = quat_rotate_inverse_jax(quat, data_out.qvel[0:3])
        base_ang_vel_body = quat_rotate_inverse_jax(quat, data_out.qvel[3:6])
        base_pos = data_out.qpos[0:3]
        upright_cos = -gravity_body[2]

        finite_ok = jnp.all(jnp.isfinite(data_out.qpos)) & jnp.all(jnp.isfinite(data_out.qvel))
        terminated = (
            (base_pos[2] < self.sim_cfg.min_base_height)
            | (upright_cos < self.sim_cfg.min_upright_cos)
            | (~finite_ok)
        )
        truncated = new_step >= self.sim_cfg.max_episode_steps
        done = terminated | truncated

        reward, _ = compute_reward_jax(
            base_pos=base_pos,
            gravity_body=gravity_body,
            base_lin_vel_body=base_lin_vel_body,
            base_ang_vel_body=base_ang_vel_body,
            joint_pos=data_out.qpos[7:],
            q_phase_ref=q_phase_ref,
            vel_cmd=state.vel_cmd,
            action=clipped_action,
            prev_action=state.prev_action,
            torques=mean_torques,
            terminated=terminated,
            sim_cfg=self.sim_cfg,
            weights=self.reward_weights,
        )
        reward = jnp.where(finite_ok, reward, jnp.array(-5.0, dtype=jnp.float32))

        obs_next = self._compute_obs(data_out, new_phase, state.vel_cmd, clipped_action, k_obs)
        ep_ret = state.episode_return + reward
        ep_len = state.episode_length + 1

        # Auto-reset if episode finished
        reset_state = self._reset_single(k_reset)

        def select_tree(reset_leaf: jnp.ndarray, cont_leaf: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(done, reset_leaf, cont_leaf)

        final_data = jax.tree_util.tree_map(select_tree, reset_state.mjx_data, data_out)
        final_obs = jnp.where(done, reset_state.obs, obs_next)

        return MJXEnvState(
            mjx_data=final_data,
            obs=final_obs,
            reward=reward,
            done=done,
            terminated=terminated,
            step_count=jnp.where(done, reset_state.step_count, new_step),
            phase_rad=jnp.where(done, reset_state.phase_rad, new_phase),
            vel_cmd=jnp.where(done, reset_state.vel_cmd, state.vel_cmd),
            prev_action=jnp.where(done, reset_state.prev_action, clipped_action),
            kp_scale=jnp.where(done, reset_state.kp_scale, state.kp_scale),
            active_wrench=jnp.where(done, reset_state.active_wrench, active_wrench),
            push_end_step=jnp.where(done, reset_state.push_end_step, push_end_step),
            next_push_step=jnp.where(done, reset_state.next_push_step, next_push_step),
            episode_return=jnp.where(done, jnp.array(0.0, dtype=jnp.float32), ep_ret),
            episode_length=jnp.where(done, jnp.array(0, dtype=jnp.int32), ep_len),
            completed_return=jnp.where(done, ep_ret, state.completed_return),
            completed_length=jnp.where(done, ep_len, state.completed_length),
            rng=k_next,
        )

    def reset(self, rng: jax.Array, num_envs: int) -> MJXEnvState:
        keys = jax.random.split(rng, num_envs)
        return self.vmap_reset(keys)

    def step(self, state: MJXEnvState, actions: jnp.ndarray) -> MJXEnvState:
        return self.vmap_step(state, actions)
