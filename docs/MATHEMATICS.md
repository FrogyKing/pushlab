# Mathematical Formulation of PushLab

## 1. Robust Markov Decision Process Formulation

PushLab models quadruped locomotion under stochastic external disturbances as an infinite-horizon discounted Markov Decision Process (MDP):

$$\mathcal{M} = \left(\mathcal{S}, \mathcal{O}, \mathcal{A}, P_{\mathcal{D}}, r, \rho_0, \gamma\right)$$

where:

- **State Space ($\mathcal{S} \subseteq \mathbb{R}^{37}$):** Full MuJoCo generalized coordinates $q_t \in \mathbb{R}^{19}$ (3D trunk position $p_t \in \mathbb{R}^3$, unit orientation quaternion $\mathbf{q}_t = (w, x, y, z) \in S^3$, and 12 joint angles $q_{j,t} \in \mathbb{R}^{12}$) paired with generalized velocities $v_t \in \mathbb{R}^{18}$ (linear velocity $\dot{p}_t \in \mathbb{R}^3$, angular velocity $\omega_t \in \mathbb{R}^3$, and joint velocities $\dot{q}_{j,t} \in \mathbb{R}^{12}$).
- **Proprioceptive Observation Space ($\mathcal{O} \subseteq \mathbb{R}^{50}$):** Neither the baseline policy nor the robust policy observes the external disturbance wrench $d_t$ directly. The policy receives a 50-dimensional proprioceptive observation vector $o_t \in \mathbb{R}^{50}$:

$$o_t = \begin{bmatrix}
g_t^{\text{body}} \in \mathbb{R}^3 \\
v_t^{\text{body}} \in \mathbb{R}^3 \\
0.25\,\omega_t^{\text{body}} \in \mathbb{R}^3 \\
v_t^{\text{cmd}} \in \mathbb{R}^3 \\
\sin(2\pi \phi_t), \cos(2\pi \phi_t) \in \mathbb{R}^2 \\
q_{j,t} - q_{\text{default}} \in \mathbb{R}^{12} \\
0.05\,\dot{q}_{j,t} \in \mathbb{R}^{12} \\
a_{t-1} \in \mathbb{R}^{12}
\end{bmatrix}$$

where the world-to-body frame rotation of any vector $\mathbf{u} \in \mathbb{R}^3$ by unit quaternion $\mathbf{q} = (w, \mathbf{q}_v)$ is evaluated as:

$$R(\mathbf{q})^T \mathbf{u} = (2w^2 - 1)\mathbf{u} - 2w(\mathbf{q}_v \times \mathbf{u}) + 2(\mathbf{q}_v \cdot \mathbf{u})\mathbf{q}_v$$

- **Action Space ($\mathcal{A} \subseteq \mathbb{R}^{12}$):** The policy outputs a 12-dimensional action vector $a_t \in \mathbb{R}^{12}$ corresponding to hip abduction, thigh flexion, and knee flexion residual targets for the four legs (`FL`, `FR`, `RL`, `RR`).
- **PD Actuation Law:** Actions modulate a diagonal trotting phase generator $q_{\text{phase}}(\phi_t, v_t^{\text{cmd}})$ at frequency $f_{\text{gait}} = 2.2\text{ Hz}$:

$$q_{\text{target}}(t) = q_{\text{phase}}(\phi_t, v_t^{\text{cmd}}) + \alpha_{\text{act}} \tanh(\mathrm{clip}(a_t, -2.5, 2.5)), \quad \alpha_{\text{act}} = 0.35\text{ rad}$$

Joint torques are computed at $200\text{ Hz}$ ($\Delta t_{\text{sim}} = 0.005\text{ s}$, decimation $K = 4$ substeps per $50\text{ Hz}$ policy step) via proportional-derivative feedback:

$$\tau_t = \mathrm{clip}\left(k_p \left(q_{\text{target}}(t) - q_{j,t}\right) - k_v \dot{q}_{j,t},\ -\tau_{\max},\ +\tau_{\max}\right)$$

with $k_p = 35.0\text{ N}\cdot\text{m/rad}$, $k_v = 1.0\text{ N}\cdot\text{m}\cdot\text{s/rad}$, and $\tau_{\max} = 28.0\text{ N}\cdot\text{m}$.

---

## 2. External Disturbance Distribution ($d_t \sim \mathcal{D}$)

For the robust optimization objective:

$$\max_\theta \ \mathbb{E}_{d \sim \mathcal{D},\ \tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t, s_{t+1})\right], \quad \gamma = 0.99$$

an external 6-DOF wrench $d_t = (\mathbf{F}_t, \boldsymbol{\tau}_t) \in \mathbb{R}^6$ is applied to the robot trunk (`xfrc_applied`) over a window of $L_{\text{push}} = 8$ control steps ($0.16\text{ s}$) at stochastic intervals $\Delta N \sim \mathcal{U}(36, 54)$ steps:

$$F_{xy} \sim \mathcal{U}(F_{\min}, F_{\max}), \quad \psi \sim \mathcal{U}(-\pi, \pi), \quad F_z \sim \mathcal{U}(-0.15 F_{xy}, +0.15 F_{xy})$$

$$\mathbf{F}_t = \begin{bmatrix} F_{xy}\cos\psi \\ F_{xy}\sin\psi \\ F_z \end{bmatrix}, \quad \boldsymbol{\tau}_t = \tau_{\text{mag}} \frac{\boldsymbol{\xi}}{\|\boldsymbol{\xi}\|_2}, \quad \tau_{\text{mag}} \sim \mathcal{U}(\tau_{\min}, \tau_{\max}), \ \boldsymbol{\xi} \sim \mathcal{N}(0, I_3)$$

- **Baseline PPO:** $F_{\min} = F_{\max} = 0\text{ N}$, $\tau_{\min} = \tau_{\max} = 0\text{ N}\cdot\text{m}$, nominal friction and actuator gains.
- **Robust PPO:** $F_{xy} \sim \mathcal{U}(20.0, 72.0)\text{ N}$ ($0.34g$ to $1.21g$ lateral acceleration on the $6.06\text{ kg}$ quadruped), $\tau_{\text{mag}} \sim \mathcal{U}(1.0, 6.5)\text{ N}\cdot\text{m}$, plus domain randomization over ground friction $\mu_{\text{scale}} \sim \mathcal{U}(0.70, 1.30)$ and actuator stiffness $k_{p,\text{scale}} \sim \mathcal{U}(0.88, 1.12)$.

---

## 3. Reward Function Design & Capture-Point Deadzone

To prevent reward hacking while encouraging both smooth nominal locomotion and rapid post-push stabilization, both policies are trained with the exact same reward function:

$$r_t = \Delta t_{\text{ctrl}} \cdot 10 \cdot \left(
w_v r_{\text{lin}} + w_\omega r_{\text{ang}} + w_u r_{\text{upright}} + w_h r_{\text{height}} + w_g r_{\text{gait}}
+ w_{\text{alive}}(1 - \mathbb{I}_{\text{term}}) + w_{\text{term}}\mathbb{I}_{\text{term}} - \mathcal{P}_{\text{reg}}
\right)$$

where:

$$r_{\text{lin}} = \exp\left(-\frac{\|v_{xy}^{\text{body}} - v_{xy}^{\text{cmd}}\|_2^2}{0.12}\right), \quad
r_{\text{ang}} = \exp\left(-\frac{(\omega_z^{\text{body}} - \omega_z^{\text{cmd}})^2}{0.18}\right)$$

$$r_{\text{upright}} = \exp\left(-\frac{(g_x^{\text{body}})^2 + (g_y^{\text{body}})^2}{0.08}\right), \quad
r_{\text{height}} = \exp\left(-\frac{(z_{\text{base}} - 0.29)^2}{0.0035}\right)$$

During external perturbations that exceed normal trotting sway (governed by deadzone operator $\mathcal{Z}(x, \tau) = \mathrm{sign}(x)\max(0, |x| - \tau)$ with $\tau_{v_y} = 0.18\text{ m/s}$ and $\tau_{\omega_x} = 0.70\text{ rad/s}$), the capture-point stabilization offset $\Delta q_{\text{cp}}(v_t^{\text{body}}, \omega_t^{\text{body}}, v_t^{\text{cmd}})$ specifies widened hip abduction and asymmetric outer-knee extension:

$$\Delta q_{\text{hip}}^{\text{cp}} = \mathrm{clip}\left(0.55\,\mathcal{Z}(v_y^{\text{body}} - v_y^{\text{cmd}}, 0.18) - 0.15\,\mathcal{Z}(\omega_x^{\text{body}}, 0.70), -0.35, 0.35\right)$$

$$\Delta q_{\text{knee}}^{\text{cp}} = \mathrm{clip}\left(0.24\,\mathcal{Z}(v_y^{\text{body}} - v_y^{\text{cmd}}, 0.18) - 0.08\,\mathcal{Z}(\omega_x^{\text{body}}, 0.70), -0.24, 0.24\right)$$

Because $\mathcal{Z}(\cdot) = 0$ during undisturbed walking, Baseline PPO optimizes pure steady-state gait efficiency, whereas Robust PPO frequently visits states outside the deadzone and learns closed-loop stabilization reflexes.

### Episode Termination Conditions

An episode terminates ($\mathbb{I}_{\text{term}} = 1$) if the trunk height falls below $z_{\min} = 0.16\text{ m}$ or the torso upright cosine $-g_z^{\text{body}} < 0.38$ ($\approx 67.7^\circ$ tilt).

---

## 4. Proximal Policy Optimization (PPO) & GAE

### Generalized Advantage Estimation

Given a trajectory rollout of length $T = 20$ across $N = 512$ parallel MJX environments, temporal-difference residuals and advantages are computed via reverse scan:

$$\delta_t = r_t + \gamma (1 - d_t) V_\phi(o_{t+1}) - V_\phi(o_t)$$

$$\hat{A}_t = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \left(\prod_{k=0}^{l-1} (1 - d_{t+k})\right) \delta_{t+l}, \quad \hat{R}_t = \hat{A}_t + V_\phi(o_t)$$

with $\gamma = 0.99$ and $\lambda = 0.95$. Within each minibatch, advantages are standardized:

$$\tilde{A}_t = \frac{\hat{A}_t - \mu_{\hat{A}}}{\sigma_{\hat{A}} + 10^{-8}}$$

### Clipped Surrogate Objective & Total Loss

With probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\theta_{\text{old}}}(a_t \mid o_t)}$, the clipped surrogate objective is:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\tilde{A}_t,\ \mathrm{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\tilde{A}_t\right)\right], \quad \epsilon = 0.20$$

Combined with clipped value function regression and Gaussian differential entropy $\mathcal{H}(\pi_\theta) = \sum_{i=1}^{12} \left(\log\sigma_i + \frac{1}{2}\log(2\pi e)\right)$:

$$\mathcal{L}_{\text{PPO}}(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_v \cdot \frac{1}{2}\mathbb{E}_t\left[\max\left((V_\phi(o_t) - \hat{R}_t)^2,\ (V_{\text{clip}}(o_t) - \hat{R}_t)^2\right)\right] - c_h \mathbb{E}_t[\mathcal{H}(\pi_\theta(\cdot \mid o_t))]$$

with $c_v = 0.50$, $c_h = 0.005$, and global gradient norm clipping $\|\nabla \mathcal{L}\|_2 \le 1.0$.
