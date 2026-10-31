from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

# Skew symmetric map (lie algebra of SO(3), solves a x b = S(a)b)
def S(u: jax.Array) -> jax.Array:
    return jnp.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

# Left quaternion product (used in the quaternion dynamics)
def q_left(q: jax.Array) -> jax.Array:
  qs = q[0]
  qv = q[1:]
  qL_A = jnp.concatenate((jnp.array([[qs]]), -qv.reshape(-1, 1)), axis=0).T
  qL_B = jnp.concatenate((qv.reshape(-1,1),  jnp.array(qs*jnp.eye(3) + S(qv))), axis=1)
  return jnp.concatenate((qL_A, qL_B), axis=0)

# Conjugate quaternion (used in the quaternion dynamics)
def q_conj(q: jax.Array) -> jax.Array:
    return jnp.concatenate((q[:1], -q[1:]))

# Convert the quaternion into a rotation (rotates vector r as:  r' = get_rotation(q) @ r)
# converts vector from body -> inertial frame (use q_conj(q) as input to go from inertial -> body)
def get_rotation(q: jax.Array) -> jax.Array:
  qw, qx, qy, qz = q
  qw2 = qw * qw
  qx2 = qx * qx
  qy2 = qy * qy
  qz2 = qz * qz

  norm_sq = qw2 + qx2 + qy2 + qz2 
  
  return jnp.array(
      [
          [qw2 + qx2 - qy2 - qz2, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
          [2 * (qx * qy + qw * qz), qw2 - qx2 + qy2 - qz2, 2 * (qy * qz - qw * qx)],
          [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw2 - qx2 - qy2 + qz2],
      ]
  ) / norm_sq


def q_to_mrp(q: jax.Array, qbar: jax.Array) -> jax.Array:
  e_scalar = qbar[0] * q[0] + jnp.dot(qbar[1:], q[1:])
  e_vector = qbar[0] * q[1:] - q[0] * qbar[1:] - jnp.cross(qbar[1:], q[1:])
  
  # 2. Setup standard and shadow denominators safely
  denom_standard = jnp.where(1.0 + e_scalar > 1e-10, 1.0 + e_scalar, 1e-10)
  denom_shadow = jnp.where(1.0 - e_scalar > 1e-10, 1.0 - e_scalar, 1e-10)
  
  # 3. Calculate both options
  phi_standard = e_vector / denom_standard
  phi_shadow = -e_vector / denom_shadow
  
  # 4. Switch to shadow MRP if scalar error is negative to avoid singularity
  return jnp.where(e_scalar >= 0.0, phi_standard, phi_shadow)

def mrp_to_q(phi: jax.Array, qbar: jax.Array) -> jax.Array:
  qbar = qbar / jnp.linalg.norm(qbar)
  
  phi_norm_sq = jnp.sum(phi**2)
  
  # If shadow (||phi|| > 1), map back to standard form
  is_shadow = phi_norm_sq > 1.0
  phi_mapped = jnp.where(is_shadow, -phi / (phi_norm_sq + 1e-12), phi)
  phi_mapped_norm_sq = jnp.sum(phi_mapped**2)
  
  # Standard MRP to quaternion formula
  denom = 1.0 + phi_mapped_norm_sq
  e_scalar = (1.0 - phi_mapped_norm_sq) / denom
  e_vector = (2.0 * phi_mapped) / denom
  
  # No sign flip needed - phi_mapped is always in standard form now
  q_error = jnp.concatenate([jnp.array([e_scalar]), e_vector])
  
  q = q_left(qbar) @ q_error
  return q


class QuatMRP:
    """
    Mixin providing quaternion <-> MRP error-coordinate utilities and
    attitude-aware linearization/discretization.

    Based on "Planning with Attitude" - Jackson, Tracy, Manchester

    Expects the consuming class to provide:
        - self.params["quat_start"]
        - self.params["num_states"]
        - self.state_dot(x, u, t, ext_param)
    """

    def get_error_coords(self, x: jax.Array, xbar: jax.Array) -> jax.Array:
        qs = self.params["quat_start"]
        d_pre = x[:qs] - xbar[:qs]
        dq = q_to_mrp(x[qs : qs + 4], xbar[qs : qs + 4])
        d_post = x[qs + 4 :] - xbar[qs + 4 :]
        return jnp.concatenate((d_pre, dq, d_post))

    def get_true_coords(self, dx: jax.Array, xbar: jax.Array) -> jax.Array:
        qs = self.params["quat_start"]
        x_pre = dx[:qs] + xbar[:qs]
        q = mrp_to_q(dx[qs : qs + 3], xbar[qs : qs + 4])
        x_post = dx[qs + 3 :] + xbar[qs + 4 :]
        return jnp.concatenate((x_pre, q, x_post))

    def E(self, x: jax.Array) -> jax.Array:
        nx = x.shape[0]
        qs = self.params["quat_start"]

        q1, q2, q3, q4 = x[qs], x[qs + 1], x[qs + 2], x[qs + 3]
        G = jnp.array(
            [
                [-q2, -q3, -q4],
                [q1, -q4, q3],
                [q4, q1, -q2],
                [-q3, q2, q1],
            ]
        )

        E_mat = jnp.zeros((nx, nx - 1))
        E_mat = E_mat.at[:qs, :qs].set(jnp.eye(qs))
        E_mat = E_mat.at[qs : qs + 4, qs : qs + 3].set(G)
        E_mat = E_mat.at[qs + 4 :, qs + 3 :].set(jnp.eye(nx - qs - 4))
        return E_mat

    def linearize_and_discretize(
        self,
        nom_traj: jax.Array,
        nom_cntrl: jax.Array,
        ext_param: jax.Array,
        dt: ArrayLike,
    ) -> tuple[jax.Array, jax.Array]:
        nx = self.params["num_states"] - 1
        df_dx = jax.jacfwd(self.state_dot, argnums=0)
        df_du = jax.jacfwd(self.state_dot, argnums=1)

        def linearize_and_discretize_single(x, xp, u, ext_param):
            A = df_dx(x, u, 0.0, ext_param)
            B = df_du(x, u, 0.0, ext_param)

            A = self.E(xp).T @ A @ self.E(x)
            B = self.E(xp).T @ B

            Ad = jnp.eye(nx) + A * dt
            Bd = B * dt
            return Ad, Bd

        Ad, Bd = jax.vmap(linearize_and_discretize_single)(
            nom_traj[:-1], nom_traj[1:], nom_cntrl, ext_param
        )
        return Ad, Bd