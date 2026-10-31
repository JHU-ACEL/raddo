from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx

if TYPE_CHECKING:
    import jax


class Controller(eqx.Module):
  def __call__(
      self,
      state: jax.Array,
      target: jax.Array,
      key: jax.Array,
  ) -> jax.Array | tuple[jax.Array, jax.Array]:
    raise NotImplementedError("Must define a controller if no control sequence is given")