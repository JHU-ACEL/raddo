from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import optax

from .learning import infinite_dataloader

""" Defines core online-adaptation function"""
@eqx.filter_jit
def last_layer_update(
    model: eqx.Module,
    model_inputs: jax.Array,
    measurements: jax.Array,
    batch_size: int=32,
    lr: float=0.1,
    steps: int=200,
    key: jax.Array=jrandom.key(123),
) -> eqx.Module:
  # Based off of (https://docs.kidger.site/equinox/examples/frozen_layer/)

  # Define dataloader to get random batches
  dataloader = infinite_dataloader((model_inputs, measurements), batch_size, key)

  # Filter to freeze everything except last layer
  filter_spec = jtu.tree_map(lambda _: False, model)
  filter_spec = eqx.tree_at(
      lambda tree: (tree.layers[-1].weight, tree.layers[-1].bias),
      filter_spec,
      replace=(True, True),
  )

  # Optimizaiton setup
  params = eqx.filter(model, eqx.is_array)
  mask_decay = jtu.tree_map(lambda p: eqx.is_array(p) and (p.ndim >= 2), params)
  optim = optax.chain(
          optax.clip_by_global_norm(1.0),
          optax.adamw(
              learning_rate=lr, 
              weight_decay=.001, 
              mask=mask_decay
          ) 
      )
  trainable_params = eqx.filter(model, filter_spec)
  opt_state = optim.init(trainable_params)

  @eqx.filter_jit
  def adapt_step(
      model: eqx.Module,
      x: jax.Array,
      y: jax.Array,
      opt_state: optax.OptState,
  ) -> tuple[eqx.Module, optax.OptState]:
      @eqx.filter_grad
      def loss(
          diff_model: eqx.Module,
          static_model: eqx.Module,
          x: jax.Array,
          y: jax.Array,
      ) -> jax.Array:
          model = eqx.combine(diff_model, static_model)
          pred_y = jax.vmap(model)(x)
          return jnp.mean((y - pred_y) ** 2)
  
      diff_model, static_model = eqx.partition(model, filter_spec)
      grads = loss(diff_model, static_model, x, y)
      params = eqx.filter(diff_model, eqx.is_array)    
      updates, opt_state = optim.update(grads, opt_state, params)
      model = eqx.apply_updates(model, updates)
      return model, opt_state
  
  # Train (fine-tuning)
  for step, (x, y) in zip(range(steps), dataloader):
      model, opt_state = adapt_step(model, x, y, opt_state)
  return model