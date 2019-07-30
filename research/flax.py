"""Fake LAX. Python-native mocks of jax.lax.

See https://github.com/google/jax/issues/999."""

def fori_loop(lower, upper, body_fun, init_val):
  val = init_val
  for i in range(lower, upper):
    val = body_fun(i, val)
  return val
