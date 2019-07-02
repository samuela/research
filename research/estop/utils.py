def Scalarify():
  def init_fn(_rng, input_shape):
    assert input_shape[-1] == 1
    return input_shape[:-1], ()

  def apply_fn(_, inputs, **_2):
    return inputs[..., 0]

  return init_fn, apply_fn

Scalarify = Scalarify()
