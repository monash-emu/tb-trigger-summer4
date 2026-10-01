"""TB model with prevalence-triggered interventions, built on summer4."""

import jax

# summer4 leaves precision to the caller. The trigger switch is a fast transition next to slow TB
# dynamics, and adaptive step control is more reliable in float64.
jax.config.update("jax_enable_x64", True)
