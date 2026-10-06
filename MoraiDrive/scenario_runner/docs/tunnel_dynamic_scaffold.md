# Tunnel dynamic obstacle scaffold

`full_loop.tunnel_dynamic` is the entry point for a pedestrian crossing in
the tunnel. This branch contains only the dynamic obstacle side. Mission
selection across tunnel scenarios belongs outside this implementation.

An implementation must provide `validate(context)`, `arm(context)`, and
`update(context, elapsed, ego_state, current_s)`. `validate` checks tunnel
placement without changing MORAI. `arm` runs after the world starts, and
`update` runs on driving ticks. Connect a validated implementation through
`tunnel_dynamic.TUNNEL_DYNAMIC_HANDLER`.

The tunnel crossing geometry and actor behavior are not validated yet, so the
config stays disabled. Enabling it without a handler raises an error before
the simulator starts. The existing tunnel sudden-brake event remains active
when `tunnel_dynamic` is disabled.
