# watchdog_safety_validation DUT specification

## 1. Overview

This held-out DUT is a configurable windowed watchdog and safety supervisor.
Once enabled, it advances through CLOSED_WINDOW, OPEN_WINDOW, PRETIMEOUT, and
RESET_PENDING states. A correct two-key service sequence is accepted only in
the legal time window. Early service, malformed keys, and external safety
faults accumulate; enough faults force an irreversible LOCKED state until
external reset.

This DUT differs from the public SPI/DMA designs and the other held-out DUTs:
coverage is driven by elapsed time, ordered authentication, irreversible state,
and safety escalation rather than addresses, transfers, or prediction tables.

Hidden evaluation parameters:

- `service_key_a`, `service_key_b`: ordered 8-bit watchdog keys.
- `key_gap`: maximum cycles permitted between the two keys (1..8).
- `escalation_limit`: faults required for safety lockout (2..5).
- `reset_hold`: duration of the reset-pending state (1..6 cycles).

## 2. Action space (16 dims)

```text
action = [reg_we, reg_addr, d0, d1, d2, d3, tick, service,
          fault_inject, reset_n, pad0, pad1, pad2, pad3, pad4, pad5]
```

- Register 0 (`CTRL`): bit 0 enables the watchdog; bit 1 permanently locks
  configuration until external reset.
- Register 1 (`WINDOW`): lower 16 bits select the first legal service tick.
- Register 2 (`TIMEOUT`): lower 16 bits select the timeout tick.
- `service`: presents `d0` as the next service-key byte.
- `tick`: advances the watchdog counter while enabled.
- `fault_inject`: adds one asynchronous safety fault.
- `reset_n`: active-low external reset. Harness functional coverage persists.

Configuration writes are accepted only while disabled and not config-locked.
The effective constraints are `2 <= WINDOW < TIMEOUT-1` and `TIMEOUT <= 255`.

## 3. Coverage intent

The 59 bins cover all six states, eight action results, counter and fault
boundaries, both lock mechanisms, key phases, state/result crosses, and eight
multi-cycle sequences: enable, window opening, authenticated service, early
service, timeout reset, fault escalation, locked write rejection, and recovery
from safety lockout by external reset.
