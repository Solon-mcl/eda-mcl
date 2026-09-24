#!/usr/bin/env python3
import sys

import numpy as np

sys.path.insert(0, "/app")
from inference import InferenceInterface

agent = InferenceInterface("/dut/dut_spec.md", "/dut/covergroup.svh")
action = agent.predict(np.zeros(86, dtype=np.float32), 0, 30000)
assert action.shape == (12,)
assert action.dtype == np.float32
print(f"interface OK: shape={action.shape}, dtype={action.dtype}")
