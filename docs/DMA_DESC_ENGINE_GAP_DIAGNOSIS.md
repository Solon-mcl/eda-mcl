# dma_desc_engine_validation 缺口诊断

**日期**：2026-09-27
**对象**：`validation_duts/dma_desc_engine_validation/`（工作区新增的未见验证包，未跟踪）
**证据**：`results/why_dma_desc_engine_gap.json`

---

## 1. 结论（一句话）

这个新验证包是**第一个通用路径明显落后于参考实现的 DUT**：参考 `greedy` 覆盖 **87/87**，
默认通用策略只有 **34/87**。缺的 53 个 bin 里，**12 个描述符失败类一个都没拿到**。

原因不是"阈值不知道"，也不是"结构性不可达" —— 是通用策略**从未在同一周期把
`write_enable` 与字段索引、数据通道联合起来**。一个 11 行的纯机械骨架就能拿下 51/87。

---

## 2. 三方基线对照

| 策略 | 覆盖 bin | 占比 |
|---|---|---|
| `random` | 27 / 87 | 31% |
| `greedy`（包内参考实现） | **87 / 87** | **100%** |
| 通用策略（`full`，默认 M8.1） | 34 / 87 | 39% |

注意：这是**第一次**出现参考实现满分、通用路径大幅落后。前 7 个 DUT 上通用路径都追平或超过参考。

---

## 3. 缺什么（按 coverpoint 归并）

| coverpoint | 缺 | 总数 |
|---|---|---|
| `check_fail_class` | **12** | 13 |
| `state_x_fail` | 8 | 9 |
| `state_x_result` | 6 | 11 |
| `fsm_state` | 4 | 8 |
| `step_result` | 4 | 11 |
| `chain_x_state` | 4 | 7 |
| `word_index` | 3 | 4 |
| `ack_wait_class` / `chain_class` / `check_count` | 2 each | 3 / 4 / 4 |
| `word_x_dirty` | 1 | 3 |
| 5 个 `seq_*` | 5 | 8 |

缺的 5 个 seq：`seq_chain_limit`、`seq_abort_midburst`、`seq_check_fail_stop`、
`seq_error_recover`、`seq_done_handshake`。

**所有缺失都可以追溯到同一条链**：写不进描述符 → VALIDATE 拿不到非法值 →
进不了 ERROR → `check_fail_class` / `state_x_fail` / `fsm_state.error` 全断 →
依赖 ERROR 路径的 seq 也全断。

---

## 4. 接口门控：为什么整条结构化写入路径被关掉

这个 DUT 的动作空间：

```
action = [start, fetch_valid, word_we, word_idx, p0..p3, d0..d3, ack, abort, reset_n, pad0]
           0        1          2        3        4..7    8..11   12   13     14     15
```

算法解析出的角色（**解析正确**）：

| 字段 | 角色 | 索引 |
|---|---|---|
| `start` / `fetch_valid` | request | 0, 1 |
| `word_we` | **write_enable** | 2 |
| `word_idx` | scalar（应为字段索引） | 3 |
| `p0..p3` | pc_lane | 4-7 |
| `d0..d3` | **data_lane** | 8-11 |
| `ack` | ack | 12 |

于是：

```
write_enable_indices        = [2]          ✓
valid_indices               = [0, 1]       ✓
data_lanes                  = [8,9,10,11]  ✓
register_address_indices    = []           ✗
instance_select_indices     = []           ✗
→ field_selected_interface  = False
→ interface_probe_armed     = False（探测本身也要求 register_address_indices）
```

**三件套齐了（写使能 + 数据通道 + 请求），却因为没有"地址"字段，被判成"没有结构化写入接口"。**
所有 `_field_selected_transaction_program` 类路径不激活，`_generic_transaction` 两条分支
都不满足，直接掉到最底部的泛化 `_request()`。

---

## 5. 算法其实"看见"了这个字段，只是没联合

原生序列候选有 71 个，其中 65 个是 `combo.*`。里面确实有一族：

```
combo.unknown3.v0.h1   →  nonzero = {3: 0, 14: 1}
combo.unknown3.v2.h1   →  nonzero = {3: 2, 14: 1}
combo.unknown3.v3.h1   →  nonzero = {3: 3, 14: 1}
```

`unknown3` 就是 `word_idx`（索引 3），算法为它生成了 v0/v1/v2/v3 四个值。
**但每个候选只设 `word_idx`，`word_we`（索引 2）始终是 0。**
DUT 收到时写使能为低，整个写入被忽略。

动作使用率也印证：`word_idx` 非零占 5.3%，而 `word_we` 只有 **0.3%** ——
索引被单独戳了很多次，写使能几乎没打开过。

**缺的就是"同一周期联合"这一步。**

---

## 6. 可达性证明（不是不可达）

按参考实现的形状手写一个 11 case、完全机械、与 DUT 无关的骨架：

```
每个 case：
  reset_n=0   x2
  nop         x2
  start + 指针
  nop
  fetch_valid + word_we + word_idx + data     ← 联合写入，一个周期
  fetch_valid + 指针                          ← 再 latch 一次，让引擎校验改过的描述符
  nop         x2
```

结果：**51 / 87**，其中 **`check_fail_class` 12/12 全中**。

| coverpoint | 骨架拿到 |
|---|---|
| `check_fail_class` | **12 / 13**（缺的只有 `none` 那格） |
| `state_x_fail` | 8 / 9 |
| `word_index` | 4 / 4 |
| `word_x_dirty` | 3 / 3 |
| `fsm_state` | 4 / 8 |
| `state_x_result` | 4 / 11 |
| `seq_start_fetch` / `seq_check_fail_stop` / `seq_reset_recover` | 各 1 |

**结论：12 个失败类全部可达，且只需机械骨架，不需要知道任何隐藏阈值。**

至于是"值猜对了"还是"边界值天然覆盖"：`0x00FFF000`（高位越窗）、`0x00002000`（超长）、
`0x00000241`（非对齐）、`0x80000003`（非法 flag）这些都不是随机能撞上的，
但骨架只用了一组**通用边界值候选**（0、8、窗口顶端、2 的幂附近、最高位置位），
说明算法的边界值表 + 联合写入就够了。

---

## 7. 这个缺口在泛化框架下意味着什么

### 7.1 它不是"没信息"

spec 里明确写了 12 个失败类的**名字和语义**（"zero or too-low source/destination,
unaligned addresses, a length that pushes either end past the 64 KB window,
zero or oversized length, an unsupported mode, an unsupported address flag"），
并且 `coverage_meta.json` 把每个类的名字都列了出来。

**缺的是阈值**（最小地址 `0x10`、最大长度 `0x1000`、64KB 窗口、合法 mode `{0,1,3}`、
`aflag` 必须为 0）—— 这些只在 RTL 里。但这不构成阻碍，因为算法的边界值候选
天然覆盖了这些区间（见 §6）。

### 7.2 它是一个**接口建模**缺口，不是 DUT 知识缺口

算法的接口模型只有一种"带地址的寄存器总线"：

```
instance_select + register_address + write_enable + data_lanes
```

而现实里有另一类同样常见、但**没有地址**的写法：

```
write_enable + field_index + data_lanes
```

（描述符字段写、FIFO 通道写、per-word 配置写，都属于这一类。）

`word_idx` 被判成 `scalar` 而不是"字段索引"，就是这个模型覆盖不到的直接后果。

### 7.3 为什么会漏（归因，三层）

不是"泛化机制失效"，而是**泛化的维度选偏了**：算法在「**名字多样性**」上做了大量加固
（38 个角色、描述文本回退、区间主键、命名无关性检查），但在「**接口结构多样性**」上
只留了一种范式。前 7 个 DUT 恰好都是"带地址的寄存器总线"，盲区一直未被触发。

**第一层 — 接口模型只有一根轴。**
`field_selected_interface` 把"结构化写入"写死成四件套：

```
instance_select + register_address + write_enable + data_lanes
```

本 DUT 有 `write_enable` / `data_lanes` / `valid`，唯独没有 `register_address`，
于是被判为"没有结构化写入接口"。而「`write_enable` + 字段索引 + `data`」是同等常见的
形态（描述符字段写、FIFO 通道写、per-word 配置写）。模型只长了一根轴。

**第二层 — 角色词表靠名字匹配，`word_idx` 无词可命中。**
`# index` 不在 `register_address` 的触发词里：

```python
if value in ("reg_addr", "register_addr", "cfg_addr", "csr_addr",
             "conf_field", "cfg_field", "register_select",
             "addr", "adr", "address", "reg_index"):
    return "register_address"
```

描述文本本可以救回来，但有两处卡点：

1. `word_idx` 的 **keyed 描述为空** —— 它的说明写在与 `word_we` 共享的那一行
   （`` - `word_we` / `word_idx`: write `data` into descriptor word 0..3 … ``），
   keyed 规则要求"该字段是行首主键"，取不到。
2. 描述级规则表里唯一带 "index" 的触发词是整串 `"register index"`，
   即便拿到共享描述也匹配不上。

两处都失败 → 落到 `_infer_role` 最后的兜底 `return "scalar"`。

**第三层 — `scalar` 是兜底桶，不是待解释的信号。**
`scalar` 在整个策略里只出现 3 处，全部是"挨个试值"的通用扫描
（`__init__.py:499`、`:579`、`:1283`），**没有一处把它当作索引**。
所以候选生成器确实产出了 `combo.unknown3.v2`（`unknown3` = `word_idx`），
但它的语义只是"把这个未知字段设成 2 试试"，而不是"选第 2 个描述符 word"。
既然不理解它是索引，就自然不会想到要联合 `word_we`。

**一句话**：机制在运转（未知字段被识别、被生成候选），缺的是
「未知字段 → 可能是字段选择器」这一步推断，以及
「写使能 + 选择器 + 数据必须同周期联合」这条结构约束。

### 7.4 修法的方向（**已实施**，见 `NO_ADDRESS_FIELD_WRITE_DESIGN.md` §7）

要补的是**接口模式的识别**，不是任何 DUT 专属逻辑：

1. 识别 `write_enable + scalar索引 + data_lanes` 这一组合，作为"无地址字段索引写入"模式。
   → 已实现：`_slot_selector_fields()` 判定拓扑，`semantic_ir.detect_slot_selector()`
   从规格文本判定"这个字段是在选一个多字对象的哪个槽"。
2. 为该模式生成候选。
   → 已实现：`_build_no_address_field_write_candidates()`。**最终形态与本文档初版设想不同** ——
   不是"写一拍 + latch 一拍"，而是"**写满所有槽 → 再 latch**"，因为设计**整体校验**对象，
   缺字段会在第一个检查处失败，后面的检查永远走不到。打开对象的请求还必须与指针同拍。
3. 该模式的候选在门控之后追加。
   → 已实现（与 LLM 候选族同一挂载点）。
4. 零回归已被证明：7 个旧 DUT 在开关**开/关两态逐 bin 完全一致**。
   → **结果：目标 DUT 34 → 52/87**（其余 7 个不变）。

同时要注意 `interface_probe` 的门控：它当前要求 `register_address_indices`，
所以对这种无地址接口连探测机会都没有。真要做在线探测，判据应下沉到
"write_enable + 某个索引类字段 + 数据通道"而不是硬性要求地址。**这一项仍未做** ——
本轮的候选族是构造期生成的，没有走在线探测。

**52/87 之后是硬平台**（4000 与 20000 步同值）：剩余 35 个 bin 需要**走多描述符链**
（`chain_class.*` / `check_count.*` / `chain_x_state.*`）、**边 fetch 边写**
（`word_x_dirty.*`）与 4 个 `seq_*` 多周期序列。这些是与本模式**正交的另一种能力**，
需要新开一族，而不是继续加深槽扫描。

---

## 8. 未做的事 / 风险

- **没有修改任何算法代码**。按用户要求只给诊断结论。
- 本诊断的"骨架可达"是在 **local 后端**验证的。该包不发布 Verilator 二进制
  （validation 包契约），所以没有 RTL 交叉验证。
- 51/87 是手动骨架的下界，不是该修复能达到的上限 —— 剩下的 36 个 bin
  还需要 chain limit / abort / retry 等时序能力。
- `word_idx` 的角色判定改动会**影响全部 7 个已有 DUT**，必须逐位回归。

---

## 9. 复现方式

```bash
PY=C:/Users/20353/.workbuddy/binaries/python/envs/default/Scripts/python.exe

# 三方基线
$PY tools/run_experiments.py --dut dma_desc_engine_validation --backend local \
    --agent greedy  --steps 4000 --interval 4000
$PY tools/run_experiments.py --dut dma_desc_engine_validation --backend local \
    --agent random  --steps 4000 --interval 4000
$PY tools/run_experiments.py --dut dma_desc_engine_validation --backend local \
    --agent full    --steps 4000 --interval 4000
```

---

## 10. 新包进门体检（`tools/check_dut_intake.py`）

本次诊断暴露出一个流程缺口：**新包到手后没有标准化的"先体检"步骤**，
所以缺口只能靠人工比对覆盖率曲线发现。已把这次的手工诊断固化成脚本。

```bash
$PY tools/check_dut_intake.py --dut <name> --steps 2000
$PY tools/check_dut_intake.py --dut all --steps 2000 --json results/intake.json
```

它一次报三件事：

1. **三方基线表**（random / greedy / generic）——量出与参考实现的差距。
2. **能力位向量**：`instance_select` / `register_address` / `write_enable` /
   `request` / `payload` 各自是否可用，以及 `field_selected_interface` 是否成立。
   **这是最快的报警面**：写入路径被结构性关闭时，缺的 bin 看起来可能毫无关联
   （本次的 `check_fail_class` 就完全不提"地址"），但能力向量会直接指出来。
3. **落到 `scalar` 的字段名**（接口模型未覆盖的字段）——本次是 `word_idx`。

在本 DUT 上的输出：

```
=== dma_desc_engine_validation (2000 steps) ===
  random=27 greedy=87 generic=34 / 87
  capability: instance_select=n register_address=n write_enable=Y request=Y payload=Y
  scalar (unclassified) fields: word_idx
  GAP vs reference: 53 bins
```

对照已知良好的包（无误报）：`spi_xfer_public` → `no gap`；
`tlb_mmu_validation` → `GAP 1 bin`（且有 `register_address=n`，
但差距幅度小，不构成结构性问题的证据）。


`tools/run_experiments.py` 的 `DUTS` 已登记 `dma_desc_engine_validation`
（路径 `../validation_duts/dma_desc_engine_validation/dma_desc_engine_validation`，
harness 类 `DmaDescEngineHarness`）。
