# 无地址字段索引写入模式：设计与落地

**日期**：2026-09-27
**背景**：`docs/DMA_DESC_ENGINE_GAP_DIAGNOSIS.md`（该 DUT 通用 34/87 vs 参考 87/87）

---

## 1. 缺口的一句话复述

算法只建模了「带地址的寄存器总线」：

```
instance_select + register_address + write_enable + data_lanes
```

`dma_desc_engine_validation` 是「**无地址的字段索引写入**」：

```
write_enable(word_we) + 字段索引(word_idx) + data_lanes(d0..d3)
```

因为缺 `register_address` / `instance_select`，`field_selected_interface=False`，
所有结构化写入路径关闭，掉到最底泛化 `_request()`。

---

## 2. 关键事实（都已实测确认）

### 2.1 必须同周期联合

DUT 的 `_fetch` 语义（`local_sim.py:279`）：

```python
if request.get("word_we"):
    self.word_idx_write(word_idx, data)
    self.result = 8
    return                      # ← 写入后直接返回，不进 VALIDATE
```

→ 带 `word_we` 的那一周期**只做写入**；要让引擎校验改后的描述符，
**需要下一个周期再来一次 `fetch_valid`**。

所以正确形态是**两拍**：

```
周期 A: fetch_valid=1 + word_we=1 + word_idx=k + data=v   ← 联合，同周期
周期 B: fetch_valid=1                                      ← 让引擎进 VALIDATE
```

### 2.2 `_program_step` 已经支持同周期联合

```python
def _program_step(self, assignments=None, data=None, address=None,
                  valid=False, cycles=1):
```

`assignments` 与 `data` 写在同一个 action 上 → 现成可用，无需新机制。

### 2.3 现有候选为什么没命中

`_build_combinatorial_candidates` 的 anchor 段：

```python
self._program_step({index: maximum, anchor: 0})   # ← anchor=0 与 index 同周期
self._program_step({anchor: 1})                   # ← 才单独开 anchor
```

**先设 `anchor=0`，下一周期才开 `anchor=1`** —— 对"同周期生效"的写使能永不成成立。

而 `_build_fallback_candidates` 里其实**有**正确的同周期配对
（`{anchor: 1, item.index: high}`），但它只在**候选池全空**时才被调用
（`if not candidates:`），这里池子已有 71 个 → 从不触发。

**缺的不是机制，是一段用对了形状的候选生成。**

---

## 3. 影响面（决定零回归能否结构性成立）

实测 8 个 DUT 是否满足新模式的前置条件：

| DUT | write_enable | register_address | instance_select | data | 触发 |
|---|---|---|---|---|---|
| dma_xfer_public | Y | Y | Y | Y | no |
| spi_master_public | Y | Y | – | Y | no |
| spi_xfer_public | Y | Y | – | Y | no |
| cache_ctrl_validation | – | – | – | Y | no |
| branch_predictor_validation | – | – | – | – | no |
| watchdog_safety_validation | Y | Y | – | Y | no |
| tlb_mmu_validation | – | – | – | – | no |
| **dma_desc_engine_validation** | **Y** | **–** | **–** | **Y** | **YES** |

**只有目标 DUT 触发。** 其余 7 个因为"有地址"或"无写使能"被排除。

→ **零回归是结构性成立的**（不是靠调参躲开的），但仍需逐位验证。

---

## 4. 设计

### 4.1 触发条件（全部满足）

```python
write_enable_indices 非空
and not register_address_indices
and not instance_select_indices
and (data_lanes or register_data_indices)
```

再加一条**保守约束**：存在至少一个落在 `scalar` 的字段作为候选索引。
（若没有 scalar 字段可选，这个模式无事可做。）

### 4.2 生成的候选形态

对每个 `scalar` 字段（最多取 4 个，与现有 `extra` 一致）：

```python
# 主形态：写一拍 + latch 一拍（两拍序列）
steps = (
    self._program_step(
        {index: value, we: 1},           # 同周期联合写使能 + 索引
        data=probe_value,                # 同周期数据
        valid=True),                     # 视请求字段是否可用
    self._program_step(valid=True),      # 只 fetch，让引擎校验
)
```

值取 `_value_probes(minimum, maximum)`（复用现有边界值表，含 0 / 1 / 0xAA / 0xFF 等）。

`we` 取 `write_enable_indices[0]`。`data` 的探针值也走 `_value_probes`。

### 4.3 挂载位置（关键）

**必须在供给侧门控之后追加**（与 LLM 候选族同一约定）：

```python
if self.combinatorial_candidates_enabled and needs_generated_families:
    candidates.extend(self._build_combinatorial_candidates())
    candidates.extend(self._build_no_address_field_write_candidates())   # ← 这里
```

理由：门控是拿"schema 驱动 + 生成族"这一组校准的，
新族若在门控之前进池会改变门控看到的集合 → 已实测过一次 −1 的教训。

### 4.4 开关

`EDA_NO_ADDRESS_FIELD_WRITE`，默认 `1`（开）。
**关闭时该函数直接返回空列表**，保证零回归可验证。

### 4.5 分工边界（泛化纪律）

- 实现里**不得出现任何 DUT 专有名词**（不许出现 "word"、"descriptor"、"dma"）。
  只用角色与索引。
- 索引字段一律通过 `role == "scalar"` 选取，**不写字面 `word_idx`**。
- 新增任何规则后**逐条 grep 专有名词**（本项目已有两次教训）。

---

## 5. 验收标准

| 项 | 要求 |
|---|---|
| 目标 DUT | `dma_desc_engine_validation` 覆盖率显著上升（骨架下界 51/87，预期 ≥ 50） |
| 零回归 | 7 个已有 DUT 在开关关闭时 **逐位一致** |
| 开关开启 | 7 个已有 DUT 也应一致（因 §3 证明它们不触发；**这是更强的断言**） |
| 泛化尺子 | `check_parsing_generalization.py` 不退化 |
| 专有名词 | grep 无 DUT 家族词 |

## 6. 风险

- 新候选会**占用周期预算**。7 个 DUT 不触发所以无影响；
  但未来若有 DUT 同时满足条件，需要重新评估优先级。
- 若某 DUT 的写使能是**电平敏感**（非脉冲），两拍形态可能不适用 ——
  留给在线搜索淘汰（它会在两次尝试无覆盖后退休该候选）。

---

## 7. 实施结果（2026-09-27 收尾）

### 7.1 最终形态与初版完全不同（这是本节的重点）

初版设计（§4.2）的「写一拍 + latch 一拍」**买不到覆盖率**。逐周期复盘后确认了三处错误：

1. **必须写满所有 slot，而不是写一个。** 规格把 `word_idx` 说明成"多字对象的一个槽"
   （"Word 0 is the source address, word 1 the length, …"）。设计**整体校验**这个对象，
   缺字段就在第一个检查处失败，后面的检查永远走不到。
2. **打开对象的请求要与指针同拍。** `start` 那一拍采样 `desc_ptr`；指针晚一拍就是
   在旧地址上 arm，写入落到了错误的对象。
3. **latch 请求要连给两拍。** 设计在 strobe 拍存储、下一拍校验；请求只给一拍就撤，
   校验那一拍什么都看不到。

修正后的形态（`_slot_write_candidate`）：

```
reset(0) ×2  →  nop ×2  →  {arm: 1, ptr}  →  {ptr}
  →  逐 slot 写：{slot_index: i, strobe: 1} + data，受测 slot 放最后
  →  {latch: 1, ptr} ×2
```

### 7.2 实测数字

**本节的数字是 `no_address_field_write` 族单独落地时的状态（34 → 52）。**
随后 §8 的 `pointer_walk` 族把同一 DUT 推到 **85**；两族的合并结果见 §8.8。

| DUT | 改动前 | 本族开关关 | 本族开关开 |
|---|---|---|---|
| dma_xfer_public | 82 | 82 | 82 |
| spi_master_public | 72 | 72 | 72 |
| spi_xfer_public | 56 | 56 | 56 |
| tlb_mmu_validation | 78 | 78 | 78 |
| cache_ctrl_validation | 56 | 56 | 56 |
| branch_predictor_validation | 66 | 66 | 66 |
| watchdog_safety_validation | 47 | 47 | 47 |
| **dma_desc_engine_validation** | **34** | **34** | **52** |

- 逐 bin 向量比对：7 个已有 DUT 在**开关开/关两个状态下完全一致（0 个 bin 不同）**。
- 改动前代码（`git stash` 取基线）与开关关闭状态一致。
- `check_parsing_generalization.py`：PASS（角色召回 87/87，候选产出 26/26）。
- `check_name_invariance.py`：最差落差仍是 2 bin（对抗下界 `tlb_opaque`），未退化。

### 7.3 顺带修掉的两个解析缺陷

这两个都是**通用**缺陷，不是为这个 DUT 打的补丁：

1. **联合 bullet key 丢失**。规格写 `- \`word_we\` / \`word_idx\`: …`，
   而 `_BULLET_KEY` 要求 key 后面紧跟 `:`，`/` 直接让匹配失败 →
   **两个字段的 keyed description 都是空**，描述推断对它们完全失效。
   修法：新增 `_JOINT_KEY` + `_joint_key_names()`，让一个 bullet 归属于它列出的每个名字。
2. **bullet 的续行没被读**。Markdown 常规写法是首行写不完就换行续写，
   而 `keyed_description_for` 只看首行 → `"Word 0 is the source address, word 1 the length"`
   这段关键信息被截断。修法：匹配到的 bullet 之后，把后续非 bullet、非标题的行一并读入。

### 7.4 剩余 35 个 bin 是**另一种能力**，不是本模式的缺口 → 已另开一族（2026-09-27）

52/87 之后是硬平台（4000 与 20000 步同值，说明不是预算问题）。缺的是：

- `chain_class.*` / `check_count.*` / `chain_x_state.*` —— 需要**走一条含多个合法描述符的链**
- `word_x_dirty.*` —— 需要"边 fetch 边写"并跟踪 dirty 标志
- `seq_chain_limit` / `seq_abort_midburst` / `seq_error_recover` / `seq_done_handshake` ——
  8 个具名多周期序列中的 4 个

这些都是**多阶段、跨描述符**的程序，与本模式（单对象的槽写入）是正交的能力。

**已按此判断新开 `pointer_walk` 族**，见 §8。结果 **52 → 85/87**。

### 7.5 可调开关（都带默认值，仅供复测）

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `EDA_NO_ADDRESS_FIELD_WRITE` | `1` | 本族总开关 |
| `EDA_SLOT_LATTICE` | `8` | 每个 slot 提议多少个值 |
| `EDA_SLOT_POINTERS` | `4` | 指针扫描几个候选地址 |
| `EDA_SLOT_POINTER_ALIGN` | `64` | 指针扫描粒度 |
| `EDA_SLOT_POINTER_WINDOW` | `1024` | 指针扫描窗口 |
| `EDA_SLOT_PRIORITY` | `8` | 候选优先级（6~12 实测同值，不敏感） |
| `EDA_SLOT_PAIR_DEPTH` | `2` | 关联槽成对扫描的深度 |

### 7.6 值格的取法（泛化要点）

**不许写死阈值。** 设计拒绝一个对象的原因（低于下限 / 高于上限 / 未对齐 / 某位被置）
是设计私有的，规格不会说。做法是走**边界格**：0、1、声明区间的两端与各端外一格、
每个 2 的幂及其邻居 —— 这样"任何形式的阈值"都被夹住，而不点名任何一个。
截断用「保留小值 + 余下按序几何铺开」，避免线性截断把预算全花在 0x100 以下。

---

## 8. `pointer_walk` 族（2026-09-27 落地，52 → 85）

### 8.1 触发条件（`_pointer_walk_fields()`，全部满足）

| 条件 | 理由 |
|---|---|
| `write_enable` 非空 | 有写入口 |
| `register_address` 为空 | 否则已被寻址族覆盖 |
| `instance_select` 为空 | 同上 |
| `request` 角色 ≥ 2 个 | 走查需要"一个开启遍历 + 一个推进"两条线 |
| 地址 lane 非空 | 遍历指针要有去处 |

实测：**8 个 DUT 只有 `dma_desc_engine_validation` 满足**，其余 7 个生成 **0 条候选** ——
零回归是结构性的，不是巧合。

### 8.2 形态

```
reset ×2 → {arm=1, ptr=base} → {ptr=base}
  ↓ 每个描述符
  {advance=1, ptr=addr} ×2          # 请求下一个元素，持两拍（设计可能请求拍锁存、下一拍动作）
  [可选] 中途 stop（abort）：紧跟在 advance 之后 → 见 §8.3
  [可选] 编辑：{advance=1, idx=slot, strobe=1, data=value} → {advance=1}  # 存一拍再 fetch 一拍
  settle 4 拍                       # 让设计跑完 validate/issue/handshake
  [可选] ack 拍                     # 服务完成握手；withhold 时不给 → 触发重试耗尽
```

### 8.3 三处关键纠正（都是逐周期复盘抓出来的）

1. **中途 stop 必须紧跟 advance，不能在 settle 之后**。放在 settle 后，请求到达时设计
   已经回 IDLE，abort 永不发生（实测 `fsm_state.abort` / `seq_abort_midburst` 全缺）。
2. **编辑与 fetch 不能同拍**。`_fetch` 里 `word_we` 那一拍把 `result` 设成 8
   （word_write）就**直接 return，不进 VALIDATE**。必须**再 fetch 一拍**（`word_we=0`）
   才走校验。少了这一拍，`fail_class` 永远是 0。
3. **基址不能用 0**。`_step_idle` 把 `desc_ptr==0` 解析成设计自己的 `desc_base`（隐藏参数），
   而走查指针从 `base + k*step` 算 —— base=0 时 arm 落在 `desc_base`、走查从 0 开始，
   全 miss。所以基址只取非零值，且 arm 与走查用**同一个** base。
   基址扫描取 `align, align*4, align*16, align*64` 里前 N 个（`align`=64 → 64/256/1024），
   覆盖"每页/每块一个区域"的常见布局；线性扫描会为了够到 256 白白浪费窗口。

### 8.4 每描述符一个失败类 → 每个非法值一条候选

`_validate` **首个失败即返回**，且被拒的元素把设计留在 ERROR；退出 ERROR 的手段
（`clear`）本接口没有描述 → 错误之后的一切不可达。

所以：
- 「一个 meaning 的多个非法值放在同一条走查里」**做不到**（第二个描述符已在 ERROR）；
- 每个 `(meaning, value)` 需要**自己的短程序**（`_single_edit_candidate`）：
  `arm → {advance, edit, value} → {advance} ×2 → settle`，只作用于第一个元素。
  这类程序与 step 无关（它不推进），所以只用 1 个 base/step 对，候选数只与值格成正比。

| meaning | 非法值格 | 命中的 fail_class |
|---|---|---|
| `length` | `0`, `0x2000` | 7 ZERO_LEN / 8 OVER_LEN |
| `mode` | `2`, `0xFFFFFFFF` | 10 UNSUPPORTED_MODE / 11 UNSUPPORTED_FLAG |
| `src`/`dst`/`addr` | `0`, `0x10001`, `0xFFFF0` | 1/4 ZERO、9 UNALIGNED、6 HIGH_DST |
| `flag` | `0xFFFFFFFF` | 11 UNSUPPORTED_FLAG |

**注意 `mode` 不能只写 `0xFFFFFFFF`**：`mode = value & 3 = 3`（合法），而 `bit31=1`
会让**先执行的** flag 检查命中 → 永远拿不到 `UNSUPPORTED_MODE`。必须补一个 `mode & 3 == 2`
的纯 mode 非法值。

**注意 `HIGH_DST` 要对齐地址**：`dst = 0x10001` 非 4 对齐 → 被更早的对齐检查截胡（报 UNALIGNED）。
要命中 `HIGH_DST` 得用 `0xFFFF0`（对齐）再让 `dst + len >= 0x10000`。

### 8.5 候选预算是硬约束（本项目反复踩）

| 版本 | 候选数 | 覆盖率 |
|---|---|---|
| 初版（笛卡尔展开） | ~1000 | **51**（还不如不做） |
| 收敛（每 shape 一条，内部轮转） | 120 | 71 → 75 → 80 |
| + 单元素编辑族 | 130 | **85** |

**生成候选与结构化路径抢同一份周期预算**：候选膨胀会饿死别的候选族。所以
"每 shape 一条、内部按描述符轮转" 是必须的写法，不能每条 `(shape × slot × value × step)` 都生成。

### 8.6 剩余 2 个 bin：规格的信息缺口，不是能力缺口

`seq_error_recover` 与 `step_result.err_clear` 的唯一触发点是 `clear`（`result=7`、
`seq6=True`），而规格里该字段只写作 `pad0`，**从未描述用途**。

通用路径不该靠"猜 `pad0` 其实是 clear"来拿这 2 个 —— 那是使用隐藏知识，属 overfit。
故**就此收手在 85/87**。若将来要收这类分，正确的做法是让 DUT 规格把该字段写清楚，
而不是在推理路径里加家族特判。

### 8.7 本族开关

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `EDA_POINTER_WALK` | `1` | 本族总开关（关 = 逐位回到 52/87） |
| `EDA_POINTER_WALK_STEPS` | `0x4,0x8,0x10,0x20,0x40` | 描述符间地址步长扫描 |
| `EDA_POINTER_WALK_DEPTH` | `6` | 每条走查推进几个元素 |
| `EDA_POINTER_WALK_BASES` | `3` | 基址候选数 |
| `EDA_POINTER_WALK_PRIORITY` | `8` | 候选优先级 |

### 8.8 零回归证据

- 逐 bin 对比（`results/family_removal_equivalence_5k.json`）：**7 个已有 DUT 与改动前基线
  完全一致（0 个 bin 变化）**；`dma_desc_engine` `newly_missing=0`、`newly_covered=51`。
- 结构性证据：`_pointer_walk_fields()` 对那 7 个 DUT 全部返回 `None` → 生成 0 条候选。
- 开关对照：`EDA_POINTER_WALK=0` → 52/87，`=1` → 85/87。
- 泛化尺子：`check_parsing_generalization.py` PASS（角色 87/87、候选 26/26）；
  `check_name_invariance.py` 最差落差仍 2 bin（未退化）。
- 契约测试 4 个全过。
