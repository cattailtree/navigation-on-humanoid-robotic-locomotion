# CVAE-MPPI 改造实施清单（执行版）

> 范围：把当前“仅 final top-k imitation”的采集/训练流程，改为以 `goal_reach_rate` 为第一目标的最小可行版本。  
> 原则：不增加昂贵反事实 rollout；保留现有 MPPI 主流程；新增字段和采样策略尽量小改。

---

## A. 目标与约束（冻结）

1. 主线上目标：`goal_reach_rate`（唯一主KPI）。
2. 数据标签采用三态：
   - `2 = reached`
   - `1 = done_fail`
   - `0 = not_taken`
3. 长程标签仅保留：
   - `outcome_success`
   - `outcome_risk_max`
   - `outcome_risk_sum`
4. 不做额外反事实执行（不新增 rollout 成本）。
5. 约束有标注样本总占比：
   - `labeled_ratio = (reached + done_fail) / total >= R_min`
   - 推荐 `R_min` 初始 `0.60`（可在 `0.55~0.65` 调整）

---

## B. 文件级改造清单

## B1) `scripts/CVAE_DATA_SCHEMA_V1.md`

**目标**：把文档升级为“v1.1-minimal（执行版）”。

### 修改项
- [ ] 增加“v1.1-minimal”段落，明确最小字段集合：
  - `mean_actions`
  - `target_actions`
  - `context`（可选）
  - `goal_state_3way`
  - `outcome_success`
  - `outcome_risk_max`
  - `outcome_risk_sum`
  - `success_supervision_mask`
  - `sample_weight`
- [ ] 明确：branch/provenance 为后续增强项，当前阶段非必需。
- [ ] 明确：`goal_reach_rate` 为线上唯一主指标；MPT/MPL 仅次级参考。

### 验收
- [ ] 文档中字段与训练脚本实际读取字段一致。

---

## B2) `exts/fdm/fdm/planner/sampling_planner/trajectory_optimizer_cfg.py`

**目标**：新增采集策略开关与配额配置。

### 新增配置字段（建议）
- [ ] `cvae_collect_all_iterations: bool = True`
- [ ] `cvae_collect_iteration_stride: int = 1`
- [ ] `cvae_bucket_ratio_high: float = 0.4`
- [ ] `cvae_bucket_ratio_mid: float = 0.3`
- [ ] `cvae_bucket_ratio_low: float = 0.3`
- [ ] `cvae_labeled_ratio_min: float = 0.60`
- [ ] `cvae_use_threeway_goal_state: bool = True`

### 验收
- [ ] 默认值不影响旧流程（当新字段关闭时行为与原版兼容）。

---

## B3) `exts/fdm/fdm/planner/sampling_planner/planner_cfg.py`

**目标**：把上面新增配置暴露到 `get_planner_cfg(...)` 返回字典。

### 修改项
- [ ] 在 `to_cfg` 中加入 `cvae_collect_*`、`cvae_labeled_ratio_min`、`cvae_use_threeway_goal_state`。
- [ ] 保持原有字段兼容（`cvae_dataset_dump_path/topk/max_samples` 不删除）。

### 验收
- [ ] 通过现有 `get_planner_cfg` 路径可配置新字段。

---

## B4) `exts/fdm/fdm/planner/sampling_planner/trajectory_optimizer.py`

**目标**：重构采集回调，不再仅 final top-k；扩展落盘标签字段。

### 关键改造
1. **采样时机**
- [ ] 替换 `if iteration != last: return` 逻辑。
- [ ] 支持多轮采样（按 stride）。

2. **分层采样**
- [ ] 对 `values` 做 high/mid/low 分层抽样，而不是只 `torch.topk`。
- [ ] 每层配额由 cfg 比例控制。

3. **风险标签**
- [ ] 对入库轨迹计算/记录：
  - `outcome_risk_max`
  - `outcome_risk_sum`
- [ ] 若当前路径可直接访问 collision trajectory，则直接统计；否则记录 proxy 并标明来源。

4. **三态标签（候选侧）**
- [ ] 默认候选样本写：`goal_state_3way = 0`（not_taken）。
- [ ] 默认 `success_supervision_mask = False`。

5. **payload 扩展**
- [ ] 在 `_flush_cvae_dataset()` 的 payload 中加入：
  - `goal_state_3way`
  - `outcome_success`（候选默认可置占位，等待执行回填）
  - `outcome_risk_max`
  - `outcome_risk_sum`
  - `success_supervision_mask`
  - `sample_weight`

6. **标注占比约束**
- [ ] 在写盘前检查 `labeled_ratio`。
- [ ] 若低于 `cvae_labeled_ratio_min`，削减 not_taken 入库数量（优先）以满足占比底线。

### 验收
- [ ] 采集文件包含新增字段且 shape 对齐。
- [ ] 关闭新开关时与旧版字段兼容。

---

## B5) `exts/fdm/fdm/planner/planner.py`

**目标**：把执行轨迹的真实结果回填到数据集标签（reached/done_fail）。

### 修改项
- [ ] 在 `dones/goal_reached` 更新处生成 executed 轨迹标签：
  - `goal_reached -> goal_state_3way=2`
  - `done_fail -> goal_state_3way=1`
- [ ] 同步回填：
  - `outcome_success = (goal_state_3way == 2)`
  - `success_supervision_mask = True`
- [ ] 确保 done 与 goal_reached 重叠时采用互斥口径（优先 reached 或按约定顺序）。

### 验收
- [ ] executed 样本全部具备硬标签；not_taken 不被误标。

---

## B6) `scripts/train_cvae_sampler.py`

**目标**：支持三态+部分监督训练。

### 修改项
- [ ] 读取新增字段：
  - `goal_state_3way`
  - `outcome_success`
  - `outcome_risk_max`
  - `outcome_risk_sum`
  - `success_supervision_mask`
  - `sample_weight`
- [ ] 损失拆分：
  - `L_recon`（现有）
  - `L_kl`（现有）
  - `L_success`（仅 `success_supervision_mask=True` 样本）
  - `L_risk`（基于 risk_max/sum 的弱监督）
- [ ] 支持配置权重：`lambda_success`, `lambda_risk`。

### 验收
- [ ] 当数据缺少新字段时给出清晰报错或降级提示。
- [ ] success loss 确认未使用 not_taken 样本。

---

## B7) `scripts/README_CVAE_PIPELINE.md`

**目标**：更新流程文档与运行步骤。

### 修改项
- [ ] 增加“v1.1-minimal 标签与采样策略”章节。
- [ ] 说明三态标签语义与 `labeled_ratio` 约束。
- [ ] 增加训练参数示例（含 `lambda_success/lambda_risk`）。

### 验收
- [ ] 文档命令与当前代码参数一致。

---

## B8) `scripts/plan_test.py`

**目标**：暴露必要开关，便于在线采集试验。

### 修改项
- [ ] 新增 CLI 参数（可选）：
  - `--cvae_collect_all_iterations`
  - `--cvae_collect_iteration_stride`
  - `--cvae_labeled_ratio_min`
- [ ] 写入 `sampling_planner_cfg_dict["to_cfg"]`。

### 验收
- [ ] 不带新参数时行为与现有脚本一致。

---

## C. 训练与评估协议

## C1) 训练采样协议
- [ ] 保证 `labeled_ratio >= R_min`。
- [ ] 不强制 reached/done 内部比例（按场景自然分布）。
- [ ] not_taken 样本仅用于重建+风险，禁入 success 硬监督。

## C2) 指标协议
- [ ] 主指标：`goal_reach_rate`。
- [ ] 辅助安全指标：collision-related rate。
- [ ] MPT/MPL 仅作为同成功率条件下的 tie-break 观察，不作为主优化目标。

---

## D. 分阶段上线顺序（建议）

1. **阶段1（数据）**：先完成采集字段扩展 + 多轮分层采样。
2. **阶段2（训练）**：接入三态/掩码损失，跑离线训练稳定性。
3. **阶段3（在线）**：只对比 `goal_reach_rate`，逐步放量。
4. **阶段4（增强）**：再考虑分支覆盖、溯源等扩展项。

---

## E. 回滚与兼容

- [ ] 保留旧字段兼容：仅有 `mean_actions/target_actions/context` 时可 warm-up。
- [ ] 新字段缺失时训练脚本提供降级路径（禁用 success/risk 分支）或直接报错并提示。
- [ ] 所有新逻辑加 cfg 开关，确保可一键回滚到旧采集策略。

---

## F. 最小验收清单（上线前）

- [ ] 数据文件含 v1.1-minimal 字段，shape/dtype 正确。
- [ ] `labeled_ratio` 连续多轮满足阈值。
- [ ] success loss 仅在 executed 样本生效。
- [ ] 在线 `goal_reach_rate` 相比旧版不下降（至少持平）。
