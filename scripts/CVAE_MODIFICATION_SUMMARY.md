# CVAE/MPPI 改动总结（当前实施版）

本文档总结当前已经落地的改动、与我们讨论目标的对齐情况，以及仍待补充的项。

## 1. 已落地改动

### 1.1 采集策略与配置
- 已新增 CVAE 数据采集相关开关：
  - `cvae_require_context`
  - `cvae_collect_all_iterations`
  - `cvae_collect_iteration_stride`
  - `cvae_bucket_ratio_high/mid/low`
  - `cvae_labeled_ratio_min`
- 这些开关已暴露到 `get_planner_cfg(...)->to_cfg`，可在 planner 侧直接配置。

### 1.2 trajectory optimizer 采集逻辑
- 采集不再限定“仅最后一轮”，支持多轮采集（按 stride）。
- 采样策略不再纯 top-k，改为 high/mid/low 分层抽样。
- 支持 context 约束：`cvae_require_context=True` 时，无 context 不落盘。
- 数据集 payload 扩展字段：
  - `goal_state_3way`
  - `outcome_success`
  - `outcome_risk_max`
  - `outcome_risk_sum`
  - `success_supervision_mask`
  - `sample_weight`
- 加入 `labeled_ratio` 约束过滤逻辑（当有标注样本存在时，限制未标注样本比例）。

### 1.3 采集 CLI
- `scripts/plan_test.py` 新增并透传参数：
  - `--cvae_collect_all_iterations`
  - `--cvae_collect_iteration_stride`
  - `--cvae_labeled_ratio_min`

### 1.4 训练脚本稳定性修复
- 修复 `train_cvae_sampler.py` 在 dataset 含 3 个以上 tensor 时的样本解包问题（避免 `sample_cond, _ = dataset[0]` 崩溃）。

---

## 2. 与讨论目标对齐情况

### 已对齐
- 保留并强化 context 条件输入（避免盲走）。
- 采集从 final-topk 扩展为多轮分层。
- 增加三态/长程字段所需的数据容器和落盘入口。
- 增加有标注占比控制参数。

### 仍待补充（下一步）
1. **执行轨迹硬标签回填**：
   - 需要在 `planner.py` 的 `done/goal_reached` 处，把 executed 轨迹写成 `goal_state_3way in {1,2}`，并将 `success_supervision_mask=True`。
2. **训练脚本监督头扩展**：
   - 当前训练脚本仍以重建+KL为主，尚未引入显式 success/risk 监督分支。
3. **文档统一**：
   - 需要把 `CVAE_DATA_SCHEMA_V1.md` 的“执行版最小字段协议”与当前代码字段完全对齐。

---

## 3. 建议上线顺序

1. 先跑采集并检查数据字段/shape是否正确。
2. 增加 executed 轨迹硬标签回填。
3. 训练脚本接入 `success_supervision_mask` 与 risk 监督。
4. 线上只看 `goal_reach_rate` 主指标做 A/B。
