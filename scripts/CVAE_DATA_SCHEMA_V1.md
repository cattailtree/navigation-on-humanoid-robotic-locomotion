# CVAE 数据 Schema v1（最小可行版）

> 目标：让 CVAE 同时具备 **分支覆盖能力**（左/右/中等主要路径都能采到）和 **长程成功偏置**（不只看短程离目标近，而是看最终能否通过）。

## 1. 存储单位与格式

- 文件格式：`torch.save(dict, path)`（`.pt`）
- 一条样本 = 某个 `env` 在某个 `plan_step` 的一条候选轨迹。
- 统一 dtype：
  - 连续值：`float32`
  - 布尔标签：`bool`
  - id / 类别：`int64`

---

## 2. 字段定义（v1）

## 2.1 必备字段（Required）

1. `mean_actions`: `(N, H, A)`
   - MPPI 当前均值动作轨迹（CVAE condition 主体）。

2. `target_actions`: `(N, H, A)`
   - 作为监督目标的候选轨迹（来自分支化筛选，不再仅仅是全局 top-k）。

3. `context`: `(N, C)`
   - 固定长度上下文向量，建议融合：goal 相对位姿 / proprio / history / perception embedding。

4. `sample_weight`: `(N,)`
   - 每条样本训练权重（用于 hard case 与稀有分支重加权）。

## 2.2 长程结果标签（Outcome Labels）

5. `outcome_success`: `(N,)` bool
   - 该样本对应 rollout 是否最终成功到达目标。

6. `outcome_terminal_type`: `(N,)` int64
   - 建议编码：
     - `0`: success
     - `1`: collision
     - `2`: timeout
     - `3`: unstable/fall
     - `4`: other

7. `outcome_final_goal_dist`: `(N,)` float32
   - 终止时目标距离（用于区分“接近但失败”）。

8. `outcome_progress`: `(N,)` float32
   - 长程进展指标（如 `start_dist - final_dist`）。

9. `outcome_risk_max`: `(N,)` float32
   - 轨迹沿途最大风险值。

10. `outcome_risk_sum`: `(N,)` float32
    - 轨迹沿途累计风险值。

## 2.3 分支覆盖标签（Branch Labels）

11. `branch_id`: `(N,)` int64
    - 分支类别 id。v1 建议：
      - `0`: 左绕
      - `1`: 右绕
      - `2`: 中路/直穿
      - `3`: 其他/不确定

12. `branch_score_rank`: `(N,)` int64
    - 样本在所属分支桶内的分数排序（越小越优）。

## 2.4 可追溯元数据（Provenance）

13. `episode_id`: `(N,)` int64
14. `env_id`: `(N,)` int64
15. `plan_step`: `(N,)` int64
16. `collection_round`: `(N,)` int64
17. `timestamp_ms`: `(N,)` int64

18. `schema_version`: `int`
    - v1 固定写 `1`。

---

## 3. 采样规则（v1）

> 核心：不再只留“全局 top-k 最优”，而要“分支覆盖 + 结果平衡”。

1. **分桶（Branch Partition）**
   - 先把 population 按路径形态分桶（左/右/中等）。
   - 可用 v1 简化规则：
     - 前 20~30% 轨迹横向位移累计符号（左负右正）
     - 首次显著偏航方向

2. **桶内筛选（Within-Branch Keep）**
   - 每个桶保留 `k_branch` 条高分轨迹（按 planner value/cost）。

3. **结果平衡（Outcome Balancing）**
   - 保留成功样本，也保留关键 hard negative：
     - 短程看起来好，但长程失败

4. **权重分配（sample_weight）**
   - 基础值：`1.0`
   - 上调权重：
     - “先远离目标但最终成功”的样本
     - 稀有分支样本
     - hard negative 样本

---

## 4. 划分规则（Train/Val/Test）

- 按 `episode_id` 切分，禁止按行随机打散（防止时间泄漏）。
- 建议初始比例：
  - train: 80%
  - val: 10%
  - test: 10%
- 每个 split 都输出：
  - 分支分布直方图
  - 成功率

---

## 5. 统计看板指标（v1）

## 5.1 数据质量看板

1. `branch_coverage_rate`
   - 场景中至少出现两个分支的比例。

2. `branch_balance_entropy`
   - 分支分布熵（越高表示覆盖越均衡）。

3. `success_ratio_overall`
4. `success_ratio_by_branch`
5. `hard_negative_ratio`

## 5.2 离线模型看板

6. `recon_mse`
7. `kl_loss`
8. `success_weighted_loss`
9. `branch_recall@K`
10. `success_rate@K`
11. `detour_success_capture`
   - 对“短程变差但最终成功”样本的召回。

## 5.3 在线规划看板

12. `goal_reach_rate`
13. `collision_rate`
14. `timeout_rate`
15. `median_time_to_goal`
16. `planner_value_gap`

---

## 6. 进入代码改造前的验收门槛

满足以下条件再进入“采集/训练/部署”代码改造：

1. 数据集字段完整（覆盖 v1 必备字段）。
2. `branch_coverage_rate` 达标（建议起始阈值 `>= 0.7`）。
3. 每个分支样本量满足最小下限。
4. `hard_negative_ratio` 非零且稳定。
5. 相比高斯基线，`branch_recall@K` 与 `success_rate@K` 同时提升。

---

## 7. 与旧数据兼容策略

旧数据若仅包含 `mean_actions/target_actions/context`：
- 标记为 `schema_version = 0`
- 可用于 warm-up，但不作为最终长程目标训练主数据源。
