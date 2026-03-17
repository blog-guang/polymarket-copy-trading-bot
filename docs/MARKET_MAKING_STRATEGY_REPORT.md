# Polymarket 做市策略算法报告

> **理论基础：** *Toward Black-Scholes for Prediction Markets* (Shaw Dalen, arxiv 2510.15205)
> **参考来源：** Polymarket 官方文档、poly-maker 开源实现、Phemex 市场分析
> **生成时间：** 2026-03-17 | 分支：claude/market-making-strategy-report-BHFYX

---

## 一、论文核心理论

### 1.1 核心思想

传统预测市场缺乏系统性的期权定价框架。论文提出将 **Black-Scholes** 体系迁移到预测市场，构建统一的做市定价模型。

**关键洞察：** 预测市场价格在 **logit（对数几率）空间**中表现出与金融市场类似的随机性，因此可以套用相同的随机过程建模工具。

### 1.2 Logit 跳跃-扩散随机过程

论文提出的核心随机过程：

```
对数几率变换：
  x(t) = log[p(t) / (1 - p(t))]   ← 将概率映射到 (-∞, +∞)

随机过程（混合模型）：
  dx = σ dW + J dN(λ)

其中：
  σ dW  = 连续扩散项（信念的渐进演化）
  J dN(λ) = 跳跃项（离散信息冲击，如新闻事件）
  σ = 信念波动率（belief volatility）
  λ = 跳跃强度（jumps/day）
  J = 跳跃幅度（服从正态分布 N(μ_j, σ_j²)）
```

**直觉解释：**
- 没有新信息时，市场概率在当前位置附近随机漂移（扩散）
- 当重大新闻发布时，概率发生离散跳跃（跳跃）
- 做市商的核心任务：根据 σ 和 λ 动态调整报价价差

### 1.3 日历效应（Calendar Effect）

论文特别指出预测市场的独特日历效应：

```
临近结算时，价格必然收敛至 0 或 1（确定性）

时间因子：τ = 剩余天数 / 市场总存续天数 ∈ [0, 1]

效应：
  - τ → 1（市场初期）：不确定性高，价差自然宽
  - τ → 0（临近结算）：虽然不确定性已解析，但价格可能剧烈波动
                         → 价差反而应该扩大（做市商风险补偿）

日历乘数 = 1 + (1 - τ)³ × calendar_factor
```

### 1.4 EM 算法参数估计

论文使用 **期望最大化（EM）算法**分离扩散和跳跃成分，过滤微观结构噪声：

```
输入：价格时间序列 p[0..n]

Step 1: 转换到 logit 空间
  x[i] = log(p[i] / (1 - p[i]))

Step 2: 计算对数收益
  r[i] = x[i] - x[i-1]  (时间间隔 dt[i])

Step 3: EM 迭代
  E-step: 对每个 r[i] 计算跳跃后验概率
    P(jump | r[i]) = π × N(r[i]; μ_j, σ_j²) / 总密度
    P(no jump | r[i]) = (1-π) × N(r[i]; 0, σ²·dt[i]) / 总密度

  M-step: 更新参数
    π     ← 平均跳跃概率
    σ     ← 非跳跃收益的扩散波动率
    μ_j   ← 跳跃均值
    σ_j   ← 跳跃方差
    λ = π / mean(dt)

Step 4: 公平价值估计
  x_fair = EMA(logit(prices), α=0.3)
  p_fair = logistic(x_fair)
```

---

## 二、做市定价算法

### 2.1 综合定价公式

融合 Stoikov 库存模型 + 论文三因子风险框架：

```
报价计算步骤：

1. 归一化库存
   q = net_position_USD / max_inventory  ∈ [-1, +1]

2. 动态价差乘数（论文三因子）
   spread_mult = 1 + γ·q² + β·σ² + ζ·λ
     γ = 库存风险厌恶系数（默认 0.2）
     β = 信念波动率敏感度（默认 1.0）
     ζ = 跳跃强度敏感度（默认 0.5）

3. 日历乘数
   τ = days_to_resolution / total_duration
   calendar_mult = 1 + (1-τ)³ × 2.0

4. 半价差
   half_spread = (base_spread/2) × spread_mult × calendar_mult
   half_spread = min(half_spread, max_spread/2)

5. 库存偏斜中间价（Stoikov）
   adjusted_mid = fair_value - γ · σ² · q
   （持有过多 YES → 压低中间价 → 促进卖出）

6. 最终报价
   bid = clamp(adjusted_mid - half_spread, 0.01, 0.99)
   ask = clamp(adjusted_mid + half_spread, 0.01, 0.99)
```

### 2.2 数值示例

假设：p_fair=0.50, σ=0.04, λ=0.5, q=0.3（轻度超买）

```
spread_mult = 1 + 0.2×0.09 + 1.0×0.0016 + 0.5×0.5
            = 1 + 0.018 + 0.0016 + 0.25
            = 1.270

τ=0.8（市场还有 80% 剩余时间）
calendar_mult = 1 + (0.2)³ × 2.0 = 1.016

half_spread = (0.02/2) × 1.270 × 1.016 = 0.01288

adjusted_mid = 0.50 - 0.2 × 0.0016 × 0.3 = 0.4999

bid = 0.4999 - 0.0129 = 0.487
ask = 0.4999 + 0.0129 = 0.513
价差 = 0.026（2.6分）
```

### 2.3 市场评分公式

```
风险调整收益评分：
  score = daily_reward_estimate / (σ × max(λ, 0.01))

daily_reward_estimate = reward_pool × market_share
  market_share = 2 × capital × proximity / (liquidity + 2 × capital × proximity)
  proximity = max(0, 1 - half_spread / 0.05)

按 score 降序选择前 MM_MARKET_LIMIT 个市场
```

---

## 三、两层市场筛选框架

### 第一层：量化筛选

| 条件 | 阈值 | 说明 |
|------|------|------|
| 价格区间 | [0.10, 0.90] | 排除近确定性市场 |
| 距结算天数 | > 7 天 | 避免临近结算风险 |
| 市场存续 | > 14 天 | 有足够价格历史用于EM |
| 奖励池 | > 0 | 有流动性激励才有正期望 |

### 第二层：质量护盾（人工/规则排除）

- 政治/法律风险高的市场（结果可能存在争议）
- 规则定义模糊的市场
- 内幕交易迹象（巨鲸主导、单边异常流量）
- 选举类市场（2024年后奖励锐减）

---

## 四、库存管理与风险控制

### 4.1 库存限制与偏斜

```
单市场限制:
  最大净头寸 = MM_MAX_INVENTORY_PER_MARKET（默认 $500）
  总敞口上限 = MM_MAX_TOTAL_INVENTORY（默认 $3,000）

订单大小缩减（基于归一化库存 q）：
  bid_scale = max(0.25, 1 - max(0, q))    ← 偏多时减少买单
  ask_scale = max(0.25, 1 - max(0, -q))   ← 偏空时减少卖单
  bid_size = base_order_size × bid_scale
  ask_size = base_order_size × ask_scale
```

### 4.2 熔断机制（Circuit Breaker）

| 触发条件 | 响应 |
|---------|------|
| 60 秒内价格变动 > 5% | 撤单 + 暂停 5 分钟 |
| 当日该市场亏损 > $200 | 撤单 + 暂停 24 小时 |
| 距结算 < 1 天 | 撤单，不再参与 |

### 4.3 临近结算处理

| 剩余时间 | 策略 |
|---------|------|
| > 30 天 | 正常做市 |
| 15-30 天 | 价差 × 1.5 倍（calendar_mult 自动处理） |
| 7-15 天 | 价差 × 2 倍，降低库存上限 50% |
| < 7 天 | 停止新开仓 |
| < 1 天 | 全部撤单 |

---

## 五、流动性奖励优化

Polymarket 奖励采用二次方评分（Quadratic Scoring）：

```
奖励公式：
  daily_reward ≈ pool × (your_score / total_score)

your_score = Σ [size_i × proximity_i × time_fraction_i]

proximity = max(0, 1 - |order_price - mid| / 0.05)
  (距中间价超过 5% 则不得分)

关键结论：
  ✓ 双边挂单 ≈ 单边挂单奖励的 3 倍
  ✓ 离中间价每增加 1%，奖励降约 20%
  ✓ postOnly 保证纯 maker 身份，避免 taker 手续费
```

---

## 六、系统架构与代码结构

### 6.1 新增文件

```
src/
├── models/
│   └── marketMakingState.ts     # MongoDB 数据模型（价格历史/订单/库存/市场）
├── utils/
│   ├── jumpDiffusionModel.ts    # 论文核心算法（EM 参数估计）
│   └── marketMakingPricer.ts    # 报价定价（Stoikov + 三因子）
└── services/
    ├── marketMakingMonitor.ts   # 市场扫描/评分（每 5 分钟）
    └── marketMakingExecutor.ts  # 报价更新主循环（每 30 秒）
```

### 6.2 修改文件

| 文件 | 改动 |
|------|------|
| `src/config/env.ts` | 新增 `BOT_MODE` + `MarketMakingConfig` 解析 |
| `src/index.ts` | 根据 `BOT_MODE` 启动不同服务 |
| `.env.example` | 补充全部 MM 配置变量文档 |

### 6.3 数据流

```
[Polymarket Data API]          [CLOB API]
  markets list                  order book prices
       │                              │
       ▼                              ▼
┌─────────────────────────────────────────────────┐
│           marketMakingMonitor（每5分钟）          │
│  - 拉取市场列表                                   │
│  - 双层筛选                                       │
│  - 拉取价格历史 → 存 mm_price_history             │
│  - EM估计 σ, λ                                   │
│  - 计算 risk_adjusted_score                       │
│  - 更新 mm_markets（top N active=true）           │
└─────────────────────────────────────────────────┘
                          │ MongoDB
                          ▼
┌─────────────────────────────────────────────────┐
│         marketMakingExecutor（每30秒）            │
│  - 读 mm_markets（active=true, sort by score）   │
│  - 对每个市场：                                   │
│    ① 拉取实时 mid price                          │
│    ② EM 估计 → computeQuotes()                  │
│    ③ 熔断检查                                    │
│    ④ shouldReprice() → 是否需要更新               │
│    ⑤ cancelMarketOrders() + 2x placeGTCOrder()   │
│  - 定期 reconcileFilledOrders()更新库存           │
└─────────────────────────────────────────────────┘
                          │
                          ▼
              [Polymarket CLOB - GTC 限价单]
```

---

## 七、三种运行模式

通过 `BOT_MODE` 环境变量控制：

### COPY（默认）
保持原有 Copy Trading 功能，不启动做市模块。

### MARKET_MAKING
仅运行做市引擎，适合专注流动性提供的场景：
- 自动扫描市场、挂双边 GTC 限价单
- 赚取买卖价差 + 流动性奖励

### HYBRID（推荐进阶用户）
同时运行两个引擎，实现**策略协同**：

```
协同效应（信号融合）：
  Copy Trading → 识别聪明钱方向
  Market Making → 在该方向的市场双边挂单

具体机制：
  - 聪明钱大量买 YES → adjusted_mid 上移 → ask 更高（高价卖出）
  - 做市在已跟单的市场同时挂单 → 额外获取流动性奖励
  - 降低总持仓成本（用奖励收益补贴 Copy Trading 滑点）
```

---

## 八、预期绩效与风险提示

### 收益预期（$10,000 本金，20 个市场）

| 来源 | 日均估算 |
|------|---------|
| 价差收益（bid-ask capture） | $50 ~ $150 |
| 流动性奖励（USDC rewards） | $100 ~ $300 |
| **合计** | **$150 ~ $450 / 天** |

> 数据基于奖励削减后（2025年后）的保守估计

### 主要风险

| 风险 | 缓解措施 |
|------|---------|
| 信息不对称（逆向选择）| postOnly 模式 + 跳跃检测熔断 |
| 库存积累 | 双边缩减订单 + 最大净头寸限制 |
| 市场突变 | 价格变动 >5% 触发熔断 |
| 结算风险 | 7 天前停止新开仓 |
| 奖励削减 | score 排序自动过滤低奖励市场 |

### 重要提示

> ⚠️ **2024 年大选后 Polymarket 大幅削减流动性奖励**
> 当前单纯依赖奖励的策略盈利能力已显著降低。
> HYBRID 模式（结合 Copy Trading）可通过多元化收益来源提高整体盈利能力。
> 建议先用少量资金（每市场 $10~$20）验证 GTC 挂单流程后再加大规模。

---

## 九、快速启动

### 做市模式

```bash
# 在 .env 中设置
BOT_MODE=MARKET_MAKING
MM_BASE_CAPITAL=1000          # 从小资金开始
MM_ORDER_SIZE_USD=10           # 每笔挂单 $10
MM_MAX_INVENTORY_PER_MARKET=100
MM_POST_ONLY=true              # 只做 maker

npm run dev
```

### 混合模式

```bash
# 在 .env 中保持原有 copy trading 配置，并添加：
BOT_MODE=HYBRID
MM_ORDER_SIZE_USD=50
MM_MARKET_LIMIT=10

npm run dev
```

---

## 十、参考资料

- [arxiv 2510.15205 – Toward Black-Scholes for Prediction Markets](https://arxiv.org/pdf/2510.15205)
- [Polymarket 官方：Automated Market Making on Polymarket](https://news.polymarket.com/p/automated-market-making-on-polymarket)
- [Polymarket CLOB L2 Methods 文档](https://docs.polymarket.com/developers/CLOB/clients/methods-l2)
- [Phemex：Polymarket's Stable Market Making Strategy](https://phemex.com/news/article/polymarkets-strategy-for-stable-market-making-43240)
- [GitHub：warproxxx/poly-maker](https://github.com/warproxxx/poly-maker)
- [NYC Servers：Market Making on Prediction Markets Complete 2026 Guide](https://newyorkcityservers.com/blog/prediction-market-making-guide)

---

*生成时间：2026-03-17 | 分支：claude/market-making-strategy-report-BHFYX*
