# Polymarket 做市策略算法报告

> 本报告基于 @mrryanchi 分享的做市策略线索，结合 Polymarket 官方文档、开源实现及市场分析，整理而成。
>
> **数据来源：**
> - [Polymarket 官方做市策略文章](https://news.polymarket.com/p/automated-market-making-on-polymarket)
> - [Phemex：Polymarket 稳定做市策略](https://phemex.com/news/article/polymarkets-strategy-for-stable-market-making-43240)
> - [预测市场做市完全指南 2026](https://newyorkcityservers.com/blog/prediction-market-making-guide)
> - [开源实现 poly-maker](https://github.com/warproxxx/poly-maker)

---

## 一、策略核心思想

Polymarket 做市策略的本质是**低波动 + 高奖励市场的流动性套利**，核心盈利来源有两块：

| 盈利来源 | 说明 |
|---------|------|
| **买卖价差（Spread）** | 同时挂 YES 买单和 YES 卖单，赚取中间差价 |
| **流动性奖励（Liquidity Rewards）** | Polymarket 每日向双边挂单者发放 USDC 奖励 |

**关键洞察：** 双边挂单相比单边挂单可获得 **近 3 倍** 的流动性奖励，且挂单越靠近当前价格，奖励越高。这使得做市的预期收益可以远超单纯的方向性博弈。

---

## 二、双层市场筛选框架

策略采用"机器理性 + 认知护盾"两层过滤机制：

### 第一层：机器理性（量化筛选）

```
筛选条件（全部满足方可入选）：
  ✓ 过去 14 天价格波动最小（低波动率）
  ✓ 当前价格区间在 0.10 ~ 0.90 之间（排除极端概率市场）
  ✓ 市场深度 > $10,000 USDC（有足够流动性缓冲冲击）
  ✓ 每日奖励池 > 1.00 USDC（最低盈利门槛）
  ✓ 结算周期在 15 ~ 90 天之间（避免即将结算的高风险市场）
```

### 第二层：认知护盾（人工排除）

```
排除条件（任意满足则剔除）：
  ✗ 高政治/法律风险（结果模糊）
  ✗ 市场规则存在歧义
  ✗ 存在内幕交易迹象（巨鲸主导、异常成交）
  ✗ 选举类市场（高度不确定，2024年后奖励锐减）
```

---

## 三、报价定价算法

### 3.1 基础报价模型

基于 **Stoikov 模型**改编，用于在二元市场（YES/NO）中计算最优挂单价：

```
公式：
  Mid Price (P_mid)  = 市场当前中间价（取最优买一/卖一均值）

  Ask Price (P_ask)  = P_mid + spread/2 + γ × σ² × q × T
  Bid Price (P_bid)  = P_mid - spread/2 + γ × σ² × q × T

参数说明：
  γ   = 风险厌恶系数（建议值 0.1 ~ 0.5）
  σ²  = 价格方差（用历史滚动窗口估计，建议 3h/24h/7d 多周期加权）
  q   = 当前库存净头寸（正值 = 持有 YES，负值 = 持有 NO）
  T   = 距结算剩余时间（以天为单位）
```

**核心逻辑：** 当你持有过多 YES 头寸时（q > 0），报价向上偏移（提高卖出意愿，降低买入意愿），主动引导市场减少你的库存风险。

### 3.2 自适应价差（Adaptive Spread）

根据市场流动性动态调整价差宽度：

```python
def calculate_spread(volatility_24h, market_depth, base_spread=0.02):
    """
    volatility_24h: 过去24小时价格标准差
    market_depth:   市值深度（USD）
    base_spread:    基础价差（默认 2 美分）
    """
    # 波动率调整：波动越大，价差越宽
    vol_multiplier = 1 + (volatility_24h / 0.05)  # 以5%波动为基准

    # 深度调整：市场越浅，价差越宽（风险补偿）
    depth_multiplier = max(1.0, 10000 / market_depth)

    spread = base_spread * vol_multiplier * depth_multiplier

    # 价差上限控制（避免过宽导致无成交）
    return min(spread, 0.08)  # 最宽 8 美分
```

### 3.3 多时间框架波动率估计

```
volatility_score = w1 × σ_3h + w2 × σ_24h + w3 × σ_7d + w4 × σ_30d

推荐权重：
  w1 = 0.40  (3小时，反映即时波动)
  w2 = 0.30  (24小时)
  w3 = 0.20  (7天)
  w4 = 0.10  (30天，长期背景)

市场分类：
  低波动  ≤ 0.03  → 紧价差 (0.02~0.03)
  中波动  ≤ 0.07  → 标准价差 (0.03~0.05)
  高波动  > 0.07  → 宽价差 (0.05~0.08) 或暂停做市
```

---

## 四、库存管理算法

库存风险是做市最大威胁。一旦市场单边运动，做市商会被迫持有大量亏损头寸。

### 4.1 库存限制

```
每个市场最大净头寸：$500 USDC（相当于初始资金的 5%）
全部市场总敞口上限：$3,000 USDC

触发对冲条件：
  |净头寸| > $300 → 开始报价偏斜（倾向减仓方向）
  |净头寸| > $500 → 暂停新增加仓方向的挂单
  |净头寸| > $700 → 触发强制平仓（市价减仓）
```

### 4.2 报价偏斜（Quote Skewing）

```python
def skew_quotes(bid, ask, inventory, max_inventory=500):
    """
    根据库存方向偏移报价，引导市场平衡库存
    """
    skew_factor = inventory / max_inventory  # -1 到 +1

    # 持有过多 YES（inventory > 0）→ 压低买价、提高卖价（促进卖出）
    bid_skewed = bid - skew_factor * 0.01
    ask_skewed = ask - skew_factor * 0.01

    return bid_skewed, ask_skewed
```

### 4.3 临近结算风险处理

```
距结算日期   → 处理策略
> 30 天      正常做市
15~30 天     将价差扩大 1.5 倍
7~15 天      将价差扩大 2 倍，降低库存上限至 50%
< 7 天       停止新开仓，仅维持现有订单或主动平仓
< 1 天       全部撤单，不再参与
```

---

## 五、流动性奖励最大化

Polymarket 的奖励公式采用 **二次方评分（Quadratic Scoring）**，倾向于奖励更紧密且持续的双边报价。

### 5.1 奖励估算公式

```
daily_reward ≈ pool_reward × (your_score / total_score)

your_score = Σ [size_i × proximity_score_i × time_fraction_i]

proximity_score = max(0, 1 - |P_order - P_mid| / max_distance)
  其中 max_distance = 0.05（距中间价超过5%则不计分）

关键结论：
  双边挂单奖励系数 ≈ 3 × 单边挂单奖励系数
  挂单距中间价每增加 1%，奖励下降约 20%
```

### 5.2 资本分配策略

```
目标：最大化 reward_per_dollar_deployed

资本分配优先级（按 risk_adjusted_reward 排序）：
  risk_adjusted_reward = daily_reward_estimate / (volatility × depth_risk)

建议分配：
  前 3 名市场：每市场 $2,000
  4~8 名市场：每市场 $500
  总部署上限：$10,000
```

---

## 六、事件风险防护（Circuit Breaker）

### 6.1 自动熔断条件

```
触发条件                          → 响应措施
-------------------------------------------------------------------
价格在 60 秒内变动 > 5%           撤销全部挂单，暂停 5 分钟
24 小时亏损 > $200               暂停该市场 24 小时
异常成交量（> 10× 均值）         扩大价差 3 倍
接入新闻 API 检测到重大事件       全市场暂停报价
```

### 6.2 新闻 API 集成

```python
class EventRiskMonitor:
    """
    监控可能影响预测市场的突发事件
    """
    RISK_KEYWORDS = [
        "breaking", "confirmed", "ruling", "verdict",
        "result", "official", "announces", "wins"
    ]

    def should_pause_market(self, market_title: str, news_feed: list) -> bool:
        """
        检查是否有与市场相关的突发新闻，如有则暂停做市
        """
        for article in news_feed[-50:]:  # 检查最近50条新闻
            if any(kw in article['title'].lower() for kw in self.RISK_KEYWORDS):
                if self._is_related(market_title, article):
                    return True
        return False
```

---

## 七、整体执行流程

```
┌─────────────────────────────────────────────────────────┐
│                    做市机器人主循环                        │
│                  （每 30 秒执行一次）                      │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 1. 市场数据采集      │  拉取价格、深度、历史波动率、奖励池数据
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 2. 市场评分排名      │  按 risk_adjusted_reward 排序
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 3. 风险检查          │  检查库存、熔断条件、新闻事件
└─────────────────────┘
         │ 通过
         ▼
┌─────────────────────┐
│ 4. 定价计算          │  Stoikov 模型 + 自适应价差 + 库存偏斜
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 5. 订单管理          │  撤销失效订单 → 按新价格重新挂单
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ 6. 绩效追踪          │  记录 PnL、奖励收益、价差收益
└─────────────────────┘
         │
         ▼ 等待 30 秒后重复
```

---

## 八、绩效基准与预期收益

基于 $10,000 起始资金的历史参考数据：

| 指标 | 入门期 | 成熟期 |
|------|--------|--------|
| 日均收益 | $150 ~ $200 | $500 ~ $800 |
| 价差收益占比 | ~30% | ~40% |
| 奖励收益占比 | ~70% | ~60% |
| 最大日亏损 | -$300 | -$500 |
| 平均参与市场数 | 5 ~ 8 个 | 15 ~ 25 个 |
| 策略夏普比率 | 1.5 ~ 2.5 | — |

> ⚠️ **风险提示：** 2024 年大选后 Polymarket 大幅削减流动性奖励，当前盈利空间已收窄。
> 开源 poly-maker 作者明确表示"当前市场环境下该策略亏损"，建议将其作为参考实现。

---

## 九、与现有 Copy Trading Bot 的集成建议

本项目当前为 Copy Trading Bot，可考虑以下集成路径：

```
方案 A：并行模式（推荐）
  - Copy Trading 负责方向性头寸（跟随聪明钱）
  - Market Making 负责已持仓市场的双边流动性（降低持仓成本）

方案 B：信号增强
  - 将聪明钱持仓偏向作为 Stoikov 模型的 P_mid 偏移信号
  - 如目标 trader 持有大量 YES → P_mid 上移 → 做市方向偏多

方案 C：奖励套利
  - 纯粹以获取流动性奖励为目的
  - 选择 Copy Trading 未覆盖的低波动市场独立做市
```

---

## 十、参考资料

- [Polymarket 官方：Automated Market Making on Polymarket](https://news.polymarket.com/p/automated-market-making-on-polymarket)
- [Phemex：Polymarket's Stable Market Making Strategy](https://phemex.com/news/article/polymarkets-strategy-for-stable-market-making-43240)
- [NYC Servers：Market Making on Prediction Markets Complete 2026 Guide](https://newyorkcityservers.com/blog/prediction-market-making-guide)
- [GitHub：warproxxx/poly-maker](https://github.com/warproxxx/poly-maker)
- [Polymarket CLOB Docs](https://docs.polymarket.com/market-makers/overview)

---

*生成时间：2026-03-17 | 分支：claude/market-making-strategy-report-BHFYX*
