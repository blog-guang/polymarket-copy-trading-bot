/**
 * Market Making Pricer
 *
 * Combines:
 *   1. Stoikov inventory model  — adjusts mid-price based on net position
 *   2. Paper's three risk factors (σ, λ, calendar effect) for spread sizing
 *   3. Volatility regime gating — EXTREME markets return null (skip)
 *   4. Trend-aware mid skewing  — lean quotes in direction of detected trend
 *
 * Reference: "Toward Black-Scholes for Prediction Markets" (arxiv 2510.15205)
 *
 * Spread formula:
 *   halfSpread = (regimeBaseSpread / 2) × spreadMultiplier × calendarMultiplier
 *
 *   spreadMultiplier = 1 + γ·q² + β·σ² + ζ·λ_norm
 *     γ = riskAversion     (inventory risk, default 0.2)
 *     q = netPosition / maxInventory  (normalised to [-1, +1])
 *     β = volSensitivity   (belief-vol risk, default 1.0)
 *     σ = diffusion volatility from hybrid EM estimate
 *     ζ = jumpSensitivity  (jump-risk premium, default 0.5)
 *     λ_norm = λ/(λ+1)    (logistic saturation to avoid microstructure inflation)
 *
 *   Regime-dependent base spread:
 *     CALM     → calmBaseSpread (e.g. 0.01) — tighter quotes = more reward share
 *     NORMAL   → baseSpread    (e.g. 0.02)
 *     VOLATILE → baseSpread × 1.5
 *     EXTREME  → null (do not quote)
 *
 *   calendarMultiplier = 1 + (1 - τ)^3 × calendarFactor
 *     τ = daysToResolution / totalMarketDuration  ∈ [0, 1]
 *
 *   Inventory-adjusted mid (Stoikov):
 *     adjustedMid = fairValue − γ · σ² · q
 *
 *   Trend lean (additional skew):
 *     trendBias = trend × trendThreshold × halfSpread
 *     finalMid  = adjustedMid + trendBias
 *     (trend ∈ [-1,+1]; positive = bullish → shift mid up → more attractive ask)
 */

import { VolatilityRegime } from './jumpDiffusionModel';

export interface QuoteParams {
    /** Fair value (probability) estimated by the EM model */
    fairValue: number;
    /** Diffusion volatility σ from hybrid jump-diffusion EM */
    sigma: number;
    /** Jump intensity λ (jumps/day) from hourly EM — realistic scale */
    lambda: number;
    /** Volatility regime classification */
    regime: VolatilityRegime;
    /** Directional trend signal in [-1, +1] (from detectTrend) */
    trend: number;
    /** Net token inventory (positive = long YES) in USD */
    netPosition: number;
    /** Maximum allowed inventory per market in USD */
    maxInventory: number;
    /** Base spread (normal regime), e.g. 0.02 */
    baseSpread: number;
    /** Base spread for CALM regime, e.g. 0.01 */
    calmBaseSpread: number;
    /** Maximum spread cap, e.g. 0.08 */
    maxSpread: number;
    /** Inventory risk-aversion coefficient γ */
    riskAversion: number;
    /** Belief-volatility sensitivity β */
    volSensitivity: number;
    /** Jump-intensity sensitivity ζ */
    jumpSensitivity: number;
    /** Trend lean factor (fraction of halfSpread to shift mid) */
    trendThreshold: number;
    /** Days remaining to market resolution */
    daysToResolution: number;
    /** Total market lifetime in days (creation → resolution) */
    totalMarketDuration: number;
    /** Calendar amplification factor (default 2.0) */
    calendarFactor: number;
}

export interface Quotes {
    bid: number;
    ask: number;
    halfSpread: number;
    adjustedMid: number;
    spreadMultiplier: number;
    calendarMultiplier: number;
    normalizedInventory: number;
    regimeBaseSpread: number;
}

/**
 * Compute bid/ask quotes using the Stoikov + paper risk-factor model.
 * Returns null for EXTREME volatility regime (market should be skipped).
 */
export function computeQuotes(p: QuoteParams): Quotes | null {
    // ── 0. Hard gate on EXTREME regime ────────────────────────────────────────
    if (p.regime === 'EXTREME') return null;

    // ── 1. Regime-dependent base spread ───────────────────────────────────────
    let regimeBaseSpread: number;
    switch (p.regime) {
        case 'CALM':
            regimeBaseSpread = p.calmBaseSpread;
            break;
        case 'VOLATILE':
            regimeBaseSpread = p.baseSpread * 1.5;
            break;
        default: // NORMAL
            regimeBaseSpread = p.baseSpread;
    }

    // ── 2. Normalise inventory to [-1, +1] ────────────────────────────────────
    const q = p.maxInventory > 0
        ? Math.max(-1, Math.min(1, p.netPosition / p.maxInventory))
        : 0;

    // ── 3. Spread multiplier (paper's three risk factors) ─────────────────────
    const inventoryRisk = p.riskAversion * q * q;
    const volRisk = p.volSensitivity * p.sigma * p.sigma;
    // λ is normalised to [0,1] before applying sensitivity:
    //   λ_norm = λ / (λ + 1)  (logistic-style saturation)
    // This prevents λ >> 1 (high-frequency microstructure noise) from dominating.
    const lambdaNorm = p.lambda / (p.lambda + 1);
    const jumpRisk = p.jumpSensitivity * lambdaNorm;
    const spreadMultiplier = 1 + inventoryRisk + volRisk + jumpRisk;

    // ── 4. Calendar multiplier (paper §4 – convergence to certainty) ──────────
    //   τ → 0 near resolution → multiplier spikes (wider spread = bigger risk)
    const tau = p.totalMarketDuration > 0
        ? Math.max(0, Math.min(1, p.daysToResolution / p.totalMarketDuration))
        : 1;
    const calendarMultiplier = 1 + Math.pow(1 - tau, 3) * p.calendarFactor;

    // ── 5. Half-spread ─────────────────────────────────────────────────────────
    const rawHalfSpread = (regimeBaseSpread / 2) * spreadMultiplier * calendarMultiplier;
    const halfSpread = Math.min(rawHalfSpread, p.maxSpread / 2);

    // ── 6. Inventory-adjusted mid (Stoikov skewing) ────────────────────────────
    //   Holding too much YES → lower mid → more attractive to sell
    const inventoryBias = p.riskAversion * p.sigma * p.sigma * q;

    // ── 7. Trend lean — shift mid AGAINST the detected trend ─────────────────
    //   Lean AGAINST the trend to stay competitive while reducing adverse selection:
    //   Downtrend (trend < 0) → shift mid DOWN → our bid is lower → less likely to
    //     be hit by informed sellers; our ask is also lower → can still capture fills
    //   Uptrend   (trend > 0) → shift mid UP → our ask is higher → we sell at premium
    //   This is consistent with Stoikov: mid adjustment reduces adverse selection
    //   Clamped so trend alone never moves mid by more than 1 full halfSpread
    const trendBias = Math.max(-halfSpread, Math.min(halfSpread,
        p.trend * p.trendThreshold * halfSpread
    ));

    const rawMid = p.fairValue - inventoryBias + trendBias;
    const adjustedMid = Math.max(halfSpread + 0.01, Math.min(1 - halfSpread - 0.01, rawMid));

    // ── 8. Final quotes clamped to valid probability range ────────────────────
    const bid = parseFloat(Math.max(0.01, adjustedMid - halfSpread).toFixed(3));
    const ask = parseFloat(Math.min(0.99, adjustedMid + halfSpread).toFixed(3));

    return {
        bid,
        ask,
        halfSpread,
        adjustedMid,
        spreadMultiplier,
        calendarMultiplier,
        normalizedInventory: q,
        regimeBaseSpread,
    };
}

/**
 * Estimate the daily reward contribution for a given market.
 * Used for market ranking (risk-adjusted reward score).
 *
 * This is a simplified approximation of the quadratic scoring rule:
 *   score ≈ deployedCapital × proximityFactor
 *
 * where proximityFactor rewards tighter quotes near mid.
 *
 * @param deployedCapitalUSD  Capital on each side of the market
 * @param halfSpread         Our half-spread (distance from mid)
 * @param rewardPoolUSD      Estimated daily reward pool for the market
 * @param totalLiquidityUSD  Estimated total liquidity in the market
 */
export function estimateDailyReward(
    deployedCapitalUSD: number,
    halfSpread: number,
    rewardPoolUSD: number,
    totalLiquidityUSD: number
): number {
    if (totalLiquidityUSD <= 0 || rewardPoolUSD <= 0) return 0;

    // Proximity factor: orders within 5% of mid receive up to 1.0; further orders get 0
    const maxDistance = 0.05;
    const proximityScore = Math.max(0, 1 - halfSpread / maxDistance);

    // Our share of total liquidity (two-sided → 2× deployed capital)
    const ourLiquidity = 2 * deployedCapitalUSD * proximityScore;
    const marketShare = ourLiquidity / (totalLiquidityUSD + ourLiquidity);

    return rewardPoolUSD * marketShare;
}

/**
 * Compute regime-aware risk-adjusted reward score for market ranking.
 *
 * score = dailyRewardEstimate × regimeMultiplier / (sigma × max(lambda, 0.01))
 *
 * regimeMultiplier:
 *   CALM     → 2.0  (best reward/risk; tight spreads capture more reward pool share)
 *   NORMAL   → 1.0
 *   VOLATILE → 0.5  (penalise for higher adverse selection risk)
 *   EXTREME  → 0.0  (excluded from selection)
 *
 * Higher score = better risk/reward for market making.
 */
export function riskAdjustedScore(
    dailyRewardEstimate: number,
    sigma: number,
    lambda: number,
    regime: VolatilityRegime
): number {
    const regimeMultiplier: Record<VolatilityRegime, number> = {
        CALM: 2.0,
        NORMAL: 1.0,
        VOLATILE: 0.5,
        EXTREME: 0.0,
    };
    if (regime === 'EXTREME') return 0;
    const risk = Math.max(sigma, 0.001) * Math.max(lambda, 0.01);
    return (dailyRewardEstimate * regimeMultiplier[regime]) / risk;
}

/**
 * Determine the order size for each side of the quote.
 *
 * Scales by volatility regime:
 *   CALM     → full size (best fill quality, low adverse selection)
 *   NORMAL   → full size
 *   VOLATILE → 60% size (reduce exposure)
 *   EXTREME  → 0 (should not be called, but safe fallback)
 *
 * Also scales down when inventory is already skewed (reduces adverse selection).
 *
 * @param baseOrderSizeUSD  Configured base order size
 * @param normalizedInventory  Inventory q in [-1, +1]
 * @param regime  Current volatility regime
 * @param trend   Trend signal [-1, +1] — enables one-sided quoting when strong
 * @param trendThreshold  Minimum |trend| to apply one-sided quoting
 * @returns { bidSize, askSize }
 */
export function computeOrderSizes(
    baseOrderSizeUSD: number,
    normalizedInventory: number,
    regime: VolatilityRegime = 'NORMAL',
    trend = 0,
    trendThreshold = 0.3
): { bidSize: number; askSize: number } {
    // Regime scaling
    const regimeScale: Record<VolatilityRegime, number> = {
        CALM: 1.0,
        NORMAL: 1.0,
        VOLATILE: 0.6,
        EXTREME: 0.0,
    };
    const rScale = regimeScale[regime];

    // Reduce buy-side size when we are already long (q > 0)
    // Reduce sell-side size when we are already short (q < 0)
    const bidScale = Math.max(0.25, 1 - Math.max(0, normalizedInventory));
    const askScale = Math.max(0.25, 1 - Math.max(0, -normalizedInventory));

    // Adverse-selection aware one-sided quoting:
    //   Uptrend   → informed buyers hit our ASK (we sell too cheap) → reduce ASK size
    //   Downtrend → informed sellers hit our BID (we buy at a loss)  → reduce BID size
    //
    // Two-tier scaling:
    //   |trend| ∈ [threshold, 0.5) → gradually scale down adverse side
    //   |trend| ≥ 0.5              → hard stop on adverse side (very small minimum)
    let trendBidScale = 1.0;
    let trendAskScale = 1.0;
    if (trend > trendThreshold) {
        // Bullish: informed buyers take our asks → reduce ask exposure
        trendAskScale = trend >= 0.5 ? 0.05 : Math.max(0.1, 1 - trend);
    } else if (trend < -trendThreshold) {
        // Bearish: informed sellers take our bids → reduce bid exposure
        trendBidScale = trend <= -0.5 ? 0.05 : Math.max(0.1, 1 + trend);
    }

    return {
        bidSize: parseFloat((baseOrderSizeUSD * rScale * bidScale * trendBidScale).toFixed(2)),
        askSize: parseFloat((baseOrderSizeUSD * rScale * askScale * trendAskScale).toFixed(2)),
    };
}

/**
 * Decide whether quotes need to be refreshed.
 * Avoids unnecessary cancel/re-place cycles when prices haven't moved.
 */
export function shouldReprice(
    currentBid: number,
    currentAsk: number,
    newBid: number,
    newAsk: number,
    threshold: number   // e.g. 0.005 = 0.5 cents
): boolean {
    return Math.abs(currentBid - newBid) > threshold ||
        Math.abs(currentAsk - newAsk) > threshold;
}
