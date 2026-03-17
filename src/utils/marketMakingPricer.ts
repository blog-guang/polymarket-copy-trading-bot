/**
 * Market Making Pricer
 *
 * Combines:
 *   1. Stoikov inventory model  — adjusts mid-price based on net position
 *   2. Paper's three risk factors (σ, λ, calendar effect) for spread sizing
 *
 * Reference: "Toward Black-Scholes for Prediction Markets" (arxiv 2510.15205)
 *
 * Spread formula:
 *   halfSpread = (baseSpread / 2) × spreadMultiplier × calendarMultiplier
 *
 *   spreadMultiplier = 1 + γ·q² + β·σ² + ζ·λ
 *     γ = riskAversion     (inventory risk, default 0.2)
 *     q = netPosition / maxInventory  (normalised to [-1, +1])
 *     β = volSensitivity   (belief-vol risk, default 1.0)
 *     σ = diffusion volatility from EM estimate
 *     ζ = jumpSensitivity  (jump-risk premium, default 0.5)
 *     λ = jump intensity from EM estimate
 *
 *   calendarMultiplier = 1 + (1 - τ)^3 × calendarFactor
 *     τ = daysToResolution / totalMarketDuration  ∈ [0, 1]
 *
 *   Inventory-adjusted mid (Stoikov):
 *     adjustedMid = fairValue − γ · σ² · q
 */

export interface QuoteParams {
    /** Fair value (probability) estimated by the EM model */
    fairValue: number;
    /** Diffusion volatility σ from jump-diffusion EM */
    sigma: number;
    /** Jump intensity λ (jumps/day) from EM */
    lambda: number;
    /** Net token inventory (positive = long YES) in USD */
    netPosition: number;
    /** Maximum allowed inventory per market in USD */
    maxInventory: number;
    /** Base spread (minimum half-spread × 2), e.g. 0.02 */
    baseSpread: number;
    /** Maximum spread cap, e.g. 0.08 */
    maxSpread: number;
    /** Inventory risk-aversion coefficient γ */
    riskAversion: number;
    /** Belief-volatility sensitivity β */
    volSensitivity: number;
    /** Jump-intensity sensitivity ζ */
    jumpSensitivity: number;
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
}

/**
 * Compute bid/ask quotes using the Stoikov + paper risk-factor model.
 */
export function computeQuotes(p: QuoteParams): Quotes {
    // ── 1. Normalise inventory to [-1, +1] ────────────────────────────────────
    const q = p.maxInventory > 0
        ? Math.max(-1, Math.min(1, p.netPosition / p.maxInventory))
        : 0;

    // ── 2. Spread multiplier (paper's three risk factors) ─────────────────────
    const inventoryRisk = p.riskAversion * q * q;
    const volRisk = p.volSensitivity * p.sigma * p.sigma;
    // λ is normalised to [0,1] before applying sensitivity:
    //   λ_norm = λ / (λ + 1)  (logistic-style saturation)
    // This prevents λ >> 1 (high-frequency microstructure noise) from dominating.
    const lambdaNorm = p.lambda / (p.lambda + 1);
    const jumpRisk = p.jumpSensitivity * lambdaNorm;
    const spreadMultiplier = 1 + inventoryRisk + volRisk + jumpRisk;

    // ── 3. Calendar multiplier (paper §4 – convergence to certainty) ─────────
    //   τ → 0 near resolution → multiplier spikes (wider spread = bigger risk)
    const tau = p.totalMarketDuration > 0
        ? Math.max(0, Math.min(1, p.daysToResolution / p.totalMarketDuration))
        : 1;
    const calendarMultiplier = 1 + Math.pow(1 - tau, 3) * p.calendarFactor;

    // ── 4. Half-spread ────────────────────────────────────────────────────────
    const rawHalfSpread = (p.baseSpread / 2) * spreadMultiplier * calendarMultiplier;
    const halfSpread = Math.min(rawHalfSpread, p.maxSpread / 2);

    // ── 5. Inventory-adjusted mid (Stoikov skewing) ───────────────────────────
    //   Holding too much YES → lower mid → more attractive to sell
    const inventoryBias = p.riskAversion * p.sigma * p.sigma * q;
    const adjustedMid = Math.max(halfSpread + 0.01, Math.min(1 - halfSpread - 0.01,
        p.fairValue - inventoryBias
    ));

    // ── 6. Final quotes clamped to valid probability range ────────────────────
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
 * Compute risk-adjusted reward score for market ranking.
 *
 * score = dailyRewardEstimate / (sigma × max(lambda, 0.01))
 *
 * Higher score = better risk/reward for market making.
 */
export function riskAdjustedScore(
    dailyRewardEstimate: number,
    sigma: number,
    lambda: number
): number {
    const risk = Math.max(sigma, 0.001) * Math.max(lambda, 0.01);
    return dailyRewardEstimate / risk;
}

/**
 * Determine the order size for each side of the quote.
 * Scales down when inventory is already skewed (reduces adverse selection).
 *
 * @param baseOrderSizeUSD  Configured base order size
 * @param normalizedInventory  Inventory q in [-1, +1]
 * @returns { bidSize, askSize }
 */
export function computeOrderSizes(
    baseOrderSizeUSD: number,
    normalizedInventory: number
): { bidSize: number; askSize: number } {
    // Reduce buy-side size when we are already long (q > 0)
    // Reduce sell-side size when we are already short (q < 0)
    const bidScale = Math.max(0.25, 1 - Math.max(0, normalizedInventory));
    const askScale = Math.max(0.25, 1 - Math.max(0, -normalizedInventory));

    return {
        bidSize: parseFloat((baseOrderSizeUSD * bidScale).toFixed(2)),
        askSize: parseFloat((baseOrderSizeUSD * askScale).toFixed(2)),
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
