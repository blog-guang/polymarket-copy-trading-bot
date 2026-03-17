/**
 * Jump-Diffusion Model for Prediction Markets
 *
 * Implements the logit jump-diffusion stochastic process described in:
 *   "Toward Black-Scholes for Prediction Markets" (arxiv 2510.15205, Shaw Dalen)
 *
 * Core insight: prediction market prices, when transformed to log-odds (logit) space,
 * follow a mixed process of:
 *   - Continuous diffusion: dX = σ dW  (belief drift)
 *   - Discrete jumps: +J_i at rate λ   (news / information shocks)
 *
 * We use an EM (Expectation-Maximization) algorithm to separate the two components
 * and estimate σ (belief volatility) and λ (jump intensity) from price history.
 *
 * Volatility regime classification:
 *   CALM    σ < 0.02  — stable, tight spreads, ideal for MM
 *   NORMAL  σ < 0.06  — standard operation
 *   VOLATILE σ < 0.10 — wide spreads, reduced size
 *   EXTREME σ ≥ 0.10  — skip market entirely
 */

// ── Volatility regime ─────────────────────────────────────────────────────────

/**
 * Volatility regime based on diffusion σ (log-odds space, per √day).
 * Used to gate market selection and scale order sizes.
 */
export type VolatilityRegime = 'CALM' | 'NORMAL' | 'VOLATILE' | 'EXTREME';

/**
 * Classify the volatility regime from a σ estimate.
 * Thresholds are calibrated against the backtest: EXTREME (≥0.10) produced
 * catastrophic losses (-190% ROI) in 14-day backtests; CALM (<0.02) produced
 * the best risk-adjusted returns.
 */
export function classifyRegime(sigma: number): VolatilityRegime {
    if (sigma < 0.02) return 'CALM';
    if (sigma < 0.06) return 'NORMAL';
    if (sigma < 0.10) return 'VOLATILE';
    return 'EXTREME';
}

// ── Helper math ───────────────────────────────────────────────────────────────

/** logit transform: price → log-odds */
export function logit(p: number): number {
    const clamped = Math.max(1e-6, Math.min(1 - 1e-6, p));
    return Math.log(clamped / (1 - clamped));
}

/** logistic transform: log-odds → price */
export function logistic(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

function mean(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function variance(arr: number[], mu?: number): number {
    if (arr.length < 2) return 1e-8;
    const m = mu ?? mean(arr);
    return arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
}

function std(arr: number[], mu?: number): number {
    return Math.sqrt(variance(arr, mu));
}

/** Normal PDF */
function normalPdf(x: number, mu: number, sigma: number): number {
    const s = Math.max(sigma, 1e-10);
    return (1 / (s * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * ((x - mu) / s) ** 2);
}

/** Exponential moving average */
export function ema(series: number[], alpha = 0.3): number {
    if (series.length === 0) return 0.5;
    let val = series[0];
    for (let i = 1; i < series.length; i++) {
        val = alpha * series[i] + (1 - alpha) * val;
    }
    return val;
}

// ── Parameter estimation ──────────────────────────────────────────────────────

export interface JumpDiffusionParams {
    /** Diffusion volatility in log-odds space (annualised, per √day) */
    sigma: number;
    /** Jump arrival rate (jumps per day) */
    lambda: number;
    /** Mean jump size in log-odds */
    muJump: number;
    /** Jump size standard deviation in log-odds */
    sigmaJump: number;
    /** Estimated fair value (probability 0–1) */
    fairValue: number;
    /** Number of price observations used */
    nObs: number;
    /** EM converged flag */
    converged: boolean;
}

/**
 * Estimate jump-diffusion parameters via EM algorithm.
 *
 * @param prices  Array of mid-price snapshots in [0, 1]
 * @param timestamps  Corresponding Unix timestamps (ms).  If omitted, uniform 60s spacing assumed.
 * @param maxIter  EM iteration limit (default 50)
 * @param tol  Log-likelihood convergence threshold (default 1e-4)
 */
export function estimateJumpDiffusion(
    prices: number[],
    timestamps?: number[],
    maxIter = 50,
    tol = 1e-4
): JumpDiffusionParams {
    const DEFAULT_PARAMS: JumpDiffusionParams = {
        sigma: 0.05,
        lambda: 0.1,
        muJump: 0,
        sigmaJump: 0.2,
        fairValue: prices.length > 0 ? prices[prices.length - 1] : 0.5,
        nObs: prices.length,
        converged: false,
    };

    if (prices.length < 5) return DEFAULT_PARAMS;

    // ── Step 1: transform to log-odds ─────────────────────────────────────────
    const logOdds = prices.map(logit);

    // ── Step 2: compute returns and time deltas ───────────────────────────────
    const n = logOdds.length - 1;
    const returns: number[] = [];
    const dts: number[] = [];   // time delta in days

    for (let i = 0; i < n; i++) {
        returns.push(logOdds[i + 1] - logOdds[i]);
        if (timestamps && timestamps[i + 1] > timestamps[i]) {
            dts.push((timestamps[i + 1] - timestamps[i]) / (1000 * 60 * 60 * 24));
        } else {
            dts.push(1 / (24 * 60)); // default: 1-minute intervals
        }
    }

    const meanDt = mean(dts);

    // ── Step 3: initialise EM parameters ─────────────────────────────────────
    const returnStd = std(returns);
    let sigmaD = returnStd * 0.7;               // diffusion component (scaled to dt)
    let lambda = 1.0;                           // jumps/day
    let pi = Math.min(0.15, lambda * meanDt);   // mixture weight: P(jump in interval)
    let muJ = 0.0;
    let sigmaJ = returnStd * 1.5;

    let prevLL = -Infinity;
    let converged = false;

    for (let iter = 0; iter < maxIter; iter++) {
        // ── E-step: compute posterior P(jump | return[i], dt[i]) ──────────────
        const gammaN: number[] = []; // P(no jump)
        const gammaJ: number[] = []; // P(jump)

        for (let i = 0; i < n; i++) {
            const dtSqrt = Math.sqrt(dts[i]);
            const pNoJump = (1 - pi) * normalPdf(returns[i], 0, sigmaD * dtSqrt);
            const pJump = pi * normalPdf(returns[i], muJ, Math.sqrt(sigmaD ** 2 * dts[i] + sigmaJ ** 2));
            const total = pNoJump + pJump + 1e-300;
            gammaN.push(pNoJump / total);
            gammaJ.push(pJump / total);
        }

        // ── M-step: update parameters ─────────────────────────────────────────
        const sumGJ = gammaJ.reduce((s, v) => s + v, 0);
        const sumGN = gammaN.reduce((s, v) => s + v, 0);

        // Jump mixture weight
        pi = Math.max(0.001, Math.min(0.499, sumGJ / n));

        // Jump mean & std (from jump-attributed returns)
        if (sumGJ > 0.5) {
            muJ = gammaJ.reduce((s, gj, i) => s + gj * returns[i], 0) / sumGJ;
            sigmaJ = Math.max(
                1e-4,
                Math.sqrt(gammaJ.reduce((s, gj, i) => s + gj * (returns[i] - muJ) ** 2, 0) / sumGJ)
            );
        }

        // Diffusion volatility (from non-jump-attributed returns, normalized by sqrt(dt))
        const sigmaNum = gammaN.reduce((s, gn, i) => s + gn * returns[i] ** 2 / dts[i], 0);
        const sigmaDen = sumGN;
        sigmaD = sigmaDen > 0.5 ? Math.max(1e-4, Math.sqrt(sigmaNum / sigmaDen)) : sigmaD;

        // Jump intensity: λ = π / mean(dt)
        lambda = pi / meanDt;

        // ── Log-likelihood ────────────────────────────────────────────────────
        let ll = 0;
        for (let i = 0; i < n; i++) {
            const dtSqrt = Math.sqrt(dts[i]);
            const pNoJump = (1 - pi) * normalPdf(returns[i], 0, sigmaD * dtSqrt);
            const pJump = pi * normalPdf(returns[i], muJ, Math.sqrt(sigmaD ** 2 * dts[i] + sigmaJ ** 2));
            ll += Math.log(pNoJump + pJump + 1e-300);
        }

        if (Math.abs(ll - prevLL) < tol) {
            converged = true;
            break;
        }
        prevLL = ll;
    }

    // ── Fair value: EMA of logit prices, mapped back to probability ───────────
    const fairValue = logistic(ema(logOdds, 0.3));

    return {
        sigma: Math.max(0.001, sigmaD),
        lambda: Math.max(0.001, lambda),
        muJump: muJ,
        sigmaJump: sigmaJ,
        fairValue,
        nObs: prices.length,
        converged,
    };
}

// ── Volatility surface utilities ──────────────────────────────────────────────

/**
 * Multi-timeframe volatility estimator.
 * Returns a weighted composite sigma using 3h / 24h / 7d / 30d windows.
 */
export function multiTimeframeVolatility(
    prices: number[],
    timestamps: number[]
): {
    vol3h: number;
    vol24h: number;
    vol7d: number;
    vol30d: number;
    composite: number;
} {
    const now = timestamps[timestamps.length - 1] ?? Date.now();
    const msHour = 3_600_000;

    function windowVol(windowMs: number): number {
        const cutoff = now - windowMs;
        const idx = timestamps.findIndex((t) => t >= cutoff);
        const slice = idx >= 0 ? prices.slice(idx) : prices;
        if (slice.length < 3) return 0.05; // fallback
        const { sigma } = estimateJumpDiffusion(slice);
        return sigma;
    }

    const vol3h = windowVol(3 * msHour);
    const vol24h = windowVol(24 * msHour);
    const vol7d = windowVol(7 * 24 * msHour);
    const vol30d = windowVol(30 * 24 * msHour);

    // Weighted composite (heavier weight on recent data)
    const composite = 0.40 * vol3h + 0.30 * vol24h + 0.20 * vol7d + 0.10 * vol30d;

    return { vol3h, vol24h, vol7d, vol30d, composite };
}

/**
 * Detect if a return series contains a "jump event" in the last interval.
 * Used for circuit-breaker triggering.
 *
 * @param recentPrices  Last N prices (at least 2)
 * @param thresholdLogOdds  Jump threshold in log-odds space (default 0.4 ≈ 10% price move)
 */
export function detectRecentJump(recentPrices: number[], thresholdLogOdds = 0.4): boolean {
    if (recentPrices.length < 2) return false;
    const n = recentPrices.length;
    const last = logit(recentPrices[n - 1]);
    const prev = logit(recentPrices[n - 2]);
    return Math.abs(last - prev) > thresholdLogOdds;
}

// ── Regime-aware hybrid estimation ────────────────────────────────────────────

/**
 * Downsample a (prices, timestamps) series to one representative point per
 * `targetMs` millisecond bucket (last-price-in-bucket).
 *
 * Motivation: EM on 1-minute data inflates λ to 400-600/day because
 * microstructure noise is classified as "jumps".  Downsampling to hourly
 * gives realistic λ ≈ 0.1-2.0/day while 1-min data is still used for σ.
 */
export function resamplePrices(
    prices: number[],
    timestamps: number[],
    targetMs: number
): { prices: number[]; timestamps: number[] } {
    if (prices.length === 0) return { prices: [], timestamps: [] };

    const buckets = new Map<number, { price: number; ts: number }>();
    for (let i = 0; i < prices.length; i++) {
        const bucket = Math.floor(timestamps[i] / targetMs);
        // Keep last price in each bucket
        buckets.set(bucket, { price: prices[i], ts: timestamps[i] });
    }

    const sorted = Array.from(buckets.entries()).sort((a, b) => a[0] - b[0]);
    return {
        prices: sorted.map((e) => e[1].price),
        timestamps: sorted.map((e) => e[1].ts),
    };
}

/**
 * Hybrid parameter estimation:
 *   - σ (diffusion volatility) estimated from high-frequency (1-min) data
 *     → captures continuous belief evolution accurately
 *   - λ (jump intensity) estimated from hourly-resampled data
 *     → avoids microstructure noise inflating jump counts
 *
 * Returns the same JumpDiffusionParams interface, combining the two estimates.
 */
export function estimateParamsHybrid(
    prices1min: number[],
    timestamps1min: number[]
): JumpDiffusionParams {
    // σ from 1-minute data (standard EM)
    const hiFreq = estimateJumpDiffusion(prices1min, timestamps1min, 30, 1e-4);

    // λ from hourly resampled data
    const hourMs = 3_600_000;
    const { prices: pricesH, timestamps: tsH } = resamplePrices(prices1min, timestamps1min, hourMs);
    const loFreq = pricesH.length >= 5
        ? estimateJumpDiffusion(pricesH, tsH, 30, 1e-4)
        : hiFreq;

    return {
        ...hiFreq,
        // Override λ with the hourly estimate (unit: jumps/day)
        lambda: loFreq.lambda,
    };
}

/**
 * Detect directional trend via linear regression on logit prices.
 *
 * Returns a normalised slope in [-1, +1]:
 *   > +trendThreshold → uptrend (bullish)
 *   < -trendThreshold → downtrend (bearish)
 *   near 0            → no significant trend
 *
 * @param prices      Price series [0,1]
 * @param timestamps  Corresponding timestamps (ms)
 * @param windowHours How many hours of history to analyse (default 6)
 * @returns slope in log-odds-per-day, normalised by the logit std of the window
 */
export function detectTrend(
    prices: number[],
    timestamps: number[],
    windowHours = 6
): number {
    if (prices.length < 4) return 0;

    const now = timestamps[timestamps.length - 1] ?? Date.now();
    const cutoff = now - windowHours * 3_600_000;
    const idx = timestamps.findIndex((t) => t >= cutoff);
    const slice = idx >= 0 ? prices.slice(idx) : prices;
    const tsSlice = idx >= 0 ? timestamps.slice(idx) : timestamps;

    if (slice.length < 4) return 0;

    // Work in logit + time (days)
    const t0 = tsSlice[0];
    const xs = tsSlice.map((t) => (t - t0) / (1000 * 60 * 60 * 24));
    const ys = slice.map(logit);

    // Simple OLS slope: β = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²
    const xMean = mean(xs);
    const yMean = mean(ys);
    let num = 0;
    let den = 0;
    for (let i = 0; i < xs.length; i++) {
        num += (xs[i] - xMean) * (ys[i] - yMean);
        den += (xs[i] - xMean) ** 2;
    }
    if (den < 1e-10) return 0;
    const slope = num / den; // log-odds per day

    // Normalise by std of y so result is ~unit-free
    const yStd = Math.max(std(ys), 0.01);
    const normalised = slope / yStd;

    // Clamp to [-1, +1]
    return Math.max(-1, Math.min(1, normalised));
}
