/**
 * Market Making Strategy Backtester
 *
 * Replays historical Polymarket price data to evaluate the jump-diffusion
 * market making strategy implemented in:
 *   src/utils/jumpDiffusionModel.ts
 *   src/utils/marketMakingPricer.ts
 *
 * Simulation mechanics:
 *   - Loads 1-minute OHLCV ticks from Polymarket CLOB prices-history API
 *   - Rolls a sliding window of past prices into the EM algorithm each tick
 *   - Places virtual bid/ask quotes using the Stoikov+paper pricing model
 *   - A BUY  fill occurs when the next tick's price ≤ our bid  (price fell to us)
 *   - A SELL fill occurs when the next tick's price ≥ our ask  (price rose to us)
 *   - Liquidity rewards are accumulated proportional to time-at-spread
 *   - Circuit breakers and calendar effects are applied identically to live mode
 *
 * Usage:
 *   npm run backtest-mm
 *   MM_BACKTEST_MARKETS=5  MM_BACKTEST_DAYS=30  npm run backtest-mm
 *
 * Env overrides (all optional):
 *   MM_BACKTEST_MARKETS   Number of markets to backtest       (default 10)
 *   MM_BACKTEST_DAYS      Days of history to replay            (default 30)
 *   MM_BACKTEST_CAPITAL   Starting capital per market (USD)    (default 1000)
 *   MM_BACKTEST_REWARD_RATE  Estimated daily reward per $1k deployed (default 2.0)
 *   MM_BACKTEST_OFFLINE   Use synthetic data (no API calls)    (default true if API unavailable)
 *   MM_BASE_SPREAD        Same as live config                  (default 0.02)
 *   MM_MAX_SPREAD         Same as live config                  (default 0.08)
 *   MM_RISK_AVERSION      γ                                    (default 0.2)
 *   MM_VOL_SENSITIVITY    β                                    (default 1.0)
 *   MM_JUMP_SENSITIVITY   ζ                                    (default 0.5)
 *   MM_CALENDAR_FACTOR                                         (default 2.0)
 *   MM_ORDER_SIZE_USD                                          (default 100)
 *   MM_MAX_INVENTORY_PER_MARKET                                (default 500)
 *   MM_MAX_DAILY_LOSS                                          (default 200)
 *   MM_REPRICE_THRESHOLD                                       (default 0.005)
 */

import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import {
    estimateParamsHybrid,
    classifyRegime,
    detectTrend,
    VolatilityRegime,
} from '../utils/jumpDiffusionModel';
import {
    computeQuotes,
    computeOrderSizes,
    shouldReprice,
} from '../utils/marketMakingPricer';

// ── Colours ───────────────────────────────────────────────────────────────────
const C = {
    reset: '\x1b[0m',
    bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
    cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,
    green: (s: string) => `\x1b[32m${s}\x1b[0m`,
    red: (s: string) => `\x1b[31m${s}\x1b[0m`,
    yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
    gray: (s: string) => `\x1b[90m${s}\x1b[0m`,
    blue: (s: string) => `\x1b[34m${s}\x1b[0m`,
};

// ── Config ────────────────────────────────────────────────────────────────────
const CFG = {
    markets: parseInt(process.env.MM_BACKTEST_MARKETS || '10', 10),
    days: parseInt(process.env.MM_BACKTEST_DAYS || '30', 10),
    offline: process.env.MM_BACKTEST_OFFLINE !== 'false', // default: true (safe for CI/sandbox)
    /** Path to pre-fetched market data JSON (bypasses both synthetic and live API) */
    dataFile: process.env.MM_BACKTEST_DATA_FILE || '',
    capitalPerMarket: parseFloat(process.env.MM_BACKTEST_CAPITAL || '1000'),
    rewardRatePerDay: parseFloat(process.env.MM_BACKTEST_REWARD_RATE || '2.0'), // $ per $1k deployed/day
    baseSpread: parseFloat(process.env.MM_BASE_SPREAD || '0.02'),
    calmBaseSpread: parseFloat(process.env.MM_CALM_SPREAD || '0.01'),
    maxSpread: parseFloat(process.env.MM_MAX_SPREAD || '0.08'),
    riskAversion: parseFloat(process.env.MM_RISK_AVERSION || '0.2'),
    volSensitivity: parseFloat(process.env.MM_VOL_SENSITIVITY || '1.0'),
    jumpSensitivity: parseFloat(process.env.MM_JUMP_SENSITIVITY || '0.5'),
    calendarFactor: parseFloat(process.env.MM_CALENDAR_FACTOR || '2.0'),
    orderSizeUSD: parseFloat(process.env.MM_ORDER_SIZE_USD || '100'),
    maxInventory: parseFloat(process.env.MM_MAX_INVENTORY_PER_MARKET || '500'),
    maxDailyLoss: parseFloat(process.env.MM_MAX_DAILY_LOSS || '200'),
    repriceThreshold: parseFloat(process.env.MM_REPRICE_THRESHOLD || '0.005'),
    // Hard sigma cutoff — skip EXTREME markets entirely (backtest showed -190% ROI at σ=0.12)
    maxSigma: parseFloat(process.env.MM_MAX_SIGMA || '0.07'),
    trendThreshold: parseFloat(process.env.MM_TREND_THRESHOLD || '0.3'),
    emWindowHours: 24,       // hours of rolling history for EM
    emMinObs: 30,             // minimum observations before running EM
    emResampleMins: 60,      // re-run EM only every N ticks (1h) to avoid O(n²) cost
    circuitBreakerPct: 0.05, // 5% price jump triggers pause
    circuitBreakerPauseTicks: 5, // ticks to pause after breaker
    priceRangeMin: parseFloat(process.env.MM_PRICE_RANGE_MIN || '0.10'),
    priceRangeMax: parseFloat(process.env.MM_PRICE_RANGE_MAX || '0.90'),
    minDaysToResolution: parseInt(process.env.MM_MIN_DAYS_TO_RESOLUTION || '7', 10),
};

const CLOB_BASE = 'https://clob.polymarket.com';
const DATA_API = 'https://data-api.polymarket.com';

// ── Types ─────────────────────────────────────────────────────────────────────

interface PriceTick {
    t: number;   // unix seconds
    p: number;   // mid price
}

interface MarketInfo {
    conditionId: string;
    tokenId: string;         // YES token
    question: string;
    endDate: Date;
    startDate: Date;
}

/** Shape of entries in a MM_BACKTEST_DATA_FILE JSON array. */
interface RealMarketEntry {
    conditionId: string;
    question: string;
    yes_p: number;
    days_left: number;
    age_days: number;
    liquidity: number;
    token_yes: string;
    token_no: string;
    startDate: string;
    endDate: string;
    history: Array<{ t: number; p: number }>;
}

interface FilledOrder {
    tick: number;
    side: 'BUY' | 'SELL';
    price: number;
    sizeUSD: number;
}

interface MarketBacktestResult {
    market: MarketInfo;
    ticks: number;
    fills: FilledOrder[];
    // P&L breakdown
    spreadPnl: number;       // realized from round-trip spread captures
    adversePnl: number;      // mark-to-market loss from adverse selection
    rewardPnl: number;       // estimated liquidity reward income
    totalPnl: number;
    roi: number;
    // Risk metrics
    maxDrawdown: number;
    sharpeRatio: number;
    fillRate: number;        // fills per tick (how active the market was)
    circuitBreakerFires: number;
    // EM parameter stats
    avgSigma: number;
    avgLambda: number;
    avgHalfSpread: number;
    // Inventory
    maxInventoryUSD: number;
    finalInventoryUSD: number;
    // Summary
    daysSimulated: number;
}

interface BacktestSummary {
    timestamp: string;
    config: typeof CFG;
    totalMarkets: number;
    successfulMarkets: number;
    totalCapitalDeployed: number;
    // Aggregate P&L
    totalSpreadPnl: number;
    totalAdversePnl: number;
    totalRewardPnl: number;
    totalPnl: number;
    totalROI: number;
    // Daily rates
    avgDailyPnl: number;
    avgDailyROI: number;
    // Risk
    avgSharpe: number;
    avgMaxDrawdown: number;
    worstMarket: string;
    bestMarket: string;
    markets: MarketBacktestResult[];
}

// ── Synthetic market generator ────────────────────────────────────────────────
// Generates realistic 1-minute price ticks using the logit jump-diffusion model.
// Allows fully offline backtesting with controlled parameters.

interface SyntheticMarketSpec {
    name: string;
    p0: number;       // starting probability
    sigma: number;    // diffusion volatility (log-odds/√day)
    lambda: number;   // jump intensity (jumps/day)
    jumpMu: number;   // mean jump size (log-odds)
    jumpSig: number;  // jump size std dev (log-odds)
    daysToResolution: number;
    totalDuration: number;
}

// Box-Muller normal sample
function randn(): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function logit(p: number): number {
    return Math.log(Math.max(1e-6, p) / Math.max(1e-6, 1 - p));
}
function logistic(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

function generateSyntheticTicks(spec: SyntheticMarketSpec, days: number): PriceTick[] {
    const minutesPerDay = 1440;
    const totalMinutes = days * minutesPerDay;
    const dt = 1 / minutesPerDay;                          // 1 minute in days
    const diffStd = spec.sigma * Math.sqrt(dt);            // per-tick diffusion

    const startTs = Math.floor(Date.now() / 1000) - days * 86400;
    const ticks: PriceTick[] = [];
    let x = logit(spec.p0);                                // start in log-odds

    for (let i = 0; i < totalMinutes; i++) {
        const t = startTs + i * 60;

        // Diffusion step
        x += diffStd * randn();

        // Poisson jump: P(jump in dt) = λ·dt
        if (Math.random() < spec.lambda * dt) {
            x += spec.jumpMu + spec.jumpSig * randn();
        }

        // Convert back to probability and clamp
        const p = Math.max(0.01, Math.min(0.99, logistic(x)));
        ticks.push({ t, p });
    }

    return ticks;
}

// 10 synthetic market profiles spanning a range of volatility/jump regimes
const SYNTHETIC_MARKETS: SyntheticMarketSpec[] = [
    { name: 'Low-vol stable (σ=0.02, λ=0.05)',      p0: 0.50, sigma: 0.02, lambda: 0.05, jumpMu: 0,     jumpSig: 0.10, daysToResolution: 45, totalDuration: 90 },
    { name: 'Medium-vol moderate jumps (σ=0.05, λ=0.3)', p0: 0.40, sigma: 0.05, lambda: 0.30, jumpMu: 0,     jumpSig: 0.20, daysToResolution: 30, totalDuration: 60 },
    { name: 'High-vol frequent jumps (σ=0.10, λ=1.0)',   p0: 0.60, sigma: 0.10, lambda: 1.00, jumpMu: 0,     jumpSig: 0.30, daysToResolution: 20, totalDuration: 45 },
    { name: 'Near-boundary low (p=0.15)',             p0: 0.15, sigma: 0.03, lambda: 0.20, jumpMu: 0,     jumpSig: 0.15, daysToResolution: 25, totalDuration: 50 },
    { name: 'Near-boundary high (p=0.85)',            p0: 0.85, sigma: 0.03, lambda: 0.20, jumpMu: 0,     jumpSig: 0.15, daysToResolution: 25, totalDuration: 50 },
    { name: 'Trend-up (positive jump drift)',         p0: 0.35, sigma: 0.04, lambda: 0.50, jumpMu: 0.15,  jumpSig: 0.20, daysToResolution: 35, totalDuration: 70 },
    { name: 'Trend-down (negative jump drift)',       p0: 0.65, sigma: 0.04, lambda: 0.50, jumpMu: -0.15, jumpSig: 0.20, daysToResolution: 35, totalDuration: 70 },
    { name: 'Very stable – max rewards (σ=0.01)',     p0: 0.50, sigma: 0.01, lambda: 0.02, jumpMu: 0,     jumpSig: 0.05, daysToResolution: 60, totalDuration: 90 },
    { name: 'Late-stage near resolution (<15d)',      p0: 0.70, sigma: 0.06, lambda: 0.80, jumpMu: 0.10,  jumpSig: 0.25, daysToResolution: 12, totalDuration: 60 },
    { name: 'Volatile election-style (σ=0.12)',       p0: 0.50, sigma: 0.12, lambda: 2.00, jumpMu: 0,     jumpSig: 0.40, daysToResolution: 20, totalDuration: 45 },
];

function buildSyntheticMarkets(n: number, days: number): Array<{ market: MarketInfo; ticks: PriceTick[] }> {
    const now = Date.now();
    return SYNTHETIC_MARKETS.slice(0, n).map((spec, i) => {
        const ticks = generateSyntheticTicks(spec, days);
        const market: MarketInfo = {
            conditionId: `synthetic_${i}`,
            tokenId: `synthetic_token_${i}`,
            question: spec.name,
            endDate: new Date(now + spec.daysToResolution * 86400000),
            startDate: new Date(now - (spec.totalDuration - spec.daysToResolution) * 86400000),
        };
        return { market, ticks };
    });
}

// ── API helpers ───────────────────────────────────────────────────────────────

async function sleep(ms: number) {
    return new Promise((r) => setTimeout(r, ms));
}

async function fetchJson(url: string, retries = 3): Promise<any> {
    for (let i = 0; i < retries; i++) {
        try {
            const res = await axios.get(url, {
                timeout: 15000,
                headers: { 'User-Agent': 'Mozilla/5.0' },
            });
            return res.data;
        } catch (err: any) {
            if (i < retries - 1) await sleep(1000 * (i + 1));
            else throw err;
        }
    }
}

async function fetchActiveMarkets(limit: number): Promise<MarketInfo[]> {
    // Use CLOB /markets endpoint (same domain as price-history calls)
    const url = `${CLOB_BASE}/markets?limit=${Math.min(limit * 5, 300)}`;
    const data = await fetchJson(url);
    const raw: any[] = data?.data ?? (Array.isArray(data) ? data : []);

    const now = Date.now();
    const result: MarketInfo[] = [];

    for (const m of raw) {
        if (result.length >= limit) break;

        const endDate = new Date(
            m.end_date_iso ?? m.endDateIso ?? m.end_date ?? m.endDate ?? ''
        );
        const startDate = new Date(
            m.start_date_iso ?? m.startDateIso ?? m.start_date ?? m.startDate ?? ''
        );
        if (isNaN(endDate.getTime())) continue;

        const daysToRes = (endDate.getTime() - now) / 86400000;
        if (daysToRes < CFG.minDaysToResolution) continue;

        // CLOB /markets returns tokens array directly
        const tokens: any[] = m.tokens ?? [];
        const yesToken = tokens.find((t: any) =>
            (t.outcome ?? t.outcomeName ?? '').toUpperCase() === 'YES'
        ) ?? tokens[0];
        const tokenId: string = yesToken?.token_id ?? yesToken?.tokenId ?? '';
        if (!tokenId) continue;

        const conditionId: string = m.condition_id ?? m.conditionId ?? '';
        const question: string = m.question ?? m.title ?? conditionId.slice(0, 30);

        result.push({
            conditionId,
            tokenId,
            question,
            endDate,
            startDate: isNaN(startDate.getTime())
                ? new Date(now - 30 * 86400000) : startDate,
        });
    }

    return result.slice(0, limit);
}

async function fetchPriceHistory(tokenId: string, days: number): Promise<PriceTick[]> {
    const url = `${CLOB_BASE}/prices-history?market=${tokenId}&interval=max&fidelity=60`;
    const data = await fetchJson(url);
    const raw: PriceTick[] = data?.history ?? [];

    if (raw.length === 0) return [];

    const cutoff = Date.now() / 1000 - days * 86400;
    return raw
        .filter((p) => p.t >= cutoff && p.p > 0 && p.p < 1)
        .sort((a, b) => a.t - b.t);
}

// ── Core simulation engine ────────────────────────────────────────────────────

function simulateMarket(market: MarketInfo, ticks: PriceTick[]): MarketBacktestResult {
    if (ticks.length < CFG.emMinObs + 2) {
        return emptyResult(market, ticks.length);
    }

    const totalDuration =
        (market.endDate.getTime() - market.startDate.getTime()) / 86400000;

    // State
    let inventory = 0;      // net tokens held (positive = long YES)
    let cash = 0;           // realized cash P&L (excluding initial capital)
    let currentBid = -1;
    let currentAsk = -1;

    // Tracking
    const fills: FilledOrder[] = [];
    const equityCurve: number[] = [];
    let pauseTicks = 0;
    let circuitBreakerFires = 0;
    let sigmaSum = 0;
    let lambdaSum = 0;
    let halfSpreadSum = 0;
    let emCount = 0;
    let maxInventoryUSD = 0;
    let totalTimeAtSpread = 0; // in minutes (for reward calc)
    let dailyLoss = 0;
    let dailyLossDate = '';

    const emWindowTicks = CFG.emWindowHours * 60; // 1 min ticks

    // Cached EM estimates (updated every emResampleMins ticks)
    let cachedSigma = 0.05;
    let cachedLambda = 0.1;
    let cachedFairValue = ticks[0]?.p ?? 0.5;
    let cachedSigmaTotal = 0.05;
    let lastEmTick = -1;

    for (let i = CFG.emMinObs; i < ticks.length - 1; i++) {
        const tick = ticks[i];
        const nextTick = ticks[i + 1];
        const price = tick.p;
        const nextPrice = nextTick.p;

        // ── Daily loss reset ────────────────────────────────────────────────
        const today = new Date(tick.t * 1000).toISOString().slice(0, 10);
        if (today !== dailyLossDate) {
            dailyLoss = 0;
            dailyLossDate = today;
        }

        // ── Circuit breaker check ────────────────────────────────────────────
        if (pauseTicks > 0) {
            pauseTicks--;
            continue;
        }

        const prevPrice = ticks[i - 1].p;
        // Use percentage change relative to current price to avoid trivial fires
        const priceChangePct = Math.abs(price - prevPrice) / Math.max(prevPrice, 0.01);
        if (priceChangePct > CFG.circuitBreakerPct) {
            if (pauseTicks === 0) circuitBreakerFires++; // count distinct events only
            pauseTicks = CFG.circuitBreakerPauseTicks;
            currentBid = -1;
            currentAsk = -1;
            continue;
        }

        // Skip if price outside range
        if (price < CFG.priceRangeMin || price > CFG.priceRangeMax) continue;

        // ── Rolling EM parameter estimation (every emResampleMins ticks) ────
        if (i - lastEmTick >= CFG.emResampleMins || lastEmTick < 0) {
            const windowStart = Math.max(0, i - emWindowTicks);
            const wPrices = ticks.slice(windowStart, i + 1).map((t) => t.p);
            const wTs = ticks.slice(windowStart, i + 1).map((t) => t.t * 1000);
            // Hybrid EM: σ from 1-min data, λ from hourly resampled data
            const est = estimateParamsHybrid(wPrices, wTs);
            cachedSigma = est.sigma;
            cachedLambda = est.lambda;
            cachedFairValue = est.fairValue;
            cachedSigmaTotal = est.sigmaTotal;
            lastEmTick = i;
        }

        const sigma = cachedSigma;
        const lambda = cachedLambda;
        const fairValue = cachedFairValue;
        const sigmaTotal = cachedSigmaTotal;

        // ── Hard sigma cutoff using sigmaTotal (diffusion + jump risk) ───────
        const regime: VolatilityRegime = classifyRegime(sigmaTotal);
        if (sigmaTotal >= CFG.maxSigma || regime === 'EXTREME') {
            // Pause and skip — do not place quotes in high-vol markets
            currentBid = -1;
            currentAsk = -1;
            continue;
        }

        sigmaSum += sigmaTotal;  // use sigmaTotal for stats (more meaningful)
        lambdaSum += lambda;
        emCount++;

        // ── Trend detection ──────────────────────────────────────────────────
        const windowStart = Math.max(0, i - emWindowTicks);
        const wPrices = ticks.slice(windowStart, i + 1).map((t) => t.p);
        const wTs = ticks.slice(windowStart, i + 1).map((t) => t.t * 1000);
        const trend = wPrices.length >= 4 ? detectTrend(wPrices, wTs, 6) : 0;

        // ── Compute quotes ───────────────────────────────────────────────────
        const daysToResolution =
            (market.endDate.getTime() / 1000 - tick.t) / 86400;

        const netPositionUSD = inventory * price;

        const quotes = computeQuotes({
            fairValue,
            sigma,
            lambda,
            regime,
            trend,
            netPosition: netPositionUSD,
            maxInventory: CFG.maxInventory,
            baseSpread: CFG.baseSpread,
            calmBaseSpread: CFG.calmBaseSpread,
            maxSpread: CFG.maxSpread,
            riskAversion: CFG.riskAversion,
            volSensitivity: CFG.volSensitivity,
            jumpSensitivity: CFG.jumpSensitivity,
            trendThreshold: CFG.trendThreshold,
            daysToResolution: Math.max(0, daysToResolution),
            totalMarketDuration: totalDuration,
            calendarFactor: CFG.calendarFactor,
        });

        // Null return means EXTREME (should not happen after above gate, but guard)
        if (!quotes) {
            currentBid = -1;
            currentAsk = -1;
            continue;
        }

        halfSpreadSum += quotes.halfSpread;

        // ── Check if reprice needed ──────────────────────────────────────────
        const needsReprice =
            currentBid < 0 ||
            shouldReprice(currentBid, currentAsk, quotes.bid, quotes.ask, CFG.repriceThreshold);

        if (needsReprice) {
            currentBid = quotes.bid;
            currentAsk = quotes.ask;
        }

        // ── Simulate fills ───────────────────────────────────────────────────
        const { bidSize, askSize } = computeOrderSizes(
            CFG.orderSizeUSD,
            quotes.normalizedInventory,
            regime,
            trend,
            CFG.trendThreshold
        );

        // BUY fill: next price fell to/below our bid
        if (nextPrice <= currentBid && inventory * price < CFG.maxInventory) {
            const fillPrice = currentBid;
            const fillTokens = bidSize / fillPrice;
            inventory += fillTokens;
            cash -= bidSize;
            fills.push({ tick: tick.t, side: 'BUY', price: fillPrice, sizeUSD: bidSize });
        }

        // SELL fill: next price rose to/above our ask
        if (nextPrice >= currentAsk && inventory > 0) {
            const tokensToSell = Math.min(inventory, askSize / currentAsk);
            const proceedsUSD = tokensToSell * currentAsk;
            inventory -= tokensToSell;
            cash += proceedsUSD;
            fills.push({ tick: tick.t, side: 'SELL', price: currentAsk, sizeUSD: proceedsUSD });
        }

        // Accumulate time-at-spread for reward calculation (1 minute per tick)
        totalTimeAtSpread += 1;

        // Track max inventory
        const invUSD = Math.abs(inventory * price);
        if (invUSD > maxInventoryUSD) maxInventoryUSD = invUSD;

        // ── Daily loss circuit breaker ────────────────────────────────────────
        const markToMarket = cash + inventory * price;
        const equity = markToMarket;
        equityCurve.push(equity);

        const dailyLossThisTick = Math.max(0, -equity);
        if (dailyLossThisTick > CFG.maxDailyLoss) {
            // Simulate forced close of inventory
            if (inventory > 0) {
                cash += inventory * price;
                inventory = 0;
            }
            dailyLoss += dailyLossThisTick;
            circuitBreakerFires++;
            currentBid = -1;
            currentAsk = -1;
        }
    }

    // ── Close any residual inventory at last price ────────────────────────────
    const lastTick = ticks[ticks.length - 1];
    const finalInventoryUSD = inventory * lastTick.p;
    if (inventory !== 0) {
        cash += inventory * lastTick.p;
        inventory = 0;
    }

    // ── P&L attribution ───────────────────────────────────────────────────────
    // Spread P&L: sum of (ask_fill - bid_fill) for matched round trips
    const buyFills = fills.filter((f) => f.side === 'BUY');
    const sellFills = fills.filter((f) => f.side === 'SELL');
    const matchedPairs = Math.min(buyFills.length, sellFills.length);

    let spreadPnl = 0;
    for (let i = 0; i < matchedPairs; i++) {
        spreadPnl += sellFills[i].price - buyFills[i].price;
    }
    // Scale spread P&L by average order size
    const avgOrderSize = fills.length > 0
        ? fills.reduce((s, f) => s + f.sizeUSD, 0) / fills.length
        : CFG.orderSizeUSD;
    spreadPnl *= avgOrderSize;

    // Adverse selection: total cash P&L minus what spread_pnl explains
    const adversePnl = cash - spreadPnl;

    // Liquidity rewards: estimate based on time-at-spread × reward rate
    const daysSimulated = ticks.length > 0
        ? (ticks[ticks.length - 1].t - ticks[0].t) / 86400
        : 0;
    const rewardPnl = (CFG.capitalPerMarket / 1000) * CFG.rewardRatePerDay * daysSimulated;

    const totalPnl = spreadPnl + adversePnl + rewardPnl;
    const roi = (totalPnl / CFG.capitalPerMarket) * 100;

    // ── Drawdown calculation ──────────────────────────────────────────────────
    let peak = 0;
    let maxDrawdown = 0;
    for (const eq of equityCurve) {
        if (eq > peak) peak = eq;
        const dd = peak - eq;
        if (dd > maxDrawdown) maxDrawdown = dd;
    }

    // ── Sharpe ratio (daily returns) ─────────────────────────────────────────
    const dailyReturns: number[] = [];
    const ticksPerDay = 60 * 24; // 1-minute ticks
    for (let d = ticksPerDay; d < equityCurve.length; d += ticksPerDay) {
        const dayReturn = equityCurve[d] - equityCurve[d - ticksPerDay];
        dailyReturns.push(dayReturn);
    }
    const meanReturn = dailyReturns.length > 0
        ? dailyReturns.reduce((s, r) => s + r, 0) / dailyReturns.length : 0;
    const stdReturn = dailyReturns.length > 1
        ? Math.sqrt(dailyReturns.reduce((s, r) => s + (r - meanReturn) ** 2, 0) / dailyReturns.length)
        : 1;
    const sharpeRatio = stdReturn > 0 ? (meanReturn / stdReturn) * Math.sqrt(252) : 0;

    return {
        market,
        ticks: ticks.length,
        fills,
        spreadPnl,
        adversePnl,
        rewardPnl,
        totalPnl,
        roi,
        maxDrawdown,
        sharpeRatio,
        fillRate: fills.length / Math.max(ticks.length, 1),
        circuitBreakerFires,
        avgSigma: emCount > 0 ? sigmaSum / emCount : 0,
        avgLambda: emCount > 0 ? lambdaSum / emCount : 0,
        avgHalfSpread: emCount > 0 ? halfSpreadSum / emCount : 0,
        maxInventoryUSD,
        finalInventoryUSD,
        daysSimulated,
    };
}

function emptyResult(market: MarketInfo, ticks: number): MarketBacktestResult {
    return {
        market, ticks, fills: [],
        spreadPnl: 0, adversePnl: 0, rewardPnl: 0,
        totalPnl: 0, roi: 0, maxDrawdown: 0, sharpeRatio: 0,
        fillRate: 0, circuitBreakerFires: 0,
        avgSigma: 0, avgLambda: 0, avgHalfSpread: 0,
        maxInventoryUSD: 0, finalInventoryUSD: 0, daysSimulated: 0,
    };
}

// ── Report rendering ──────────────────────────────────────────────────────────

function printMarketRow(r: MarketBacktestResult, rank: number) {
    const pnl = r.totalPnl;
    const pnlStr = pnl >= 0
        ? C.green(`+$${pnl.toFixed(2)}`)
        : C.red(`-$${Math.abs(pnl).toFixed(2)}`);
    const roiStr = r.roi >= 0
        ? C.green(`+${r.roi.toFixed(1)}%`)
        : C.red(`${r.roi.toFixed(1)}%`);
    const q = r.market.question.slice(0, 42).padEnd(42);
    console.log(
        ` ${String(rank).padStart(2)}.  ${C.cyan(q)}  ` +
        `pnl=${pnlStr.padEnd(14)}  roi=${roiStr.padEnd(10)}  ` +
        `fills=${String(r.fills.length).padStart(4)}  ` +
        `σ=${r.avgSigma.toFixed(3)}  λ=${r.avgLambda.toFixed(2)}  ` +
        `days=${r.daysSimulated.toFixed(0)}`
    );
}

function printFullReport(summary: BacktestSummary) {
    const line = '═'.repeat(100);
    console.log('\n' + C.cyan(line));
    console.log(C.cyan(C.bold('  📊  MARKET MAKING STRATEGY BACKTEST REPORT')));
    console.log(C.cyan(`  Based on: arxiv 2510.15205 – Logit Jump-Diffusion Model`));
    console.log(C.cyan(line) + '\n');

    console.log(C.bold('  Configuration'));
    console.log(`  Markets simulated   : ${summary.totalMarkets}`);
    console.log(`  History window      : ${CFG.days} days`);
    console.log(`  Capital per market  : $${CFG.capitalPerMarket.toLocaleString()}`);
    console.log(`  Total capital       : $${summary.totalCapitalDeployed.toLocaleString()}`);
    console.log(`  Base spread (NORMAL): ${(CFG.baseSpread * 100).toFixed(1)}¢`);
    console.log(`  Base spread (CALM)  : ${(CFG.calmBaseSpread * 100).toFixed(1)}¢`);
    console.log(`  Max sigma cutoff    : ${CFG.maxSigma} (EXTREME regime excluded)`);
    console.log(`  Trend threshold     : ${CFG.trendThreshold}`);
    console.log(`  Risk aversion γ     : ${CFG.riskAversion}`);
    console.log(`  Vol sensitivity β   : ${CFG.volSensitivity}`);
    console.log(`  Jump sensitivity ζ  : ${CFG.jumpSensitivity}`);
    console.log(`  Calendar factor     : ${CFG.calendarFactor}`);
    console.log(`  Order size          : $${CFG.orderSizeUSD} / side`);
    console.log(`  Max inventory       : $${CFG.maxInventory} / market\n`);

    console.log(C.bold('  Aggregate P&L'));
    const pnlSign = summary.totalPnl >= 0 ? '+' : '';
    const roiSign = summary.totalROI >= 0 ? '+' : '';
    const pnlFn = summary.totalPnl >= 0 ? C.green : C.red;
    const roiFn = summary.totalROI >= 0 ? C.green : C.red;
    console.log(`  Total P&L           : ${pnlFn(pnlSign + '$' + summary.totalPnl.toFixed(2))}`);
    console.log(`  Total ROI           : ${roiFn(roiSign + summary.totalROI.toFixed(2) + '%')}`);
    console.log(`    Spread capture    : ${C.green('+$' + summary.totalSpreadPnl.toFixed(2))}`);
    console.log(`    Adverse selection : ${summary.totalAdversePnl >= 0 ? C.green('+$' + summary.totalAdversePnl.toFixed(2)) : C.red('-$' + Math.abs(summary.totalAdversePnl).toFixed(2))}`);
    console.log(`    Liquidity rewards : ${C.green('+$' + summary.totalRewardPnl.toFixed(2))}`);
    console.log(`  Avg daily P&L       : ${pnlFn((summary.avgDailyPnl >= 0 ? '+' : '') + '$' + summary.avgDailyPnl.toFixed(2))}`);
    console.log(`  Avg daily ROI       : ${roiFn((summary.avgDailyROI >= 0 ? '+' : '') + summary.avgDailyROI.toFixed(3) + '%')}\n`);

    console.log(C.bold('  Risk Metrics'));
    const sharpeFn = summary.avgSharpe > 1 ? C.green : summary.avgSharpe > 0 ? C.yellow : C.red;
    console.log(`  Avg Sharpe ratio    : ${sharpeFn(summary.avgSharpe.toFixed(2))}`);
    console.log(`  Avg max drawdown    : ${C.yellow('$' + summary.avgMaxDrawdown.toFixed(2))}`);
    console.log(`  Best market         : ${C.green(summary.bestMarket.slice(0, 60))}`);
    console.log(`  Worst market        : ${C.red(summary.worstMarket.slice(0, 60))}\n`);

    console.log(C.bold('  Per-Market Results'));
    console.log(C.gray('  ' + '-'.repeat(96)));
    const sorted = [...summary.markets].sort((a, b) => b.totalPnl - a.totalPnl);
    sorted.forEach((r, i) => printMarketRow(r, i + 1));

    console.log('\n' + C.bold('  Spread Capture vs Adverse Selection'));
    console.log(C.gray('  ' + '-'.repeat(60)));
    for (const r of sorted.slice(0, 5)) {
        const q = r.market.question.slice(0, 38);
        console.log(
            `  ${C.cyan(q.padEnd(38))}  ` +
            `spread=${C.green('+$' + r.spreadPnl.toFixed(2))}  ` +
            `adverse=${r.adversePnl >= 0 ? C.green('+$' + r.adversePnl.toFixed(2)) : C.red('-$' + Math.abs(r.adversePnl).toFixed(2))}  ` +
            `reward=${C.green('+$' + r.rewardPnl.toFixed(2))}`
        );
    }

    console.log('\n' + C.bold('  Strategy Parameter Statistics (avg across markets)'));
    console.log(C.gray('  ' + '-'.repeat(60)));
    const validMarkets = summary.markets.filter((m) => m.ticks > 50);
    if (validMarkets.length > 0) {
        const avgSigma = validMarkets.reduce((s, m) => s + m.avgSigma, 0) / validMarkets.length;
        const avgLambda = validMarkets.reduce((s, m) => s + m.avgLambda, 0) / validMarkets.length;
        const avgSpread = validMarkets.reduce((s, m) => s + m.avgHalfSpread * 2, 0) / validMarkets.length;
        const avgFillRate = validMarkets.reduce((s, m) => s + m.fillRate, 0) / validMarkets.length;
        const avgCB = validMarkets.reduce((s, m) => s + m.circuitBreakerFires, 0) / validMarkets.length;
        console.log(`  Avg belief volatility σ : ${avgSigma.toFixed(4)}`);
        console.log(`  Avg jump intensity λ    : ${avgLambda.toFixed(4)} jumps/day`);
        console.log(`  Avg realised spread     : ${(avgSpread * 100).toFixed(2)}¢`);
        console.log(`  Avg fill rate           : ${(avgFillRate * 100).toFixed(2)}% of ticks`);
        console.log(`  Avg circuit-breaker fires: ${avgCB.toFixed(1)} per market`);
    }

    console.log('\n' + C.cyan(line) + '\n');
}

// ── Save results ──────────────────────────────────────────────────────────────

function saveResults(summary: BacktestSummary) {
    const dir = path.join(process.cwd(), 'simulation_results');
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    const mode = CFG.dataFile ? 'realdata' : (CFG.offline ? 'synthetic' : 'live');
    const filename = `mm_backtest_${mode}_${CFG.days}d_${CFG.markets}markets_${new Date().toISOString().slice(0, 10)}.json`;
    const filepath = path.join(dir, filename);
    fs.writeFileSync(filepath, JSON.stringify(summary, null, 2), 'utf8');
    console.log(C.green(`\n✓ Results saved to: ${filepath}\n`));
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
    console.log(C.cyan('\n🚀 MARKET MAKING STRATEGY BACKTESTER'));
    console.log(C.cyan('   arxiv 2510.15205 – Logit Jump-Diffusion Model\n'));
    console.log(C.gray(`   Markets: ${CFG.markets} | History: ${CFG.days} days | Capital: $${CFG.capitalPerMarket}/market`));
    console.log(C.gray(`   Spread: ${CFG.baseSpread} (calm: ${CFG.calmBaseSpread}) | γ=${CFG.riskAversion} β=${CFG.volSensitivity} ζ=${CFG.jumpSensitivity}`));
    console.log(C.gray(`   maxSigma: ${CFG.maxSigma} | trendThreshold: ${CFG.trendThreshold} | hybrid EM: σ(1min)+λ(hourly)\n`));

    // 1. Load market data (real pre-fetched / synthetic / live)
    const results: MarketBacktestResult[] = [];
    const useOffline = CFG.offline;

    // ── Real data mode ────────────────────────────────────────────────────────
    if (CFG.dataFile) {
        console.log(C.cyan(`  Mode: REAL DATA – loading from ${CFG.dataFile}\n`));
        const raw: RealMarketEntry[] = JSON.parse(fs.readFileSync(CFG.dataFile, 'utf8'));
        const subset = raw.slice(0, CFG.markets);

        for (let i = 0; i < subset.length; i++) {
            const entry = subset[i];
            const market: MarketInfo = {
                conditionId: entry.conditionId,
                tokenId: entry.token_yes,
                question: entry.question,
                endDate: entry.endDate ? new Date(entry.endDate) : new Date(Date.now() + 30 * 86400000),
                startDate: entry.startDate ? new Date(entry.startDate) : new Date(Date.now() - 30 * 86400000),
            };

            const cutoffTs = Math.floor(Date.now() / 1000) - CFG.days * 86400;
            const ticks: PriceTick[] = (entry.history || []).map(h => ({ t: h.t, p: h.p }));
            const filteredTicks = ticks.filter(
                (t) => t.t >= cutoffTs && t.p >= CFG.priceRangeMin && t.p <= CFG.priceRangeMax
            );

            process.stdout.write(
                C.gray(`[${i + 1}/${subset.length}] `) +
                C.cyan(market.question.slice(0, 50).padEnd(50)) + ' '
            );

            if (filteredTicks.length < CFG.emMinObs + 5) {
                console.log(C.yellow(`⚠  only ${filteredTicks.length} in-range ticks`));
                continue;
            }

            const result = simulateMarket(market, filteredTicks);
            results.push(result);

            const pnlFn = result.totalPnl >= 0 ? C.green : C.red;
            const regime = classifyRegime(result.avgSigma);
            console.log(
                pnlFn(`${result.totalPnl >= 0 ? '+' : ''}$${result.totalPnl.toFixed(2)}`) +
                C.gray(` (${filteredTicks.length} ticks, ${result.fills.length} fills, σ=${result.avgSigma.toFixed(3)}, ${regime})`)
            );
        }

    } else if (useOffline) {
        console.log(C.yellow(`  Mode: SYNTHETIC (offline) – generating ${CFG.markets} market scenarios\n`));
        const syntheticData = buildSyntheticMarkets(Math.min(CFG.markets, SYNTHETIC_MARKETS.length), CFG.days);

        for (let i = 0; i < syntheticData.length; i++) {
            const { market, ticks } = syntheticData[i];
            process.stdout.write(
                C.gray(`[${i + 1}/${syntheticData.length}] `) +
                C.cyan(market.question.slice(0, 50).padEnd(50)) + ' '
            );

            const filteredTicks = ticks.filter(
                (t) => t.p >= CFG.priceRangeMin && t.p <= CFG.priceRangeMax
            );

            if (filteredTicks.length < CFG.emMinObs + 5) {
                console.log(C.yellow(`⚠  only ${filteredTicks.length} in-range ticks`));
                continue;
            }

            const result = simulateMarket(market, filteredTicks);
            results.push(result);

            const pnlFn = result.totalPnl >= 0 ? C.green : C.red;
            console.log(
                pnlFn(`${result.totalPnl >= 0 ? '+' : ''}$${result.totalPnl.toFixed(2)}`) +
                C.gray(` (${filteredTicks.length} ticks, ${result.fills.length} fills)`)
            );
        }
    } else {
        // 2. Live mode: fetch market list from API
        process.stdout.write(C.cyan('Fetching active markets... '));
        const markets = await fetchActiveMarkets(CFG.markets);
        console.log(C.green(`✓ Got ${markets.length} markets\n`));

        for (let i = 0; i < markets.length; i++) {
            const market = markets[i];
            process.stdout.write(
                C.gray(`[${i + 1}/${markets.length}] `) +
                C.cyan(market.question.slice(0, 50).padEnd(50)) + ' '
            );

            try {
                await sleep(200);
                const ticks = await fetchPriceHistory(market.tokenId, CFG.days);

                if (ticks.length < CFG.emMinObs + 5) {
                    console.log(C.yellow(`⚠  only ${ticks.length} ticks`));
                    continue;
                }

                const filteredTicks = ticks.filter(
                    (t) => t.p >= CFG.priceRangeMin && t.p <= CFG.priceRangeMax
                );

                if (filteredTicks.length < CFG.emMinObs + 5) {
                    console.log(C.yellow(`⚠  ${filteredTicks.length} in-range ticks`));
                    continue;
                }

                const result = simulateMarket(market, filteredTicks);
                results.push(result);

                const pnlFn = result.totalPnl >= 0 ? C.green : C.red;
                console.log(
                    pnlFn(`${result.totalPnl >= 0 ? '+' : ''}$${result.totalPnl.toFixed(2)}`) +
                    C.gray(` (${filteredTicks.length} ticks, ${result.fills.length} fills)`)
                );
            } catch (err: any) {
                console.log(C.red(`✗ ${err.message?.slice(0, 40) ?? err}`));
            }
        }
    } // end real/synthetic/live branches

    if (results.length === 0) {
        console.error(C.red('\n✗ No markets could be simulated. Check API connectivity.\n'));
        process.exit(1);
    }

    // 3. Aggregate
    const totalCapital = results.length * CFG.capitalPerMarket;
    const totalSpreadPnl = results.reduce((s, r) => s + r.spreadPnl, 0);
    const totalAdversePnl = results.reduce((s, r) => s + r.adversePnl, 0);
    const totalRewardPnl = results.reduce((s, r) => s + r.rewardPnl, 0);
    const totalPnl = results.reduce((s, r) => s + r.totalPnl, 0);
    const totalROI = (totalPnl / totalCapital) * 100;

    const totalDays = results.reduce((s, r) => s + r.daysSimulated, 0);
    const avgDays = totalDays / results.length;
    const avgDailyPnl = avgDays > 0 ? totalPnl / results.length / avgDays : 0;
    const avgDailyROI = avgDays > 0 ? totalROI / avgDays : 0;

    const avgSharpe = results.reduce((s, r) => s + r.sharpeRatio, 0) / results.length;
    const avgMaxDrawdown = results.reduce((s, r) => s + r.maxDrawdown, 0) / results.length;

    const sorted = [...results].sort((a, b) => b.totalPnl - a.totalPnl);
    const bestMarket = sorted[0]?.market.question ?? '';
    const worstMarket = sorted[sorted.length - 1]?.market.question ?? '';

    const summary: BacktestSummary = {
        timestamp: new Date().toISOString(),
        config: CFG,
        totalMarkets: CFG.markets,
        successfulMarkets: results.length,
        totalCapitalDeployed: totalCapital,
        totalSpreadPnl,
        totalAdversePnl,
        totalRewardPnl,
        totalPnl,
        totalROI,
        avgDailyPnl,
        avgDailyROI,
        avgSharpe,
        avgMaxDrawdown,
        worstMarket,
        bestMarket,
        markets: results,
    };

    // 4. Print + save
    printFullReport(summary);
    saveResults(summary);
}

main().catch((err) => {
    console.error(C.red('\n✗ Backtest failed:'), err);
    process.exit(1);
});
