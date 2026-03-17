/**
 * Market Making Monitor
 *
 * Responsibilities:
 *   1. Periodically scan active Polymarket markets
 *   2. Filter markets through the two-layer selection framework
 *   3. Estimate EM parameters (σ, λ) from price history
 *   4. Score markets with risk-adjusted reward and update MM_MARKETS collection
 *   5. Persist rolling price history for the EM algorithm
 */

import { ENV } from '../config/env';
import fetchData from '../utils/fetchData';
import Logger from '../utils/logger';
import {
    estimateParamsHybrid,
    classifyRegime,
    detectTrend,
} from '../utils/jumpDiffusionModel';
import { riskAdjustedScore, estimateDailyReward } from '../utils/marketMakingPricer';
import {
    MMPriceHistoryModel,
    MMMarketModel,
    IMMMarket,
} from '../models/marketMakingState';

const MM_CFG = ENV.MARKET_MAKING;
const CLOB_BASE = ENV.CLOB_HTTP_URL.replace(/\/$/, '');
const DATA_API = 'https://data-api.polymarket.com';

// ── Type definitions for external API responses ───────────────────────────────

interface PolymarketMarket {
    condition_id: string;
    tokens: Array<{ token_id: string; outcome: string }>;
    question: string;
    end_date_iso?: string;
    end_date?: string;
    start_date_iso?: string;
    rewards?: { rates?: Array<{ dailyRewardRate?: number }> };
    volume?: number;
    liquidity?: number;
    active?: boolean;
    closed?: boolean;
    archived?: boolean;
}

interface PricePoint {
    t: number;   // unix seconds
    p: number;   // price
}

// ── Module state ──────────────────────────────────────────────────────────────

let monitorTimer: ReturnType<typeof setTimeout> | null = null;
let running = false;

export function stopMarketMakingMonitor(): void {
    running = false;
    if (monitorTimer) {
        clearTimeout(monitorTimer);
        monitorTimer = null;
    }
    Logger.info('[MM Monitor] Stopped');
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function daysAgo(dateStr: string | undefined): number {
    if (!dateStr) return 0;
    const ms = Date.now() - new Date(dateStr).getTime();
    return ms / (1000 * 60 * 60 * 24);
}

function daysUntil(dateStr: string | undefined): number {
    if (!dateStr) return 0;
    const ms = new Date(dateStr).getTime() - Date.now();
    return ms / (1000 * 60 * 60 * 24);
}

function extractRewardPool(market: PolymarketMarket): number {
    // Sum daily reward rates across all token pools if available
    if (!market.rewards?.rates) return 0;
    return market.rewards.rates.reduce((s, r) => s + (r.dailyRewardRate ?? 0), 0);
}

/** Fetch 1-minute price history from CLOB prices-history endpoint */
async function fetchPriceHistory(tokenId: string): Promise<PricePoint[]> {
    try {
        const url = `${CLOB_BASE}/prices-history?market=${tokenId}&interval=max&fidelity=60`;
        const data = await fetchData(url);
        if (data?.history && Array.isArray(data.history)) {
            return data.history as PricePoint[];
        }
    } catch {
        // silently return empty; monitor will use existing DB history
    }
    return [];
}

/** Save new price ticks to DB, pruning records older than the configured window */
async function savePriceHistory(
    conditionId: string,
    tokenId: string,
    points: PricePoint[]
): Promise<void> {
    if (points.length === 0) return;

    const cutoff = new Date(
        Date.now() - MM_CFG.priceHistoryWindowHours * 3_600_000
    );

    // Build upsert-style inserts (skip existing timestamps)
    const docs = points.map((p) => ({
        conditionId,
        tokenId,
        price: p.p,
        timestamp: new Date(p.t * 1000),
    }));

    // Insert only new points (ignore duplicates via try/catch on each)
    for (const doc of docs) {
        if (doc.timestamp > cutoff) {
            await MMPriceHistoryModel.updateOne(
                { conditionId, tokenId, timestamp: doc.timestamp },
                { $setOnInsert: doc },
                { upsert: true }
            ).catch(() => {/* ignore duplicate key */});
        }
    }

    // Prune old records
    await MMPriceHistoryModel.deleteMany({ conditionId, timestamp: { $lt: cutoff } });
}

/** Load price history from DB for EM estimation */
async function loadPriceHistory(
    conditionId: string
): Promise<{ prices: number[]; timestamps: number[] }> {
    const cutoff = new Date(
        Date.now() - MM_CFG.priceHistoryWindowHours * 3_600_000
    );

    const records = await MMPriceHistoryModel
        .find({ conditionId, timestamp: { $gte: cutoff } })
        .sort({ timestamp: 1 })
        .lean();

    return {
        prices: records.map((r) => r.price),
        timestamps: records.map((r) => new Date(r.timestamp).getTime()),
    };
}

// ── Market fetching and filtering ─────────────────────────────────────────────

async function fetchActiveMarkets(): Promise<PolymarketMarket[]> {
    const url = `${DATA_API}/markets?active=true&closed=false&limit=500`;
    try {
        const data = await fetchData(url);
        if (Array.isArray(data)) return data as PolymarketMarket[];
        if (data?.data && Array.isArray(data.data)) return data.data as PolymarketMarket[];
    } catch (err) {
        Logger.error(`[MM Monitor] Failed to fetch markets: ${err}`);
    }
    return [];
}

function passesFilter(market: PolymarketMarket, currentPrice: number): boolean {
    if (market.closed || market.archived) return false;
    if (!market.tokens || market.tokens.length < 2) return false;

    const endDate = market.end_date_iso || market.end_date;
    const startDate = market.start_date_iso;

    const daysToRes = daysUntil(endDate);
    const age = daysAgo(startDate);

    if (daysToRes < MM_CFG.minDaysToResolution) return false;
    if (age < MM_CFG.minMarketAgeDays) return false;
    if (currentPrice < MM_CFG.priceRangeMin || currentPrice > MM_CFG.priceRangeMax) return false;

    return true;
}

/** Fetch current mid-price for a YES token from the CLOB order book */
async function fetchMidPrice(tokenId: string): Promise<number | null> {
    try {
        const url = `${CLOB_BASE}/book?token_id=${tokenId}`;
        const book = await fetchData(url);
        if (!book) return null;
        const bestBid = parseFloat(book.bids?.[0]?.price ?? '0');
        const bestAsk = parseFloat(book.asks?.[0]?.price ?? '1');
        if (bestBid > 0 && bestAsk < 1) return (bestBid + bestAsk) / 2;
        if (bestBid > 0) return bestBid;
        if (bestAsk < 1) return bestAsk;
    } catch {
        // fall through
    }
    return null;
}

// ── Scoring and persisting selected markets ───────────────────────────────────

async function scoreAndSaveMarket(market: PolymarketMarket): Promise<number> {
    const yesToken = market.tokens.find(
        (t) => t.outcome?.toUpperCase() === 'YES'
    ) ?? market.tokens[0];
    const noToken = market.tokens.find(
        (t) => t.outcome?.toUpperCase() === 'NO'
    ) ?? market.tokens[1];

    // Fetch & persist price history
    const rawHistory = await fetchPriceHistory(yesToken.token_id);
    await savePriceHistory(market.condition_id, yesToken.token_id, rawHistory);

    // Load from DB (includes previously saved points)
    const { prices, timestamps } = await loadPriceHistory(market.condition_id);

    // Current mid-price
    let currentPrice = prices.length > 0 ? prices[prices.length - 1] : 0.5;
    const livePrice = await fetchMidPrice(yesToken.token_id);
    if (livePrice !== null) currentPrice = livePrice;

    // Filter check
    if (!passesFilter(market, currentPrice)) return 0;

    // EM parameter estimation (hybrid: σ from 1-min, λ from hourly resampled)
    const emResult = prices.length >= 5
        ? estimateParamsHybrid(prices, timestamps)
        : { sigma: 0.05, lambda: 0.1, sigmaTotal: 0.05, fairValue: currentPrice, muJump: 0, sigmaJump: 0.03, nObs: prices.length, converged: false };
    const { sigma, lambda, sigmaTotal, fairValue } = emResult;

    // Classify regime using sigmaTotal (includes jump risk, not just diffusion)
    const regime = classifyRegime(sigmaTotal);
    if (sigmaTotal >= MM_CFG.maxSigma) {
        // Mark as inactive (too volatile for market making)
        await MMMarketModel.updateOne(
            { conditionId: market.condition_id },
            { $set: { active: false, sigma, lambda, regime, lastScored: new Date() } },
            { upsert: false }
        ).catch(() => {});
        return 0;
    }

    // Trend detection (6-hour window in logit space)
    const trend = prices.length >= 4
        ? detectTrend(prices, timestamps, 6)
        : 0;

    // Reward estimation (use regime-appropriate base spread for halfSpread)
    const rewardPool = extractRewardPool(market);
    const totalLiquidity = market.liquidity ?? 1000;
    const deployedCapital = Math.min(MM_CFG.orderSizeUSD, MM_CFG.maxInventoryPerMarket);
    const regimeBaseSpread = regime === 'CALM' ? MM_CFG.calmBaseSpread
        : regime === 'VOLATILE' ? MM_CFG.baseSpread * 1.5
        : MM_CFG.baseSpread;
    const halfSpread = regimeBaseSpread / 2;
    const dailyReward = estimateDailyReward(deployedCapital, halfSpread, rewardPool, totalLiquidity);

    const score = riskAdjustedScore(dailyReward, sigma, lambda, regime);

    const endDate = market.end_date_iso || market.end_date;
    const startDate = market.start_date_iso;
    const totalDuration = startDate ? daysUntil(endDate) + daysAgo(startDate) : 90;

    await MMMarketModel.updateOne(
        { conditionId: market.condition_id },
        {
            $set: {
                conditionId: market.condition_id,
                tokenIdYes: yesToken.token_id,
                tokenIdNo: noToken?.token_id ?? '',
                question: market.question ?? '',
                endDate: endDate ? new Date(endDate) : new Date(Date.now() + 30 * 86400000),
                currentPrice: fairValue,
                marketAge: daysAgo(startDate),
                daysToResolution: daysUntil(endDate),
                rewardPool,
                score,
                sigma,
                lambda,
                regime,
                trend,
                active: true,
                lastScored: new Date(),
            } as Partial<IMMMarket>,
        },
        { upsert: true }
    );

    return score;
}

// ── Main scan loop ────────────────────────────────────────────────────────────

async function scanMarkets(): Promise<void> {
    Logger.info('[MM Monitor] Starting market scan...');

    const markets = await fetchActiveMarkets();
    if (markets.length === 0) {
        Logger.warning('[MM Monitor] No active markets returned from API');
        return;
    }

    Logger.info(`[MM Monitor] Evaluating ${markets.length} markets...`);

    // Score markets (process in small batches to avoid rate limiting)
    const BATCH = 10;
    const scored: Array<{ conditionId: string; score: number }> = [];

    for (let i = 0; i < markets.length; i += BATCH) {
        const batch = markets.slice(i, i + BATCH);
        const results = await Promise.allSettled(
            batch.map((m) => scoreAndSaveMarket(m))
        );
        results.forEach((r, j) => {
            if (r.status === 'fulfilled' && r.value > 0) {
                scored.push({ conditionId: batch[j].condition_id, score: r.value });
            }
        });
        // small delay between batches
        await new Promise((res) => setTimeout(res, 500));
    }

    // Mark only top MM_MARKET_LIMIT as active; deactivate others
    scored.sort((a, b) => b.score - a.score);
    const topIds = scored.slice(0, MM_CFG.marketLimit).map((m) => m.conditionId);

    await MMMarketModel.updateMany(
        { conditionId: { $nin: topIds } },
        { $set: { active: false } }
    );
    await MMMarketModel.updateMany(
        { conditionId: { $in: topIds } },
        { $set: { active: true } }
    );

    Logger.success(
        `[MM Monitor] Selected ${topIds.length} markets | top score: ${scored[0]?.score?.toFixed(4) ?? 'n/a'}`
    );
}

// ── Service entry point ───────────────────────────────────────────────────────

/** Scan interval in ms – run every 5 minutes by default */
const SCAN_INTERVAL_MS = 5 * 60 * 1000;

export default async function marketMakingMonitor(): Promise<void> {
    running = true;
    Logger.info('[MM Monitor] Service started');

    const tick = async () => {
        if (!running) return;
        try {
            await scanMarkets();
        } catch (err) {
            Logger.error(`[MM Monitor] Scan error: ${err}`);
        }
        if (running) {
            monitorTimer = setTimeout(tick, SCAN_INTERVAL_MS);
        }
    };

    // Initial scan immediately, then repeat
    await tick();
}
