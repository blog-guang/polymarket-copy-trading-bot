/**
 * Market Making Executor
 *
 * Main loop that:
 *   1. Loads the top-scored markets from DB
 *   2. For each market: fetches order book, computes fair value & quotes
 *   3. Cancels stale GTC orders and re-places refreshed bid/ask pairs
 *   4. Manages inventory, applies circuit breakers, tracks P&L
 *
 * Order flow:
 *   getOpenOrders() → cancel stale → createOrder(GTC, postOnly) × 2 (bid + ask)
 */

import { ClobClient, Side, OrderType } from '@polymarket/clob-client';
import { ENV } from '../config/env';
import Logger from '../utils/logger';
import fetchData from '../utils/fetchData';
import {
    estimateParamsHybrid,
    classifyRegime,
    detectTrend,
    defaultJumpDiffusionParams,
    VolatilityRegime,
} from '../utils/jumpDiffusionModel';
import {
    computeQuotes,
    computeOrderSizes,
    shouldReprice,
} from '../utils/marketMakingPricer';
import {
    MMMarketModel,
    MMOpenOrderModel,
    MMInventoryModel,
    MMPriceHistoryModel,
    IMMMarket,
    IMMInventory,
} from '../models/marketMakingState';

const MM_CFG = ENV.MARKET_MAKING;
const CLOB_BASE = ENV.CLOB_HTTP_URL.replace(/\/$/, '');

// ── Module state ──────────────────────────────────────────────────────────────

let executorTimer: ReturnType<typeof setTimeout> | null = null;
let running = false;

export function stopMarketMakingExecutor(): void {
    running = false;
    if (executorTimer) {
        clearTimeout(executorTimer);
        executorTimer = null;
    }
    Logger.info('[MM Executor] Stopped');
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function todayStr(): string {
    return new Date().toISOString().slice(0, 10);
}

/** Fetch mid-price from CLOB order book */
async function fetchMidPrice(tokenId: string): Promise<{ mid: number; bestBid: number; bestAsk: number } | null> {
    try {
        const url = `${CLOB_BASE}/book?token_id=${tokenId}`;
        const book = await fetchData(url);
        if (!book) return null;
        const bestBid = parseFloat(book.bids?.[0]?.price ?? '0');
        const bestAsk = parseFloat(book.asks?.[0]?.price ?? '1');
        if (bestBid > 0 && bestAsk < 1) {
            return { mid: (bestBid + bestAsk) / 2, bestBid, bestAsk };
        }
    } catch {
        // fall through
    }
    return null;
}

/** Load recent price history for EM estimation */
async function loadRecentPrices(conditionId: string): Promise<{ prices: number[]; timestamps: number[] }> {
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

/** Get or create inventory record atomically (no race condition on first call). */
async function getInventory(conditionId: string, tokenId: string): Promise<IMMInventory> {
    const inv = await MMInventoryModel.findOneAndUpdate(
        { conditionId },
        {
            $setOnInsert: {
                conditionId,
                tokenId,
                netPosition: 0,
                avgEntryPrice: 0,
                costBasis: 0,
                realizedPnl: 0,
                unrealizedPnl: 0,
                totalFillsBuy: 0,
                totalFillsSell: 0,
                lastUpdated: new Date(),
                dailyLoss: 0,
                dailyLossDate: todayStr(),
            },
        },
        { upsert: true, new: true }
    );
    return inv;
}

/** Update inventory after a fill */
async function recordFill(
    conditionId: string,
    side: 'BUY' | 'SELL',
    price: number,
    sizeUSD: number
): Promise<void> {
    const inv = await MMInventoryModel.findOne({ conditionId });
    if (!inv) return;

    const today = todayStr();
    if (inv.dailyLossDate !== today) {
        inv.dailyLoss = 0;
        inv.dailyLossDate = today;
    }

    const tokensTraded = sizeUSD / price;

    if (side === 'BUY') {
        const newCost = inv.costBasis + sizeUSD;
        const newPos = inv.netPosition + tokensTraded;
        inv.avgEntryPrice = newPos > 0 ? newCost / newPos : price;
        inv.netPosition = newPos;
        inv.costBasis = newCost;
        inv.totalFillsBuy += 1;
    } else {
        const entryCost = inv.avgEntryPrice * tokensTraded; // cost basis of tokens being sold
        const pnl = sizeUSD - entryCost;
        inv.realizedPnl += pnl;
        if (pnl < 0) inv.dailyLoss += Math.abs(pnl);
        inv.netPosition = Math.max(0, inv.netPosition - tokensTraded);
        inv.costBasis = Math.max(0, inv.costBasis - entryCost); // decrement cost basis for sold tokens
        inv.totalFillsSell += 1;
    }

    inv.lastUpdated = new Date();
    Logger.info(
        `[MM Inventory] ${conditionId.slice(0, 8)} ${side} fill $${sizeUSD.toFixed(2)} @ ${price.toFixed(3)} | ` +
        `netPos=${inv.netPosition.toFixed(3)} avgEntry=${inv.avgEntryPrice.toFixed(3)} realizedPnl=${inv.realizedPnl.toFixed(2)}`
    );
    await inv.save();
}

// ── Circuit breaker ───────────────────────────────────────────────────────────

async function checkCircuitBreaker(
    market: IMMMarket,
    inv: IMMInventory,
    currentPrice: number,
    previousPrice: number
): Promise<boolean> {
    // Already paused?
    if (inv.pausedUntil && new Date() < inv.pausedUntil) {
        Logger.warning(`[MM Executor] ${market.conditionId.slice(0, 8)}… paused until ${inv.pausedUntil.toISOString()}`);
        return true;
    }

    const today = todayStr();
    if (inv.dailyLossDate !== today) {
        inv.dailyLoss = 0;
        inv.dailyLossDate = today;
        await inv.save();
    }

    // Price jump > 5% in one tick
    const priceChange = Math.abs(currentPrice - previousPrice);
    if (priceChange > 0.05) {
        Logger.warning(`[MM Executor] Circuit breaker: price jump ${(priceChange * 100).toFixed(1)}% on ${market.question?.slice(0, 40)}`);
        inv.pausedUntil = new Date(Date.now() + 5 * 60 * 1000); // pause 5 min
        await inv.save();
        return true;
    }

    // Daily loss exceeded
    if (inv.dailyLoss >= MM_CFG.maxDailyLoss) {
        Logger.warning(`[MM Executor] Circuit breaker: daily loss $${inv.dailyLoss.toFixed(2)} on ${market.question?.slice(0, 40)}`);
        inv.pausedUntil = new Date(Date.now() + 24 * 60 * 60 * 1000); // pause 24h
        await inv.save();
        return true;
    }

    // < 1 day to resolution: stop making
    if (market.daysToResolution < 1) {
        Logger.info(`[MM Executor] Skipping ${market.question?.slice(0, 40)}: < 1 day to resolution`);
        return true;
    }

    return false;
}

// ── Order management ──────────────────────────────────────────────────────────

interface OpenOrder {
    id: string;
    asset_id?: string;
    side?: string;
    price?: string;
    size_matched?: string;
    original_size?: string;
    status?: string;
}

async function getOpenOrdersForMarket(
    clobClient: ClobClient,
    conditionId: string
): Promise<OpenOrder[]> {
    try {
        const result = await (clobClient as any).getOpenOrders({ market: conditionId });
        if (Array.isArray(result)) return result;
        if (result?.data && Array.isArray(result.data)) return result.data;
    } catch (err) {
        Logger.error(`[MM Executor] getOpenOrders failed: ${err}`);
    }
    return [];
}

async function cancelMarketOrders(
    clobClient: ClobClient,
    conditionId: string
): Promise<void> {
    try {
        // Capture partial fill state BEFORE cancelling so we account for
        // any size_matched on live orders that will disappear after cancel.
        const liveOrders = await getOpenOrdersForMarket(clobClient, conditionId);
        const partialMap = new Map<string, number>(); // orderId → tokens matched
        for (const o of liveOrders) {
            const matched = parseFloat(o.size_matched ?? '0');
            if (matched > 0) partialMap.set(o.id, matched);
        }

        await (clobClient as any).cancelMarketOrders(conditionId);

        // Record partial fills for any orders that had tokens matched
        if (partialMap.size > 0) {
            const dbOrders = await MMOpenOrderModel.find({
                orderId: { $in: [...partialMap.keys()] },
                status: 'OPEN',
            }).lean();
            await Promise.allSettled(
                dbOrders.map(async (dbOrd) => {
                    const matched = partialMap.get(dbOrd.orderId)!;
                    const fillUSD = matched * dbOrd.price;
                    await recordFill(dbOrd.conditionId, dbOrd.side as 'BUY' | 'SELL', dbOrd.price, fillUSD);
                    await MMOpenOrderModel.updateOne(
                        { orderId: dbOrd.orderId },
                        { $set: { status: 'PARTIALLY_FILLED', updatedAt: new Date() } }
                    );
                })
            );
        }

        // Mark remaining OPEN orders as CANCELLED
        await MMOpenOrderModel.updateMany(
            { conditionId, status: 'OPEN' },
            { $set: { status: 'CANCELLED', updatedAt: new Date() } }
        );
    } catch (err) {
        Logger.error(`[MM Executor] cancelMarketOrders failed: ${err}`);
    }
}

async function placeGTCOrder(
    clobClient: ClobClient,
    market: IMMMarket,
    side: 'BUY' | 'SELL',
    price: number,
    sizeUSD: number
): Promise<string | null> {
    try {
        const tokenId = market.tokenIdYes;
        // Size in tokens = USD / price
        const sizeTokens = parseFloat((sizeUSD / price).toFixed(2));
        if (sizeTokens < 1) return null; // too small

        const orderArgs = {
            tokenID: tokenId,
            side: side === 'BUY' ? Side.BUY : Side.SELL,
            price,
            size: sizeTokens,
        };

        const signedOrder = await clobClient.createOrder(orderArgs as any);
        const resp = await clobClient.postOrder(
            signedOrder,
            OrderType.GTC,
            MM_CFG.postOnly
        );

        if (resp?.success || resp?.orderID) {
            const orderId = resp.orderID ?? resp.id ?? `${Date.now()}-${side}`;
            // Upsert instead of create: safe against network-timeout double-placement
            await MMOpenOrderModel.updateOne(
                { orderId },
                {
                    $set: {
                        orderId, conditionId: market.conditionId, tokenId,
                        side, price, size: sizeTokens, sizeUSD,
                        status: 'OPEN', updatedAt: new Date(),
                    },
                    $setOnInsert: { createdAt: new Date() },
                },
                { upsert: true }
            );
            return orderId;
        }
    } catch (err) {
        // Differentiate fatal vs transient errors so callers can act appropriately
        const errMsg = String(err).toLowerCase();
        if (errMsg.includes('401') || errMsg.includes('unauthorized') || errMsg.includes('invalid signature')) {
            // Auth / signing error — won't recover without operator action
            Logger.error(`[MM Executor] FATAL: auth/signature error placing ${side} order — check CLOB credentials: ${err}`);
        } else if (errMsg.includes('insufficient') || errMsg.includes('balance')) {
            Logger.error(`[MM Executor] Insufficient balance placing ${side} $${sizeUSD}: ${err}`);
        } else {
            Logger.error(`[MM Executor] Place ${side} order failed: ${err}`);
        }
    }
    return null;
}

// ── Per-market quote update ───────────────────────────────────────────────────

async function updateQuotesForMarket(
    clobClient: ClobClient,
    market: IMMMarket
): Promise<void> {
    // 1. Fetch live order book
    const book = await fetchMidPrice(market.tokenIdYes);
    if (!book) return;

    const { mid: currentPrice } = book;

    // Market resolution guard: price collapsing to 0 or 1 signals imminent settlement.
    // Cancel all orders and deactivate the market rather than quoting into a resolved event.
    const RESOLUTION_THRESHOLD = 0.02;
    if (currentPrice <= RESOLUTION_THRESHOLD || currentPrice >= 1 - RESOLUTION_THRESHOLD) {
        Logger.info(
            `[MM Executor] Market near resolution (price=${currentPrice.toFixed(3)}): ${market.question?.slice(0, 40)} — cancelling and deactivating`
        );
        await cancelMarketOrders(clobClient, market.conditionId);
        await MMMarketModel.updateOne(
            { conditionId: market.conditionId },
            { $set: { active: false } }
        );
        return;
    }

    // 2. Load price history & run hybrid EM (σ from 1-min, λ from hourly)
    const { prices, timestamps } = await loadRecentPrices(market.conditionId);

    // Append current price to history snapshot (in-memory only for EM).
    // Cap at last 2000 points to bound memory allocation per tick.
    const BASE = Math.max(0, prices.length - 1999);
    const allPrices = [...prices.slice(BASE), currentPrice];
    const allTs = [...timestamps.slice(BASE), Date.now()];

    const emResult = allPrices.length >= 5
        ? estimateParamsHybrid(allPrices, allTs)
        : defaultJumpDiffusionParams(currentPrice, allPrices.length);
    const { sigma, lambda, sigmaTotal, fairValue } = emResult;

    // Classify regime and apply hard σ cutoff (using sigmaTotal: includes jump risk)
    const regime: VolatilityRegime = classifyRegime(sigmaTotal);
    if (regime === 'EXTREME' || sigmaTotal >= MM_CFG.maxSigma) {
        Logger.warning(
            `[MM Executor] Skipping ${market.question?.slice(0, 40)} — σ_total=${sigmaTotal.toFixed(3)} (EXTREME)`
        );
        await cancelMarketOrders(clobClient, market.conditionId);
        return;
    }

    // Detect trend
    const trend = allPrices.length >= 4
        ? detectTrend(allPrices, allTs, 6)
        : (market.trend ?? 0);

    // 3. Get inventory
    const inv = await getInventory(market.conditionId, market.tokenIdYes);

    // Update unrealized P&L every cycle: netPosition × currentPrice − costBasis
    if (inv.netPosition !== 0) {
        const unrealizedPnl = inv.netPosition * currentPrice - inv.costBasis;
        await MMInventoryModel.updateOne(
            { conditionId: market.conditionId },
            { $set: { unrealizedPnl, lastUpdated: new Date() } }
        );
        (inv as any).unrealizedPnl = unrealizedPnl; // keep in-memory value consistent
    }

    // 4. Circuit breaker
    const prevPrice = prices.length > 0 ? prices[prices.length - 1] : currentPrice;
    const shouldPause = await checkCircuitBreaker(market, inv, currentPrice, prevPrice);
    if (shouldPause) {
        await cancelMarketOrders(clobClient, market.conditionId);
        return;
    }

    // 5. Compute quotes (regime-aware)
    const totalDuration = market.marketAge + market.daysToResolution;
    const quotes = computeQuotes({
        fairValue,
        sigma,
        lambda,
        regime,
        trend,
        netPosition: inv.netPosition * inv.avgEntryPrice,
        maxInventory: MM_CFG.maxInventoryPerMarket,
        baseSpread: MM_CFG.baseSpread,
        calmBaseSpread: MM_CFG.calmBaseSpread,
        maxSpread: MM_CFG.maxSpread,
        riskAversion: MM_CFG.riskAversion,
        volSensitivity: MM_CFG.volSensitivity,
        jumpSensitivity: MM_CFG.jumpSensitivity,
        trendThreshold: MM_CFG.trendThreshold,
        daysToResolution: market.daysToResolution,
        totalMarketDuration: totalDuration > 0 ? totalDuration : 90,
        calendarFactor: MM_CFG.calendarFactor,
    });

    // Regime gate (redundant safety check — computeQuotes returns null for EXTREME)
    if (!quotes) {
        await cancelMarketOrders(clobClient, market.conditionId);
        return;
    }

    // 6. Check existing open orders
    const openOrders = await getOpenOrdersForMarket(clobClient, market.conditionId);
    const openBid = openOrders.find((o) => o.side?.toUpperCase() === 'BUY');
    const openAsk = openOrders.find((o) => o.side?.toUpperCase() === 'SELL');

    const needsReprice =
        openOrders.length === 0 ||
        (openBid && shouldReprice(
            parseFloat(openBid.price ?? '0'), parseFloat(openAsk?.price ?? '1'),
            quotes.bid, quotes.ask,
            MM_CFG.repriceThreshold
        ));

    if (!needsReprice) return;

    // 7. Cancel and re-place
    await cancelMarketOrders(clobClient, market.conditionId);

    const { bidSize, askSize } = computeOrderSizes(
        MM_CFG.orderSizeUSD,
        quotes.normalizedInventory,
        regime,
        trend,
        MM_CFG.trendThreshold
    );

    // ── Quote sanity guard ────────────────────────────────────────────────────
    if (quotes.bid >= quotes.ask) {
        Logger.warning(
            `[MM Executor] Spread collapsed (bid=${quotes.bid.toFixed(3)} >= ask=${quotes.ask.toFixed(3)}) on ${market.question?.slice(0, 30)}, skipping`
        );
        return;
    }
    if (quotes.bid < 0.01 || quotes.ask > 0.99) {
        Logger.warning(
            `[MM Executor] Quote out of valid range (bid=${quotes.bid.toFixed(3)}, ask=${quotes.ask.toFixed(3)}), skipping`
        );
        return;
    }
    if (bidSize < 1 || askSize < 1) {
        Logger.warning(`[MM Executor] Order size below $1 minimum on ${market.question?.slice(0, 30)}, skipping`);
        return;
    }

    // ── Total inventory ceiling ───────────────────────────────────────────────
    // Suppress new BUY orders when aggregate exposure across all markets hits the cap
    const allInv = await MMInventoryModel.find({}, { netPosition: 1, avgEntryPrice: 1 }).lean();
    const totalInventoryUSD = allInv.reduce(
        (sum, i) => sum + Math.abs(i.netPosition * i.avgEntryPrice),
        0
    );
    const atInventoryCeiling = totalInventoryUSD >= MM_CFG.maxTotalInventory;
    if (atInventoryCeiling) {
        Logger.warning(
            `[MM Executor] Total inventory $${totalInventoryUSD.toFixed(0)} >= limit $${MM_CFG.maxTotalInventory} — bids suppressed for ${market.question?.slice(0, 30)}`
        );
    }

    const bidId = atInventoryCeiling
        ? null
        : await placeGTCOrder(clobClient, market, 'BUY', quotes.bid, bidSize);
    const askId = await placeGTCOrder(clobClient, market, 'SELL', quotes.ask, askSize);

    Logger.info(
        `[MM] ${market.question?.slice(0, 28)}… | ` +
        `${regime} σ=${sigma.toFixed(3)} λ=${lambda.toFixed(2)} trend=${trend.toFixed(2)} | ` +
        `bid=${quotes.bid.toFixed(3)} ask=${quotes.ask.toFixed(3)} spread=${(quotes.halfSpread * 2).toFixed(3)} | ` +
        `inv=${quotes.normalizedInventory.toFixed(2)} | ` +
        `bidId=${bidId?.slice(0, 8) ?? 'FAIL'} askId=${askId?.slice(0, 8) ?? 'FAIL'}`
    );
}

// ── Fill reconciliation (sync DB status against live open orders) ─────────────

async function reconcileFilledOrders(clobClient: ClobClient): Promise<void> {
    try {
        const dbOpenOrders = await MMOpenOrderModel.find({ status: 'OPEN' }).lean();
        if (dbOpenOrders.length === 0) return;

        const liveOrders: OpenOrder[] = await (async () => {
            try {
                const res = await (clobClient as any).getOpenOrders({});
                return Array.isArray(res) ? res : (res?.data ?? []);
            } catch { return []; }
        })();

        const liveMap = new Map(liveOrders.map((o) => [o.id, o]));

        // (A) Orders no longer in live set → fully filled (or cancelled externally)
        const filledOrders = dbOpenOrders.filter((o) => !liveMap.has(o.orderId));

        // (B) Orders still live but with new tokens matched → incremental partial fill
        const partialOrders = dbOpenOrders.filter((o) => {
            const live = liveMap.get(o.orderId);
            if (!live) return false;
            const currentMatched = parseFloat(live.size_matched ?? '0');
            return currentMatched > (o.sizeMatchedTokens ?? 0);
        });

        // Process (A): fully filled orders
        if (filledOrders.length > 0) {
            await MMOpenOrderModel.bulkWrite(
                filledOrders.map((o) => ({
                    updateOne: {
                        filter: { orderId: o.orderId },
                        update: { $set: { status: 'FILLED', updatedAt: new Date() } },
                    },
                }))
            );
            // Record only the unmatched portion (remainder not already recorded via partial fills)
            await Promise.allSettled(
                filledOrders.map((o) => {
                    const alreadyMatched = o.sizeMatchedTokens ?? 0;
                    const remainingTokens = Math.max(0, o.size - alreadyMatched);
                    const fillUSD = remainingTokens * o.price;
                    return fillUSD > 0
                        ? recordFill(o.conditionId, o.side as 'BUY' | 'SELL', o.price, fillUSD)
                        : Promise.resolve();
                })
            );
            Logger.info(`[MM Executor] Reconciled ${filledOrders.length} fully filled order(s)`);
        }

        // Process (B): incremental partial fills for still-live orders
        if (partialOrders.length > 0) {
            await Promise.allSettled(
                partialOrders.map(async (o) => {
                    const live = liveMap.get(o.orderId)!;
                    const currentMatched = parseFloat(live.size_matched ?? '0');
                    const incrementTokens = currentMatched - (o.sizeMatchedTokens ?? 0);
                    if (incrementTokens <= 0) return;
                    const fillUSD = incrementTokens * o.price;
                    await recordFill(o.conditionId, o.side as 'BUY' | 'SELL', o.price, fillUSD);
                    await MMOpenOrderModel.updateOne(
                        { orderId: o.orderId },
                        { $set: { status: 'PARTIALLY_FILLED', sizeMatchedTokens: currentMatched, updatedAt: new Date() } }
                    );
                })
            );
            Logger.info(`[MM Executor] Recorded incremental fills for ${partialOrders.length} partial order(s)`);
        }
    } catch (err) {
        Logger.error(`[MM Executor] Reconcile error: ${err}`);
    }
}

// ── Main executor loop ────────────────────────────────────────────────────────

async function executorTick(clobClient: ClobClient): Promise<void> {
    // Reconcile fills first
    await reconcileFilledOrders(clobClient);

    // Load active markets sorted by score
    const markets = await MMMarketModel
        .find({ active: true })
        .sort({ score: -1 })
        .limit(MM_CFG.marketLimit)
        .lean() as unknown as IMMMarket[];

    if (markets.length === 0) {
        Logger.info('[MM Executor] No active markets yet – waiting for monitor scan...');
        return;
    }

    // Process each market sequentially to avoid CLOB rate limits
    for (const market of markets) {
        if (!running) break;
        try {
            await updateQuotesForMarket(clobClient, market);
        } catch (err) {
            Logger.error(`[MM Executor] Error on market ${market.conditionId.slice(0, 8)}: ${err}`);
        }
        // small inter-market delay
        await new Promise((res) => setTimeout(res, 300));
    }
}

// ── Service entry point ───────────────────────────────────────────────────────

/**
 * Cold-start cleanup: cancel every order that's recorded as OPEN in our DB
 * on CLOB.  This prevents stale quotes from previous sessions interfering with
 * fresh inventory accounting.  Called once before the first tick.
 */
async function cancelAllStaleOrders(clobClient: ClobClient): Promise<void> {
    const staleOrders = await MMOpenOrderModel.find({ status: 'OPEN' }).lean();
    if (staleOrders.length === 0) return;

    const staleByMarket = new Map<string, typeof staleOrders>();
    for (const o of staleOrders) {
        const list = staleByMarket.get(o.conditionId) ?? [];
        list.push(o);
        staleByMarket.set(o.conditionId, list);
    }

    let cancelled = 0;
    for (const [conditionId] of staleByMarket) {
        try {
            await (clobClient as any).cancelMarketOrders(conditionId);
            cancelled++;
        } catch (err) {
            Logger.warning(`[MM Executor] Could not cancel stale orders for ${conditionId.slice(0, 8)}: ${err}`);
        }
    }

    // Mark all as CANCELLED regardless — on restart we start fresh
    await MMOpenOrderModel.updateMany(
        { status: 'OPEN' },
        { $set: { status: 'CANCELLED', updatedAt: new Date() } }
    );

    Logger.info(`[MM Executor] Startup: cancelled stale orders across ${cancelled}/${staleByMarket.size} markets`);
}

export default async function marketMakingExecutor(clobClient: ClobClient): Promise<void> {
    running = true;
    Logger.info('[MM Executor] Service started');

    // Cancel any orders left open from a previous session before placing new quotes
    await cancelAllStaleOrders(clobClient);

    // Allow at most (rebalanceInterval - 5s) for each tick before hard timeout.
    // Prevents a stalled tick from blocking the entire executor indefinitely.
    const TICK_TIMEOUT_MS = Math.max(10_000, (MM_CFG.rebalanceIntervalSec - 5) * 1000);

    const tick = async () => {
        if (!running) return;
        const timeout = new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('Executor tick timed out')), TICK_TIMEOUT_MS)
        );
        try {
            await Promise.race([executorTick(clobClient), timeout]);
        } catch (err) {
            Logger.error(`[MM Executor] Tick error: ${err}`);
        } finally {
            if (running) {
                executorTimer = setTimeout(tick, MM_CFG.rebalanceIntervalSec * 1000);
            }
        }
    };

    await tick();
}
