import mongoose, { Schema, Document } from 'mongoose';

// ── Price History ────────────────────────────────────────────────────────────
// Stores 1-minute mid-price snapshots per market for the EM algorithm
export interface IMMPriceHistory extends Document {
    conditionId: string;
    tokenId: string;
    price: number;       // mid-price (0–1)
    timestamp: Date;
}

const MMPriceHistorySchema = new Schema<IMMPriceHistory>(
    {
        conditionId: { type: String, required: true, index: true },
        tokenId: { type: String, required: true, index: true },
        price: { type: Number, required: true },
        timestamp: { type: Date, required: true, index: true },
    },
    { collection: 'mm_price_history' }
);

// Unique index prevents duplicate ticks for the same (market, token, time) triple
MMPriceHistorySchema.index({ conditionId: 1, tokenId: 1, timestamp: 1 }, { unique: true });
// Sparse timestamp index for global pruning query
MMPriceHistorySchema.index({ timestamp: -1 });

export const MMPriceHistoryModel = mongoose.model<IMMPriceHistory>(
    'MMPriceHistory',
    MMPriceHistorySchema
);

// ── Open Orders ───────────────────────────────────────────────────────────────
// Tracks all GTC limit orders placed by the market making engine
export interface IMMOpenOrder extends Document {
    orderId: string;
    conditionId: string;
    tokenId: string;
    side: 'BUY' | 'SELL';
    price: number;
    size: number;             // in tokens
    sizeUSD: number;          // in USDC
    status: 'OPEN' | 'FILLED' | 'CANCELLED' | 'PARTIALLY_FILLED';
    createdAt: Date;
    updatedAt: Date;
}

const MMOpenOrderSchema = new Schema<IMMOpenOrder>(
    {
        orderId: { type: String, required: true, unique: true },
        conditionId: { type: String, required: true, index: true },
        tokenId: { type: String, required: true, index: true },
        side: { type: String, enum: ['BUY', 'SELL'], required: true },
        price: { type: Number, required: true },
        size: { type: Number, required: true },
        sizeUSD: { type: Number, required: true },
        status: {
            type: String,
            enum: ['OPEN', 'FILLED', 'CANCELLED', 'PARTIALLY_FILLED'],
            default: 'OPEN',
            index: true,
        },
        createdAt: { type: Date, default: Date.now },
        updatedAt: { type: Date, default: Date.now },
    },
    { collection: 'mm_open_orders' }
);

MMOpenOrderSchema.index({ conditionId: 1, status: 1 });

export const MMOpenOrderModel = mongoose.model<IMMOpenOrder>(
    'MMOpenOrder',
    MMOpenOrderSchema
);

// ── Inventory ─────────────────────────────────────────────────────────────────
// Tracks the bot's current net position per market
export interface IMMInventory extends Document {
    conditionId: string;
    tokenId: string;
    netPosition: number;       // tokens; positive = long YES, negative = short YES
    avgEntryPrice: number;
    costBasis: number;         // total USDC spent (or received) net
    realizedPnl: number;
    unrealizedPnl: number;
    totalFillsBuy: number;     // count
    totalFillsSell: number;
    lastUpdated: Date;
    // Circuit-breaker state
    pausedUntil?: Date;
    dailyLoss: number;
    dailyLossDate: string;     // YYYY-MM-DD
}

const MMInventorySchema = new Schema<IMMInventory>(
    {
        conditionId: { type: String, required: true, unique: true },
        tokenId: { type: String, required: true },
        netPosition: { type: Number, default: 0 },
        avgEntryPrice: { type: Number, default: 0 },
        costBasis: { type: Number, default: 0 },
        realizedPnl: { type: Number, default: 0 },
        unrealizedPnl: { type: Number, default: 0 },
        totalFillsBuy: { type: Number, default: 0 },
        totalFillsSell: { type: Number, default: 0 },
        lastUpdated: { type: Date, default: Date.now },
        pausedUntil: { type: Date },
        dailyLoss: { type: Number, default: 0 },
        dailyLossDate: { type: String, default: '' },
    },
    { collection: 'mm_inventory' }
);

export const MMInventoryModel = mongoose.model<IMMInventory>(
    'MMInventory',
    MMInventorySchema
);

// ── Selected Markets ──────────────────────────────────────────────────────────
// Cache of scored & selected markets for the current session
export interface IMMMarket extends Document {
    conditionId: string;
    tokenIdYes: string;       // YES outcome token ID
    tokenIdNo: string;        // NO outcome token ID
    question: string;
    endDate: Date;
    currentPrice: number;     // mid-price for YES token
    marketAge: number;        // days since market creation
    daysToResolution: number;
    rewardPool: number;       // estimated daily USDC reward pool
    score: number;            // risk-adjusted reward score
    sigma: number;            // belief volatility (hybrid EM estimate)
    lambda: number;           // jump intensity (hourly EM estimate)
    /** Volatility regime: CALM | NORMAL | VOLATILE | EXTREME */
    regime: string;
    /** Directional trend signal in [-1, +1] from linear regression on logit prices */
    trend: number;
    active: boolean;
    lastScored: Date;
}

const MMMarketSchema = new Schema<IMMMarket>(
    {
        conditionId: { type: String, required: true, unique: true },
        tokenIdYes: { type: String, required: true },
        tokenIdNo: { type: String, required: true },
        question: { type: String, default: '' },
        endDate: { type: Date, required: true },
        currentPrice: { type: Number, default: 0.5 },
        marketAge: { type: Number, default: 0 },
        daysToResolution: { type: Number, default: 30 },
        rewardPool: { type: Number, default: 0 },
        score: { type: Number, default: 0 },
        sigma: { type: Number, default: 0.01 },
        lambda: { type: Number, default: 0.01 },
        regime: { type: String, default: 'NORMAL', index: true },
        trend: { type: Number, default: 0 },
        active: { type: Boolean, default: true, index: true },
        lastScored: { type: Date, default: Date.now },
    },
    { collection: 'mm_markets' }
);

export const MMMarketModel = mongoose.model<IMMMarket>('MMMarket', MMMarketSchema);
