import mongoose from 'mongoose';
import { ENV } from './env';
import chalk from 'chalk';

const uri = ENV.MONGO_URI || 'mongodb://localhost:27017/polymarket_copytrading';

const connectDB = async () => {
    try {
        await mongoose.connect(uri, {
            serverSelectionTimeoutMS: 5000,
            socketTimeoutMS: 45000,
            connectTimeoutMS: 10000,
            retryWrites: true,
            retryReads: true,
            maxPoolSize: 5,
        });
        console.log(chalk.green('✓'), 'MongoDB connected');
    } catch (error) {
        console.log(chalk.red('✗'), 'MongoDB connection failed:', error);
        process.exit(1);
    }
};

/**
 * Close MongoDB connection gracefully
 */
export const closeDB = async (): Promise<void> => {
    try {
        await mongoose.connection.close();
        console.log(chalk.green('✓'), 'MongoDB connection closed');
    } catch (error) {
        console.log(chalk.red('✗'), 'Error closing MongoDB connection:', error);
    }
};

export default connectDB;
