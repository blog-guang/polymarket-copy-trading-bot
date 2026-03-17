import axios, { AxiosError } from 'axios';
import { ENV } from '../config/env';

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const isNetworkError = (error: unknown): boolean => {
    if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;
        const code = axiosError.code;
        return (
            code === 'ETIMEDOUT' ||
            code === 'ENETUNREACH' ||
            code === 'ECONNRESET' ||
            code === 'ECONNREFUSED' ||
            !axiosError.response
        );
    }
    return false;
};

/** Returns the Retry-After delay in ms from a 429 response, defaulting to 5s. */
function getRateLimitDelay(error: AxiosError): number {
    const retryAfter = error.response?.headers?.['retry-after'];
    if (retryAfter) {
        const parsed = parseFloat(String(retryAfter));
        if (!isNaN(parsed)) return parsed * 1000;
    }
    return 5_000; // default 5s
}

const fetchData = async (url: string) => {
    const retries = ENV.NETWORK_RETRY_LIMIT;
    const timeout = ENV.REQUEST_TIMEOUT_MS;
    const retryDelay = 1000;

    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            const response = await axios.get(url, {
                timeout,
                headers: {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                },
                family: 4,
            });
            return response.data;
        } catch (error) {
            const isLastAttempt = attempt === retries;

            // Handle HTTP 429 rate limit — retry with Retry-After header or 5s default
            if (axios.isAxiosError(error) && error.response?.status === 429) {
                if (!isLastAttempt) {
                    const delay = getRateLimitDelay(error as AxiosError);
                    console.warn(`⚠️  Rate limited (429) on ${url}, waiting ${delay / 1000}s...`);
                    await sleep(delay);
                    continue;
                }
                throw new Error(`Rate limit exceeded (429) after ${retries} attempts: ${url}`);
            }

            if (isNetworkError(error) && !isLastAttempt) {
                const delay = retryDelay * Math.pow(2, attempt - 1);
                console.warn(
                    `⚠️  Network error (attempt ${attempt}/${retries}), retrying in ${delay / 1000}s...`
                );
                await sleep(delay);
                continue;
            }

            if (isLastAttempt && isNetworkError(error)) {
                console.error(
                    `❌ Network timeout after ${retries} attempts -`,
                    axios.isAxiosError(error) ? error.code : 'Unknown error'
                );
            }
            throw error;
        }
    }
};

export default fetchData;
