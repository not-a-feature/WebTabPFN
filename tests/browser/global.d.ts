declare global {
  interface Window {
    WebTabPFN: {
      clearCache: () => Promise<boolean>;
      isCached: (precision?: "fp32" | "int4" | "int8") => Promise<boolean>;
      load: (...args: unknown[]) => Promise<unknown>;
      preload: (options?: unknown) => Promise<{ fromCache: boolean }>;
    };
    runWebTabPFNBenchmark: (options?: unknown) => Promise<{
      configurations: Array<{
        status: string;
        cases?: Array<{ metrics: { accuracy: number }; timingMs: { count: number } }>;
      }>;
    }>;
  }
}

export {};
