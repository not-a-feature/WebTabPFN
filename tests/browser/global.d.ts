declare global {
  interface Window {
    WebTabPFN: {
      clearCache: () => Promise<boolean>;
      isCached: (options: { task: "classification" | "regression"; precision: "int4" | "int8" }) => Promise<boolean>;
      load: (...args: unknown[]) => Promise<unknown>;
      preload: (options: unknown) => Promise<{ fromCache: boolean }>;
    };
    runWebTabPFNBenchmark: (options?: unknown) => Promise<{
      configurations: Array<{
        precision: "fp32" | "int4" | "int8";
        status: string;
        cases?: Array<{
          metrics: { accuracy?: number; r2?: number };
          parity?: { predictionMae?: number; probabilityMae?: number };
          timingMs: { count: number };
        }>;
      }>;
    }>;
  }
}

export {};
