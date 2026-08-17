(() => {
  const defaultMatrix = [
    { backend: "webgpu", precision: "fp32" },
    { backend: "webgpu", precision: "int4" },
    { backend: "webgpu", precision: "int8" },
    { backend: "wasm", precision: "fp32" },
    { backend: "wasm", precision: "int4" },
    { backend: "wasm", precision: "int8" },
  ];

  window.runWebTabPFNBenchmark = async (options = {}) => {
    const reference = await fetchJson(options.referenceUrl ?? "./reference.json");
    const configurations = options.configurations ?? defaultMatrix;
    const runs = options.runs ?? 5;
    const warmups = options.warmups ?? 1;
    const selectedCases = reference.cases.slice(0, options.caseLimit ?? reference.cases.length);
    const results = [];

    for (const configuration of configurations) {
      progress(options, `Starting ${configuration.backend}/${configuration.precision}`);
      if (options.clearCache !== false) await WebTabPFN.clearCache();
      try {
        const cold = await WebTabPFN.preload(configuration);
        const warm = await WebTabPFN.preload(configuration);
        progress(options, `Creating ${configuration.backend}/${configuration.precision} session`);
        const classifier = await WebTabPFN.load(configuration);
        const cases = [];

        for (const benchmarkCase of selectedCases) {
          progress(options, `Running ${configuration.backend}/${configuration.precision}: ${benchmarkCase.name}`);
          const split = benchmarkCase.yTrain.length;
          classifier.fit(benchmarkCase.x.slice(0, split), benchmarkCase.yTrain);
          const xTest = benchmarkCase.x.slice(split);
          for (let index = 0; index < warmups; index += 1) await classifier.infer(xTest);
          const timings = [];
          let result;
          for (let index = 0; index < runs; index += 1) {
            result = await classifier.infer(xTest);
            timings.push(result.inferenceMs);
          }
          if (result === undefined) throw new Error("benchmark produced no inference result");
          const metrics = classificationMetrics(benchmarkCase.yTest, result.probabilities);
          const nativeMetrics = {
            accuracy: benchmarkCase.python.metrics.accuracy,
            mcc: benchmarkCase.python.metrics.mcc ?? matthewsCorrelation(benchmarkCase.yTest, benchmarkCase.python.predictions),
          };
          cases.push({
            name: benchmarkCase.name,
            rows: benchmarkCase.x.length,
            trainRows: split,
            testRows: benchmarkCase.yTest.length,
            features: benchmarkCase.x[0].length,
            timingMs: timingSummary(timings),
            metrics,
            nativeMetrics,
            delta: {
              accuracy: metrics.accuracy - nativeMetrics.accuracy,
              mcc: metrics.mcc - nativeMetrics.mcc,
            },
            parity: parityMetrics(benchmarkCase.python.probabilities, result.probabilities),
          });
        }

        results.push({
          ...configuration,
          status: "ok",
          model: classifier.info.model,
          coldModelLoadMs: cold.modelLoadMs,
          cachedModelLoadMs: warm.modelLoadMs,
          sessionModelReadMs: classifier.info.modelLoadMs,
          sessionCreateMs: classifier.info.sessionCreateMs,
          cases,
        });
      } catch (error) {
        results.push({ ...configuration, status: "error", error: message(error) });
      }
    }

    return {
      schemaVersion: 2,
      createdAt: new Date().toISOString(),
      kind: "browser",
      environment: await browserEnvironment(),
      benchmark: { runs, warmups, cases: selectedCases.map((value) => value.name) },
      configurations: results,
    };
  };

  const runButton = document.getElementById("run");
  const status = document.getElementById("status");
  const output = document.getElementById("output");
  const download = document.getElementById("download");
  runButton.addEventListener("click", async () => {
    runButton.disabled = true;
    try {
      const query = new URLSearchParams(location.search);
      const matrix = query.get("matrix");
      const result = await window.runWebTabPFNBenchmark({
        referenceUrl: query.get("reference") ?? undefined,
        runs: query.has("runs") ? Number.parseInt(query.get("runs"), 10) : undefined,
        warmups: query.has("warmups") ? Number.parseInt(query.get("warmups"), 10) : undefined,
        caseLimit: query.has("caseLimit") ? Number.parseInt(query.get("caseLimit"), 10) : undefined,
        clearCache: query.get("clearCache") !== "false",
        configurations: matrix === null ? undefined : matrix.split(",").map((value) => {
          const [backend, precision] = value.split(":");
          return { backend, precision };
        }),
        onProgress: (value) => { status.textContent = value; },
      });
      const json = `${JSON.stringify(result, null, 2)}\n`;
      output.textContent = json;
      download.href = URL.createObjectURL(new Blob([json], { type: "application/json" }));
      download.download = "browser.json";
      download.hidden = false;
      status.textContent = "Benchmark complete";
    } catch (error) {
      status.textContent = message(error);
    } finally {
      runButton.disabled = false;
    }
  });

  function progress(options, value) {
    if (options.onProgress !== undefined) options.onProgress(value);
  }

  async function browserEnvironment() {
    let adapter = null;
    if ("gpu" in navigator) {
      const selected = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (selected !== null) {
        const info = selected.info;
        adapter = {
          vendor: info.vendor,
          architecture: info.architecture,
          device: info.device,
          description: info.description,
          isFallbackAdapter: info.isFallbackAdapter ?? null,
        };
      }
    }
    return {
      userAgent: navigator.userAgent,
      platform: navigator.platform,
      hardwareConcurrency: navigator.hardwareConcurrency,
      crossOriginIsolated,
      webgpuAvailable: "gpu" in navigator,
      adapter,
    };
  }

  async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`download failed: ${response.status} ${response.statusText}`);
    return response.json();
  }

  function classificationMetrics(yTrue, probabilities) {
    const predictions = probabilities.map(argmax);
    const correct = predictions.filter((value, index) => value === yTrue[index]).length;
    return { accuracy: correct / yTrue.length, mcc: matthewsCorrelation(yTrue, predictions) };
  }

  function matthewsCorrelation(yTrue, yPred) {
    if (yTrue.length === 0 || yTrue.length !== yPred.length) throw new Error("MCC inputs have incompatible lengths");
    const classes = Math.max(...yTrue, ...yPred) + 1;
    const matrix = Array.from({ length: classes }, () => Array(classes).fill(0));
    for (let index = 0; index < yTrue.length; index += 1) matrix[yTrue[index]][yPred[index]] += 1;
    const samples = yTrue.length;
    const correct = matrix.reduce((sum, row, index) => sum + row[index], 0);
    const actual = matrix.map((row) => row.reduce((sum, value) => sum + value, 0));
    const predicted = matrix.map((_, column) => matrix.reduce((sum, row) => sum + row[column], 0));
    const covariance = correct * samples - actual.reduce((sum, value, index) => sum + value * predicted[index], 0);
    const actualVariance = samples ** 2 - actual.reduce((sum, value) => sum + value ** 2, 0);
    const predictedVariance = samples ** 2 - predicted.reduce((sum, value) => sum + value ** 2, 0);
    const denominator = Math.sqrt(actualVariance * predictedVariance);
    return denominator === 0 ? 0 : covariance / denominator;
  }

  function parityMetrics(reference, candidate) {
    let absoluteSum = 0;
    let absoluteMaximum = 0;
    let count = 0;
    let agreements = 0;
    for (let row = 0; row < reference.length; row += 1) {
      if (argmax(reference[row]) === argmax(candidate[row])) agreements += 1;
      for (let column = 0; column < reference[row].length; column += 1) {
        const difference = Math.abs(reference[row][column] - candidate[row][column]);
        absoluteSum += difference;
        absoluteMaximum = Math.max(absoluteMaximum, difference);
        count += 1;
      }
    }
    return {
      predictionAgreement: agreements / reference.length,
      probabilityMae: absoluteSum / count,
      probabilityMaxAbs: absoluteMaximum,
    };
  }

  function timingSummary(values) {
    const sorted = [...values].sort((left, right) => left - right);
    const mean = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;
    const variance = sorted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / sorted.length;
    return {
      runs: values,
      count: values.length,
      min: sorted[0],
      mean,
      std: Math.sqrt(variance),
      p50: percentile(sorted, 0.5),
      p75: percentile(sorted, 0.75),
      p90: percentile(sorted, 0.9),
      p95: percentile(sorted, 0.95),
      p99: percentile(sorted, 0.99),
      max: sorted[sorted.length - 1],
    };
  }

  function percentile(sorted, fraction) {
    const index = (sorted.length - 1) * fraction;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
  }

  function argmax(values) {
    let best = 0;
    for (let index = 1; index < values.length; index += 1) if (values[index] > values[best]) best = index;
    return best;
  }

  function message(error) {
    return error instanceof Error ? error.message : String(error);
  }
})();
