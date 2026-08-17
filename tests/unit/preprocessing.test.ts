import { describe, expect, it } from "vitest";

import { encodeLabels, FeaturePreprocessor } from "../../src/preprocessing";

describe("minimal preprocessing", () => {
  it("fits categorical mappings on training data only", () => {
    const preprocessor = FeaturePreprocessor.fit([[1, "red"], [2, "blue"], [null, "red"]]);

    expect(preprocessor.transform([[3, "blue"], [null, "green"]], "xTest")).toEqual([
      [3, 1],
      [Number.NaN, Number.NaN],
    ]);
  });

  it("supports explicitly categorical numeric columns", () => {
    const preprocessor = FeaturePreprocessor.fit([[10], [20]], [0]);

    expect(preprocessor.transform([[20], [30]], "xTest")).toEqual([[1], [Number.NaN]]);
  });

  it("encodes labels densely in first-observed order", () => {
    expect(encodeLabels(["case", "control", "case"])).toEqual({
      classes: ["case", "control"],
      values: [0, 1, 0],
    });
  });

  it("fails on malformed or infinite features", () => {
    expect(() => FeaturePreprocessor.fit([[1, 2], [3]])).toThrow(/expected 2/u);
    const preprocessor = FeaturePreprocessor.fit([[1], [2]]);
    expect(() => preprocessor.transform([[Number.POSITIVE_INFINITY]], "xTest")).toThrow(/infinite/u);
  });
});
