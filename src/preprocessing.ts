export type FeatureValue = number | string | boolean | null;
export type FeatureMatrix = readonly (readonly FeatureValue[])[];
export type ClassificationLabel = number | string | boolean;

export interface FitOptions {
  readonly categoricalFeatures?: readonly number[];
}

type Column =
  | { readonly kind: "numeric" }
  | { readonly kind: "categorical"; readonly categories: ReadonlyMap<ClassificationLabel, number> };

export class FeaturePreprocessor {
  readonly #columns: readonly Column[];

  private constructor(columns: readonly Column[]) {
    this.#columns = columns;
  }

  static fit(x: FeatureMatrix, categoricalFeatures: readonly number[] = []): FeaturePreprocessor {
    const features = validateRows(x, "xTrain");
    const explicit = new Set(categoricalFeatures);
    if (explicit.size !== categoricalFeatures.length) throw new Error("categorical feature indices must be unique");
    for (const index of explicit) {
      if (!Number.isInteger(index) || index < 0 || index >= features) {
        throw new Error(`categorical feature index ${index} is out of range`);
      }
    }

    const columns: Column[] = [];
    for (let columnIndex = 0; columnIndex < features; columnIndex += 1) {
      const categorical = explicit.has(columnIndex)
        || x.some((row) => typeof row[columnIndex] === "string" || typeof row[columnIndex] === "boolean");
      if (!categorical) {
        columns.push({ kind: "numeric" });
        continue;
      }
      const categories = new Map<ClassificationLabel, number>();
      for (let rowIndex = 0; rowIndex < x.length; rowIndex += 1) {
        const value = x[rowIndex]![columnIndex];
        if (value === undefined) throw new Error(`xTrain[${rowIndex}][${columnIndex}] is missing`);
        if (value === null || (typeof value === "number" && Number.isNaN(value))) continue;
        assertPrimitive(value, `xTrain[${rowIndex}][${columnIndex}]`);
        if (!categories.has(value)) categories.set(value, categories.size);
      }
      columns.push({ kind: "categorical", categories });
    }
    return new FeaturePreprocessor(columns);
  }

  transform(x: FeatureMatrix, name: string): number[][] {
    validateRows(x, name, this.#columns.length);
    return x.map((row, rowIndex) => row.map((value, columnIndex) => {
      const column = this.#columns[columnIndex]!;
      if (column.kind === "numeric") {
        if (value === null) return Number.NaN;
        if (typeof value !== "number") throw new Error(`${name}[${rowIndex}][${columnIndex}] is not numeric`);
        if (!Number.isFinite(value) && !Number.isNaN(value)) {
          throw new Error(`${name}[${rowIndex}][${columnIndex}] must not be infinite`);
        }
        return value;
      }
      if (value === null || (typeof value === "number" && Number.isNaN(value))) return Number.NaN;
      assertPrimitive(value, `${name}[${rowIndex}][${columnIndex}]`);
      return column.categories.get(value) ?? Number.NaN;
    }));
  }
}

export function encodeLabels(labels: readonly ClassificationLabel[]): {
  readonly classes: readonly ClassificationLabel[];
  readonly values: readonly number[];
} {
  const classes: ClassificationLabel[] = [];
  const indices = new Map<ClassificationLabel, number>();
  const values = labels.map((label, index) => {
    assertPrimitive(label, `yTrain[${index}]`);
    let encoded = indices.get(label);
    if (encoded === undefined) {
      encoded = classes.length;
      indices.set(label, encoded);
      classes.push(label);
    }
    return encoded;
  });
  if (classes.length < 2) throw new Error("classification requires at least two classes");
  return { classes, values };
}

export function validateRegressionTargets(targets: readonly number[]): number[] {
  return targets.map((target, index) => {
    if (!Number.isFinite(target)) throw new Error(`yTrain[${index}] must be finite`);
    return target;
  });
}

function validateRows(x: FeatureMatrix, name: string, expectedFeatures?: number): number {
  if (x.length === 0) throw new Error(`${name} must contain at least one row`);
  const features = expectedFeatures ?? x[0]!.length;
  if (features < 1) throw new Error(`${name} must contain at least one feature`);
  for (let rowIndex = 0; rowIndex < x.length; rowIndex += 1) {
    if (x[rowIndex]!.length !== features) {
      throw new Error(`${name} row ${rowIndex} has ${x[rowIndex]!.length} features; expected ${features}`);
    }
  }
  return features;
}

function assertPrimitive(value: unknown, name: string): asserts value is ClassificationLabel {
  if (typeof value !== "number" && typeof value !== "string" && typeof value !== "boolean") {
    throw new Error(`${name} must be a number, string, boolean, or null`);
  }
  if (typeof value === "number" && !Number.isFinite(value)) throw new Error(`${name} must be finite`);
}
