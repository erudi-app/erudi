import { describe, it, expect } from "vitest";
import {
  canAttachImages,
  maxImagesForModel,
  modelSupportsVision,
  parseParamSizeB,
  isVerySmallModel,
  SMALL_MODEL_PARAM_THRESHOLD_B,
  webSearchUnavailableForModel,
} from "./modelCapabilities";

describe("canAttachImages", () => {
  it("blocks attaching only when the model explicitly cannot see images", () => {
    expect(canAttachImages({ supports_vision: false })).toBe(false);
  });

  it("allows attaching for a vision model", () => {
    expect(canAttachImages({ supports_vision: true })).toBe(true);
  });

  it("is permissive when the capability is unknown", () => {
    // null/undefined/missing model -> never wrongly block a real VLM (#133).
    expect(canAttachImages({ supports_vision: null })).toBe(true);
    expect(canAttachImages({})).toBe(true);
    expect(canAttachImages(undefined)).toBe(true);
    expect(canAttachImages(null)).toBe(true);
  });
});

describe("maxImagesForModel", () => {
  it("falls back to the default cap when the param size is unknown or invalid", () => {
    expect(maxImagesForModel(undefined)).toBe(4);
    expect(maxImagesForModel({})).toBe(4);
    expect(maxImagesForModel({ param_size: null })).toBe(4);
    expect(maxImagesForModel({ param_size: "7B" })).toBe(4);
    expect(maxImagesForModel({ param_size: 0 })).toBe(4);
    expect(maxImagesForModel({ param_size: -1 })).toBe(4);
  });

  it("caps small models (<3B) at 2 images", () => {
    expect(maxImagesForModel({ param_size: 0.5 })).toBe(2);
    expect(maxImagesForModel({ param_size: 2.9 })).toBe(2);
  });

  it("caps mid-size models (3-8B) at 4 images", () => {
    expect(maxImagesForModel({ param_size: 3 })).toBe(4);
    expect(maxImagesForModel({ param_size: 7.9 })).toBe(4);
  });

  it("caps large models (>=8B) at 6 images", () => {
    expect(maxImagesForModel({ param_size: 8 })).toBe(6);
    expect(maxImagesForModel({ param_size: 70 })).toBe(6);
  });
});

describe("modelSupportsVision", () => {
  it("is a positive signal: false for a missing model", () => {
    expect(modelSupportsVision(null)).toBe(false);
    expect(modelSupportsVision(undefined)).toBe(false);
  });

  it("affirms vision when the engine detected the installed model as a VLM", () => {
    expect(modelSupportsVision({ supports_vision: true })).toBe(true);
  });

  it("affirms vision pre-download from the catalog vision category", () => {
    expect(modelSupportsVision({ supports_vision: null, category: "vision" })).toBe(true);
  });

  it("stays false when the capability is unknown and the category is not vision", () => {
    expect(modelSupportsVision({ supports_vision: null, category: "chat" })).toBe(false);
    expect(modelSupportsVision({})).toBe(false);
    expect(modelSupportsVision({ supports_vision: false, category: "chat" })).toBe(false);
  });
});

describe("parseParamSizeB", () => {
  it("passes a positive number through as billions", () => {
    expect(parseParamSizeB(0.6)).toBe(0.6);
    expect(parseParamSizeB(7)).toBe(7);
  });

  it("parses B-suffixed strings, with decimals and any casing/spacing", () => {
    expect(parseParamSizeB("0.6B")).toBe(0.6);
    expect(parseParamSizeB("1.7B")).toBe(1.7);
    expect(parseParamSizeB("7B")).toBe(7);
    expect(parseParamSizeB("7b")).toBe(7);
    expect(parseParamSizeB(" 4.0 B ")).toBe(4);
  });

  it("parses M-suffixed strings into fractional billions", () => {
    expect(parseParamSizeB("270M")).toBeCloseTo(0.27);
    expect(parseParamSizeB("500m")).toBeCloseTo(0.5);
  });

  it("returns null for unknown, empty, non-positive or malformed values", () => {
    expect(parseParamSizeB(undefined)).toBeNull();
    expect(parseParamSizeB(null)).toBeNull();
    expect(parseParamSizeB("")).toBeNull();
    expect(parseParamSizeB("Unknown")).toBeNull();
    expect(parseParamSizeB("B")).toBeNull();
    expect(parseParamSizeB("7GB")).toBeNull();
    expect(parseParamSizeB("large")).toBeNull();
    expect(parseParamSizeB(0)).toBeNull();
    expect(parseParamSizeB(-3)).toBeNull();
    expect(parseParamSizeB(NaN)).toBeNull();
    expect(parseParamSizeB({})).toBeNull();
  });
});

describe("isVerySmallModel (#381)", () => {
  it("exposes the ~4B threshold the QA guidance is based on", () => {
    expect(SMALL_MODEL_PARAM_THRESHOLD_B).toBe(4);
  });

  it("flags models strictly below the threshold from the numeric param_size", () => {
    expect(isVerySmallModel({ param_size: 0.6 })).toBe(true);
    expect(isVerySmallModel({ param_size: 1.7 })).toBe(true);
    expect(isVerySmallModel({ param_size: 3.99 })).toBe(true);
  });

  it("flags models from the parameters string when param_size is absent", () => {
    expect(isVerySmallModel({ parameters: "0.6B" })).toBe(true);
    expect(isVerySmallModel({ parameters: "1.7B" })).toBe(true);
    expect(isVerySmallModel({ parameters: "270M" })).toBe(true);
  });

  it("does not flag models at or above the threshold", () => {
    expect(isVerySmallModel({ param_size: 4 })).toBe(false);
    expect(isVerySmallModel({ param_size: 7 })).toBe(false);
    expect(isVerySmallModel({ parameters: "4B" })).toBe(false);
    expect(isVerySmallModel({ parameters: "7B" })).toBe(false);
    expect(isVerySmallModel({ parameters: "12.0B" })).toBe(false);
  });

  it("prefers the numeric param_size over the parameters string", () => {
    expect(isVerySmallModel({ param_size: 8, parameters: "0.6B" })).toBe(false);
    expect(isVerySmallModel({ param_size: 0.5, parameters: "7B" })).toBe(true);
  });

  it("never flags a model whose size is unknown", () => {
    expect(isVerySmallModel(undefined)).toBe(false);
    expect(isVerySmallModel(null)).toBe(false);
    expect(isVerySmallModel({})).toBe(false);
    expect(isVerySmallModel({ parameters: undefined })).toBe(false);
    expect(isVerySmallModel({ parameters: "Unknown" })).toBe(false);
    expect(isVerySmallModel({ param_size: null, parameters: null })).toBe(false);
  });
});

describe("webSearchUnavailableForModel (#570)", () => {
  it("disables when the model is positively known unable to execute tools", () => {
    expect(webSearchUnavailableForModel({ supports_tools: false })).toBe(true);
    expect(webSearchUnavailableForModel({ supports_tools_wire: false })).toBe(true);
    expect(
      webSearchUnavailableForModel({ supports_tools: false, supports_tools_wire: false })
    ).toBe(true);
  });

  it("stays enabled when the model can execute tools", () => {
    expect(webSearchUnavailableForModel({ supports_tools: true, supports_tools_wire: true })).toBe(
      false
    );
  });

  it("is permissive when a flag is unknown -- only positive knowledge disables it", () => {
    expect(webSearchUnavailableForModel({ supports_tools: null })).toBe(false);
    expect(webSearchUnavailableForModel({ supports_tools_wire: null })).toBe(false);
    expect(webSearchUnavailableForModel({})).toBe(false);
    expect(webSearchUnavailableForModel({ supports_tools: true, supports_tools_wire: null })).toBe(
      false
    );
  });

  it("never disables for a missing/unknown/orphaned model", () => {
    expect(webSearchUnavailableForModel(undefined)).toBe(false);
    expect(webSearchUnavailableForModel(null)).toBe(false);
  });
});
