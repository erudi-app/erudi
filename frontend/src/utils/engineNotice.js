// Pure helpers for interpreting an inference-engine failure.
//
// Two sources feed the same descriptor, so one modal serves both:
//
//   - STARTUP. The backend's pre-flight found a GPU the bundled CUDA build
//     cannot drive and emitted `{event: "engine_notice", code, gpu_name,
//     compute_capability, driver_cuda_version, required_cuda_version, raw}` on
//     the lifecycle channel `main.js` forwards to the renderer. Nothing has
//     failed yet -- the first message would.
//   - RUNTIME. The inference child crashed on the first message and the chat
//     stream's error event carried `{t: "error", text, code, raw}`. Here the
//     NVML readings are unknown: only the code and the captured trace exist.
//
// The copy therefore comes in two grades per code: a version-free sentence
// that is always true, and a richer one used only when the numbers are there.
// Everything resolves through `i18n.t` at call time so the descriptor speaks
// the active language, exactly like `describeBackendError`.

import i18n from "../i18n";

// Backend engine-failure code -> `errors:engine.codes.*` key. The vocabulary is
// defined in `backend/src/engines/cuda_compatibility.py`.
const MESSAGE_KEYS = {
  CUDA_COMPUTE_CAPABILITY_TOO_LOW: "computeCapabilityTooLow",
  CUDA_DRIVER_TOO_OLD: "driverTooOld",
  CUDA_OUT_OF_MEMORY: "outOfMemory",
  CUDA_ERROR: "cudaError",
};

// An unrecognised code still opens the dialog: the trace and the report links
// are useful even when we cannot name the cause.
const FALLBACK_KEY = "cudaError";

function text(value) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

/** True when a forwarded backend lifecycle event is the startup engine notice. */
export function isEngineNotice(evt) {
  return !!evt && evt.event === "engine_notice" && !!evt.code;
}

/**
 * Turn a startup notice or a chat-stream error event into the descriptor the
 * failure dialog renders: {code, title, detail, hint, raw, gpuName,
 * computeCapability, driverCudaVersion, requiredCudaVersion}.
 *
 * Returns null when the event carries no engine code -- an ordinary failed
 * turn stays an ordinary red bubble and must not raise a dialog.
 */
export function describeEngineFailure(evt) {
  const code = evt && evt.code;
  if (!code) {
    return null;
  }
  const key = MESSAGE_KEYS[code] || FALLBACK_KEY;
  const gpuName = text(evt.gpu_name);
  const computeCapability = text(evt.compute_capability);
  const driverCudaVersion = text(evt.driver_cuda_version);
  const requiredCudaVersion = text(evt.required_cuda_version);

  // The richer sentence only fires when every value it interpolates exists;
  // otherwise the version-free one, which never renders an empty gap.
  const values = {
    gpu: gpuName || i18n.t("errors:engine.modal.yourGpu"),
    capability: computeCapability,
    driver: driverCudaVersion,
    required: requiredCudaVersion,
  };
  let detailKey = `errors:engine.codes.${key}.detail`;
  if (key === "computeCapabilityTooLow" && computeCapability) {
    detailKey = "errors:engine.codes.computeCapabilityTooLow.detailWithCapability";
  } else if (key === "driverTooOld" && driverCudaVersion && requiredCudaVersion) {
    detailKey = "errors:engine.codes.driverTooOld.detailWithVersions";
  }

  return {
    code,
    title: i18n.t(`errors:engine.codes.${key}.title`),
    detail: i18n.t(detailKey, values),
    hint: i18n.t(`errors:engine.codes.${key}.hint`),
    raw: text(evt.raw),
    gpuName,
    computeCapability,
    driverCudaVersion,
    requiredCudaVersion,
  };
}
