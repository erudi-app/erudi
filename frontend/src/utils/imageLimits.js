// What the app accepts as an image, shared by the composer (QuestionInput) and
// the main process's image reader (utils/imageReader.js), so a file the
// composer attaches is a file the thread can show again later.

// Raster formats the vision pipeline can actually decode. SVG and other vector
// or exotic types are rejected up front: the backend image decoder can't read
// them, so letting one through would fail the whole turn.
export const SUPPORTED_IMAGE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/gif",
  "image/bmp",
];
export const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"];

// 20 MB per image. The vision models downscale what they get (Gemma 3 to
// 896 x 896, Qwen2.5-VL to about 12.8 megapixels at most), and a 20 MB PNG or
// JPEG is already past that, so the cap refuses nothing a model could use,
// while it keeps a stray multi-gigabyte file from being read whole into the
// main process and copied, base64-encoded, to the window.
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
