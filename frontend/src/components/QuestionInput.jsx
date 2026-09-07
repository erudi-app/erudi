import React, { useState, useRef, useEffect } from "react";
import PropTypes from "prop-types";
import { ArrowRight, FileText, ImagePlus, Paperclip, Plus, X } from "lucide-react";
import { useTranslation } from "react-i18next";

const DEFAULT_MAX_IMAGES = 4;
// Raster formats the vision pipeline can actually decode. SVG and other vector
// or exotic types are rejected up front: the backend image decoder can't read
// them, so letting one through would fail the whole turn.
const SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"];
const IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"];
const MAX_IMAGE_BYTES = 20 * 1024 * 1024; // 20 MB per image

// Documents (#492): exactly the set the Knowledge Base parser handles, so a
// file accepted here is a file the backend's DocumentReader can read.
const SUPPORTED_DOCUMENT_EXTENSIONS = [".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md"];
const SUPPORTED_ACCEPT = [...SUPPORTED_IMAGE_TYPES, ...SUPPORTED_DOCUMENT_EXTENSIONS].join(",");
// Per-file gate mirroring the image one. 50 MB is far above any document these
// deterministic parsers are meant for and still bounds the parse time of a
// pathological file; what actually reaches the model is bounded separately by
// the backend's per-question character budget.
const MAX_DOCUMENT_BYTES = 50 * 1024 * 1024;
// Per-question file count. A dropped FOLDER counts as one entry here: the
// backend walks it and applies its own file-count cap.
const DEFAULT_MAX_ATTACHMENTS = 10;

const hasExtension = (name, extensions) => {
  const lower = (name || "").toLowerCase();
  return extensions.some((ext) => lower.endsWith(ext));
};

// Images keep their own pipeline (bytes + vision gating); everything else is a
// document candidate, judged on its extension.
const looksLikeImage = (file) =>
  file.type?.startsWith("image/") || hasExtension(file.name, IMAGE_EXTENSIONS);

export default function QuestionInput({
  placeholder,
  onSend,
  disabled = false,
  className = "",
  canAttachImages = true,
  maxImages = DEFAULT_MAX_IMAGES,
  maxAttachments = DEFAULT_MAX_ATTACHMENTS,
}) {
  const { t } = useTranslation();
  const effectivePlaceholder = placeholder ?? t("chat:composer.placeholder");
  const [value, setValue] = useState("");
  const [images, setImages] = useState([]);
  const [imagePaths, setImagePaths] = useState([]);
  // Attached documents and folders: {name, path}. Only the path travels.
  const [attachments, setAttachments] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [attachError, setAttachError] = useState("");
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  const canSend = !disabled && (value.trim() !== "" || images.length > 0 || attachments.length > 0);

  // Clipboard images have no source path, so persist their bytes to a real file
  // and use that path; otherwise they'd be stored as a bare [image] placeholder
  // and vanish on reload (#136). File-origin images already have a path.
  const persistPastedImage = async (dataUrl) => {
    if (!window.imageAPI?.savePasted) {
      return "";
    }
    try {
      return (await window.imageAPI.savePasted(dataUrl)) || "";
    } catch {
      return "";
    }
  };

  // Documents and folders (#492): collected as {name, path} and capped, with
  // the same reject-early discipline as images. Returns the error to show, or
  // "" when everything was accepted.
  const collectAttachments = (candidates) => {
    const accepted = [];
    let message = "";
    for (const { file, isFolder } of candidates) {
      if (!isFolder && !hasExtension(file.name, SUPPORTED_DOCUMENT_EXTENSIONS)) {
        message = t("chat:composer.errors.unsupportedDocument");
        continue;
      }
      if (!isFolder && file.size > MAX_DOCUMENT_BYTES) {
        message = t("chat:composer.errors.documentTooLarge");
        continue;
      }
      // The backend reads the file itself, on this machine: without a real path
      // there is nothing to attach (a pasted item with no file origin).
      const path = window.electron?.getFilePath?.(file) || "";
      if (!path) {
        message = t("chat:composer.errors.noFilePath");
        continue;
      }
      accepted.push({ name: file.name, path });
    }
    const remaining = Math.max(0, maxAttachments - attachments.length);
    if (accepted.length > remaining) {
      message = t("chat:composer.errors.tooManyFiles", { count: maxAttachments });
    }
    const toAdd = accepted.slice(0, remaining);
    if (toAdd.length) {
      setAttachments((prev) => [...prev, ...toAdd]);
    }
    return message;
  };

  const addFiles = (files, folderCandidates = []) => {
    const list = Array.from(files || []);
    const imageFiles = list.filter(looksLikeImage);
    const documentFiles = list.filter((file) => !looksLikeImage(file));

    let message = collectAttachments([
      ...documentFiles.map((file) => ({ file, isFolder: false })),
      ...folderCandidates.map((file) => ({ file, isFolder: true })),
    ]);

    // Images keep their own gate: a non-vision model never collects an image
    // the backend would just strip (#133). Documents are model-agnostic.
    if (!canAttachImages) {
      setAttachError(message);
      return;
    }
    // Validate before anything touches disk or the model: reject unsupported
    // formats (e.g. SVG) and oversized files instead of failing the turn later.
    const supported = [];
    for (const file of imageFiles) {
      if (!SUPPORTED_IMAGE_TYPES.includes(file.type)) {
        message = t("chat:composer.errors.unsupportedFormat");
      } else if (file.size > MAX_IMAGE_BYTES) {
        message = t("chat:composer.errors.tooLarge");
      } else {
        supported.push(file);
      }
    }
    // Cap at the per-model budget; tell the user if they picked more than fits.
    const remaining = Math.max(0, maxImages - images.length);
    const toAdd = supported.slice(0, remaining);
    if (supported.length > remaining) {
      message = t("chat:composer.errors.tooMany", { count: maxImages });
    }
    setAttachError(message);

    toAdd.forEach((file) => {
      const knownPath = window.electron?.getFilePath?.(file) || "";
      const reader = new FileReader();
      reader.onerror = () => setAttachError(t("chat:composer.errors.unreadable"));
      reader.onload = async () => {
        const result = reader.result;
        if (typeof result !== "string" || !result.startsWith("data:image/")) {
          setAttachError(t("chat:composer.errors.unreadable"));
          return;
        }
        // No source path (clipboard paste) -> persist to disk to obtain one.
        const filePath = knownPath || (await persistPastedImage(result));
        setImages((prev) => (prev.length >= maxImages ? prev : [...prev, result]));
        setImagePaths((prev) => (prev.length >= maxImages ? prev : [...prev, filePath]));
      };
      reader.readAsDataURL(file);
    });
  };

  const removeImage = (idx) => {
    setImages((prev) => prev.filter((_, i) => i !== idx));
    setImagePaths((prev) => prev.filter((_, i) => i !== idx));
    setAttachError("");
  };

  const removeAttachment = (idx) => {
    setAttachments((prev) => prev.filter((_, i) => i !== idx));
    setAttachError("");
  };

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed && images.length === 0 && attachments.length === 0) {
      return;
    }
    onSend?.(
      trimmed,
      images,
      imagePaths,
      attachments.map((a) => a.path)
    );
    setValue("");
    setImages([]);
    setImagePaths([]);
    setAttachments([]);
    resizeTextarea();
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handlePaste = (e) => {
    const items = e.clipboardData?.items;
    if (!items) {
      return;
    }
    const files = [];
    for (const item of items) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        const f = item.getAsFile();
        if (f) {
          files.push(f);
        }
      }
    }
    if (files.length) {
      e.preventDefault();
      addFiles(files);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (!e.dataTransfer?.files?.length) {
      return;
    }
    // A dropped FOLDER is only reliably identifiable through the entry API:
    // its File has no extension and no type, which no size or name check can
    // tell apart from an extension-less file. Where the API is missing the
    // folder falls through to the file lane and is rejected as unsupported --
    // honest, and never a silently empty attachment.
    const folders = [];
    const dropped = [];
    const items = e.dataTransfer.items ? Array.from(e.dataTransfer.items) : [];
    if (items.length) {
      for (const item of items) {
        if (item.kind !== "file") {
          continue;
        }
        const file = item.getAsFile?.();
        if (!file) {
          continue;
        }
        if (item.webkitGetAsEntry?.()?.isDirectory) {
          folders.push(file);
        } else {
          dropped.push(file);
        }
      }
      addFiles(dropped, folders);
      return;
    }
    addFiles(e.dataTransfer.files);
  };

  const resizeTextarea = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  };
  useEffect(() => {
    resizeTextarea();
  }, [value]);

  return (
    <div className={["relative w-full", className].join(" ")}>
      {/* Attached images live in their own glass panel (matching the chat
          header) above the composer; the text input yields beneath it. */}
      {(images.length > 0 || attachments.length > 0 || attachError) && (
        <div
          className={[
            "mb-2 w-full rounded-[20px] p-2.5",
            "border border-white/10",
            "bg-[rgba(22,40,36,0.45)] backdrop-blur-[18px] saturate-[1.4]",
            "shadow-[0_10px_30px_-6px_rgba(0,0,0,0.5)]",
          ].join(" ")}
        >
          {images.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {images.map((src, idx) => (
                <div key={idx} className="relative">
                  <img
                    src={src}
                    alt={t("chat:composer.attachmentAlt", { index: idx + 1 })}
                    className="h-20 w-20 object-cover rounded-xl border border-white/10"
                  />
                  <button
                    type="button"
                    onClick={() => removeImage(idx)}
                    aria-label={t("chat:composer.removeImage")}
                    className="absolute -top-2 -right-2 rounded-full bg-black/70 p-0.5 text-white/90 hover:text-white"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))}
              {/* Add-another tile: opens the same picker as the composer icon. */}
              {canAttachImages && images.length < maxImages && (
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={disabled}
                  aria-label={t("chat:composer.addImage")}
                  title={t("chat:composer.addAnotherImage")}
                  className="h-20 w-20 flex items-center justify-center rounded-xl border border-dashed border-white/25 text-white/60 hover:text-white hover:border-white/50 disabled:opacity-40 transition"
                >
                  <Plus className="h-6 w-6" />
                </button>
              )}
            </div>
          )}
          {/* Attached documents and folders: a chip per entry (#492). Only the
              name is shown -- the content is read by the backend, not here. */}
          {attachments.length > 0 && (
            <div className={`flex flex-wrap gap-2 ${images.length > 0 ? "mt-2" : ""}`}>
              {attachments.map((attachment, idx) => (
                <span
                  key={`${attachment.path}-${idx}`}
                  className="inline-flex items-center gap-1.5 max-w-full rounded-xl border border-white/10 bg-black/20 px-2 py-1 text-xs text-white/85"
                >
                  <FileText className="h-3.5 w-3.5 shrink-0 text-white/60" />
                  <span className="truncate max-w-[14rem]">{attachment.name}</span>
                  <button
                    type="button"
                    onClick={() => removeAttachment(idx)}
                    aria-label={t("chat:composer.removeFile")}
                    className="text-white/60 hover:text-white"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </span>
              ))}
            </div>
          )}
          {attachError && (
            <p className="mt-2 text-xs text-red-300/90" role="alert">
              {attachError}
            </p>
          )}
        </div>
      )}

      {/* Panel "glassy" with emerald-900 tint */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={[
          "relative flex items-center w-full rounded-[20px] overflow-hidden",
          "border",
          dragging ? "border-emerald-400/60" : "border-emerald-200/20",
          "bg-emerald-200/5 backdrop-blur-[10px] saturate-[1.3]",
          "shadow-[0_10px_30px_-6px_rgba(0,0,0,0.5),0_2px_6px_-1px_rgba(0,0,0,0.45)]",
        ].join(" ")}
      >
        {/* Frost overlays with emerald tint */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 rounded-[20px] mix-blend-overlay"
          style={{
            background:
              "linear-gradient(180deg, rgba(16,185,129,0.12) 0%, rgba(16,185,129,0.06) 28%, rgba(16,185,129,0.02) 60%, rgba(16,185,129,0) 100%)",
          }}
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 rounded-[20px]"
          style={{
            boxShadow: "inset 0 1px 0 rgba(16,185,129,0.15), inset 0 -1px 0 rgba(16,185,129,0.08)",
          }}
        />

        <input
          ref={fileInputRef}
          type="file"
          accept={canAttachImages ? SUPPORTED_ACCEPT : SUPPORTED_DOCUMENT_EXTENSIONS.join(",")}
          multiple
          className="hidden"
          onChange={(e) => {
            addFiles(e.target.files);
            e.target.value = "";
          }}
        />
        {/* Attach button. It opens the same picker either way; the icon and the
            label say what this model can take: images plus documents when the
            model has vision, documents alone otherwise (#492). */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={
            disabled || (images.length >= maxImages && attachments.length >= maxAttachments)
          }
          className="pl-3 md:pl-4 text-white/70 hover:text-white disabled:opacity-40 transition"
          aria-label={
            canAttachImages ? t("chat:composer.attachImage") : t("chat:composer.attachFile")
          }
          title={
            canAttachImages ? t("chat:composer.attachImageHint") : t("chat:composer.attachFileHint")
          }
        >
          {canAttachImages ? <ImagePlus className="w-5 h-5" /> : <Paperclip className="w-5 h-5" />}
        </button>

        <textarea
          ref={textareaRef}
          rows={1}
          placeholder={effectivePlaceholder}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          disabled={disabled}
          style={{ maxHeight: "160px" }}
          className={[
            "w-full bg-transparent px-3 md:px-4 py-3 md:py-4",
            "border-0 text-gray-100 placeholder-gray-300",
            "focus:outline-none focus:ring-0 focus:shadow-none",
            "disabled:opacity-50 resize-none overflow-y-auto",
            "text-[0.95rem] md:text-[1rem] leading-6",
          ].join(" ")}
        />

        <div className="pr-0 md:pr-2 flex items-center">
          <button
            onClick={handleSend}
            disabled={!canSend}
            className={[
              "inline-flex items-center justify-center",
              "p-2",
              "text-white/70 hover:text-white disabled:opacity-50 transition",
            ].join(" ")}
            aria-label={t("common:actions.send")}
          >
            <ArrowRight className="w-6 h-6" />
          </button>
        </div>
      </div>
    </div>
  );
}

QuestionInput.propTypes = {
  placeholder: PropTypes.string,
  onSend: PropTypes.func.isRequired,
  disabled: PropTypes.bool,
  className: PropTypes.string,
  canAttachImages: PropTypes.bool,
  maxImages: PropTypes.number,
  maxAttachments: PropTypes.number,
};
