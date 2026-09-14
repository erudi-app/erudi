"""HuggingFace Model Metadata Utilities.

This module provides professional-grade utilities for fetching, parsing, and formatting
model metadata from HuggingFace Hub. Includes size estimation, parameter count extraction,
and structured metadata formatting with robust error handling.

Architecture:
    - Type-safe data structures (dataclasses) for model metadata
    - Enum-based categorization for model sizes and quantization types
    - Separation of concerns (fetch, parse, format)
    - Comprehensive error handling with custom exceptions
    - Configurable size mappings and parameter patterns

Key Features:
    - Calculate actual disk size for quantized models via HF API
    - Estimate sizes for known model families (Mistral, Gemma, Qwen, etc.)
    - Extract parameter counts from model names/links (7B, 1.5B, 350M, etc.)
    - Format ModelInfo objects into structured strings for storage
    - Type-safe, testable, and maintainable design

Data Structures:
    - ModelSize: Structured size representation with uncertainty bounds
    - ParameterCount: Typed parameter count with scale (billions/millions)
    - QuantizationType: Enum for quantization methods (4-bit, 8-bit, FP16, etc.)
    - ModelSizeCategory: Enum for model size ranges (tiny, small, medium, large, xlarge)

Functions:
    - get_disk_size_after_quant: Fetch actual size from HF API
    - get_parameter_count_from_name: Extract parameter count from naming
    - format_model_info_metadata: Format ModelInfo to structured string
    - parse_quantization_type: Detect quantization method from repo name
    - calculate_size_from_parameters: Estimate size from parameter count

Examples:
    >>> # Get actual size of MLX quantized model
    >>> from src.utils.hf_model_metadata import get_disk_size_after_quant
    >>>
    >>> size = get_disk_size_after_quant("mlx-community/Mistral-7B-v0.3-4bit")
    >>> print(size.to_string())  # "~3.2 GB"
    >>>
    >>> # Extract parameter count
    >>> from src.utils.hf_model_metadata import extract_parameter_pattern
    >>>
    >>> params = extract_parameter_pattern(
    ...     "Qwen 2.5 7B",
    ...     "Qwen/Qwen2.5-7B-Instruct"
    ... )
    >>> print(params.to_string() if params else "Unknown")  # "7B"

Dependencies:
    - huggingface_hub: get_hf_api() for repo info fetching
    - src.core.config: get_hf_api() lazy loader function
    - src.core.logging: Structured logging
    - dataclasses: Type-safe data structures
    - enum: Enumeration types

Notes:
    - Thread-safe and stateless functions
    - Graceful fallbacks when API fails
    - Regex-based parameter extraction (7b, 1.5b, 350m patterns)
    - Size estimates based on fp16/bf16 precision assumptions
    - Comprehensive logging for debugging and monitoring
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Union
from huggingface_hub import ModelInfo

from src.core.logging import logger
from src.core.config import get_hf_api


# ============ Enumerations ============


class QuantizationType(Enum):
    """Quantization methods for neural network models.

    Attributes:
        FP16: 16-bit floating point (full precision for most use cases).
        BF16: Brain float 16-bit (better range than FP16).
        INT8: 8-bit integer quantization (~50% compression).
        INT4: 4-bit integer quantization (~75% compression).
        GGUF: GGUF format (CPU-optimized, variable precision).
        UNKNOWN: Unknown or mixed precision quantization.
    """

    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "8bit"
    INT4 = "4bit"
    GGUF = "gguf"
    UNKNOWN = "unknown"


class ParameterScale(Enum):
    """Scale for parameter counts (billions or millions).

    Attributes:
        BILLION: Parameters in billions (e.g., 7B, 13B).
        MILLION: Parameters in millions (e.g., 350M, 125M).
    """

    BILLION = "B"
    MILLION = "M"


# Bytes per gigabyte for every MODEL ARTIFACT size we display (#316).
#
# Decimal (10^9), NOT binary (2^30). Model sizes are quoted decimal by Hugging
# Face, by the download progress and by `catalog_classify`, so measuring the
# same artifact in GiB while labelling it "GB" made a model appear to shrink by
# ~600 MB the moment it finished downloading (9.00 GB in the catalog, 8.4 GB
# once installed, for one unchanged 9,001,752,960 byte file).
#
# Deliberately NOT used for RAM/VRAM/disk-capacity readouts: those are reported
# in GiB by the OS and by every hardware vendor, and converting them here would
# make the machine readout disagree with Task Manager.
BYTES_PER_GB = 1_000_000_000

# ============ Data Structures ============


@dataclass(frozen=True)
class ModelSize:
    """Structured representation of model size with uncertainty.

    Attributes:
        size_gb: Size in gigabytes (central estimate).
        min_gb: Minimum size in GB (lower bound of estimate).
        max_gb: Maximum size in GB (upper bound of estimate).
        is_estimate: True if size is estimated, False if from API.
        source: Source of size information ("api", "estimate", "unknown").
        size_bytes: The exact byte count behind an API-measured size (the files
            the downloader fetches), or None for any estimate. Stored as-is on
            the catalog row (``llms.artifact_size_bytes``) so the UI shows the
            real download size instead of a per-parameter guess; an estimate
            must never be laundered into that column.

    Example:
        >>> size = ModelSize(size_gb=3.2, min_gb=3.0, max_gb=3.5, is_estimate=False, source="api")
        >>> print(size.to_string())  # "~3.2 GB"
        >>>
        >>> estimate = ModelSize(size_gb=13.5, min_gb=13.0, max_gb=14.0, is_estimate=True, source="estimate")
        >>> print(estimate.to_string())  # "~13.5 GB"
    """

    size_gb: float
    min_gb: Optional[float] = None
    max_gb: Optional[float] = None
    is_estimate: bool = True
    source: str = "estimate"
    size_bytes: Optional[int] = None

    def to_string(self) -> str:
        """Format size as human-readable string.

        Returns:
            String like "~3.2 GB" (precise) or "~3-4 GB" (range estimate).
        """
        if self.min_gb is not None and self.max_gb is not None and self.is_estimate:
            # Range estimate
            if self.min_gb == self.max_gb:
                return f"~{self.size_gb:.1f} GB"
            return f"~{self.min_gb:.1f}-{self.max_gb:.1f} GB"
        else:
            # Precise or single estimate
            return f"~{self.size_gb:.1f} GB"

    def __str__(self) -> str:
        return self.to_string()


@dataclass(frozen=True)
class ParameterCount:
    """Structured representation of model parameter count.

    Attributes:
        count: Numeric parameter count (e.g., 7.0 for 7B).
        scale: Scale of parameters (BILLION or MILLION).
        is_estimate: True if count is estimated, False if from metadata.

    Example:
        >>> params = ParameterCount(count=7.0, scale=ParameterScale.BILLION, is_estimate=False)
        >>> print(params.to_string())  # "7B"
        >>>
        >>> small = ParameterCount(count=350, scale=ParameterScale.MILLION, is_estimate=True)
        >>> print(small.to_string())  # "350M"
    """

    count: float
    scale: ParameterScale
    is_estimate: bool = True

    def to_string(self) -> str:
        """Format parameter count as human-readable string.

        Returns:
            String like "7B", "13B", "1.5B", "350M".
        """
        if self.scale == ParameterScale.BILLION:
            # Format with decimal for non-integer billions
            if self.count % 1 == 0:
                return f"{int(self.count)}{self.scale.value}"
            return f"{self.count:.1f}{self.scale.value}"
        else:
            # Millions are always integers
            return f"{int(self.count)}{self.scale.value}"

    def __str__(self) -> str:
        return self.to_string()

    @property
    def total_billions(self) -> float:
        """Get total parameter count in billions for comparison.

        Returns:
            Parameter count converted to billions (for sorting/comparison).
        """
        if self.scale == ParameterScale.BILLION:
            return self.count
        else:
            return self.count / 1000.0


# ============ Configuration ============

# Parameter size to disk size multipliers (bytes per parameter)
# Assumes fp16 (2 bytes/param) + overhead for embeddings, configs, etc.
SIZE_MULTIPLIERS: Dict[QuantizationType, float] = {
    QuantizationType.FP16: 2.0,  # 2 bytes per parameter
    QuantizationType.BF16: 2.0,  # 2 bytes per parameter
    QuantizationType.INT8: 1.0,  # 1 byte per parameter
    QuantizationType.INT4: 0.5,  # 0.5 bytes per parameter
    QuantizationType.GGUF: 1.0,  # Variable, default to 1 byte
    QuantizationType.UNKNOWN: 2.0,  # Default to fp16
}


# ============ Exception Classes ============


class HFMetadataError(Exception):
    """Base exception for HuggingFace metadata operations."""

    pass


class HFAPIError(HFMetadataError):
    """Exception raised when HuggingFace API calls fail."""

    pass


class ParameterExtractionError(HFMetadataError):
    """Exception raised when parameter count cannot be extracted."""

    pass


# ============ Helper Functions ============


def humanize_model_name(link: str) -> str:
    """Turn a HuggingFace repo id into an unambiguous, readable display name.

    Derived deterministically from the real slug — NEVER a hand-written label — so
    the family/version/size/variant are always preserved. This kills ambiguous
    labels like "Gemma-4B" that actually hid ``google/gemma-3-4b-it`` (Gemma 3).

    Examples:
        google/gemma-3-4b-it                -> "Gemma 3 4B Instruct"
        google/gemma-2-2b-it                -> "Gemma 2 2B Instruct"
        google/gemma-4-E2B-it               -> "Gemma 4 E2B Instruct"
        mistralai/Mistral-7B-Instruct-v0.3  -> "Mistral 7B Instruct v0.3"
        Qwen/Qwen2.5-VL-3B-Instruct         -> "Qwen2.5 VL 3B Instruct"
    """
    slug = link.split("/")[-1]
    out = []
    for i, tok in enumerate(slug.split("-")):
        low = tok.lower()
        if low in ("it", "instruct"):
            out.append("Instruct")
        elif re.fullmatch(r"\d+(?:\.\d+)?[bm]", low):  # 270m, 2b, 7b, 12b
            out.append(low[:-1].upper() + low[-1].upper())  # -> 270M, 2B, 12B
        elif re.fullmatch(r"[ea]\d+b", low):  # e2b, e4b, a4b
            out.append(low.upper())  # -> E2B, A4B
        elif re.fullmatch(r"v\d+(?:\.\d+)?", low):  # v0.3
            out.append(low)
        elif re.fullmatch(r"\d{4}", low):  # 2410, 2407 (date codes)
            out.append(low)
        elif re.fullmatch(r"\d+(?:\.\d+)?", low):  # 2, 3, 3.1, 2.5
            out.append(low)
        elif low == "vl":
            out.append("VL")
        else:  # family + misc tokens
            out.append(tok[:1].upper() + tok[1:])
    return " ".join(out)


def parse_quantization_type(repo_id: str) -> QuantizationType:
    """Detect quantization type from repository ID.

    Args:
        repo_id: HuggingFace repository ID (e.g., "mlx-community/Model-4bit").

    Returns:
        Detected QuantizationType enum value.

    Example:
        >>> quant = parse_quantization_type("mlx-community/Mistral-7B-v0.3-4bit")
        >>> print(quant)  # QuantizationType.INT4
    """
    repo_lower = repo_id.lower()

    if "4bit" in repo_lower or "4-bit" in repo_lower:
        return QuantizationType.INT4
    elif "8bit" in repo_lower or "8-bit" in repo_lower:
        return QuantizationType.INT8
    elif "gguf" in repo_lower:
        return QuantizationType.GGUF
    elif "bf16" in repo_lower:
        return QuantizationType.BF16
    elif "fp16" in repo_lower:
        return QuantizationType.FP16
    else:
        return QuantizationType.UNKNOWN


def calculate_size_from_parameters(
    param_count: ParameterCount, quant_type: QuantizationType = QuantizationType.FP16
) -> ModelSize:
    """Estimate model size from parameter count and quantization type.

    Args:
        param_count: Structured parameter count.
        quant_type: Quantization type (default FP16).

    Returns:
        Estimated ModelSize with uncertainty bounds.

    Example:
        >>> params = ParameterCount(count=7.0, scale=ParameterScale.BILLION, is_estimate=False)
        >>> size = calculate_size_from_parameters(params, QuantizationType.FP16)
        >>> print(size.to_string())  # "~14.0 GB" (7B * 2 bytes)
    """
    # Convert to total parameters in billions
    total_params_billions = param_count.total_billions

    # Get multiplier for quantization type
    bytes_per_param = SIZE_MULTIPLIERS[quant_type]

    # Calculate base size (params * bytes/param)
    base_size_gb = total_params_billions * bytes_per_param

    # Add overhead for embeddings, configs, tokenizer (10-15%)
    overhead_factor = 1.0 + 0.125  # 12.5% overhead
    size_gb = base_size_gb * overhead_factor

    # Calculate uncertainty bounds (±10%)
    min_gb = size_gb * 0.9
    max_gb = size_gb * 1.1

    return ModelSize(
        size_gb=size_gb, min_gb=min_gb, max_gb=max_gb, is_estimate=True, source="calculated"
    )


def extract_parameter_pattern(text: str) -> Optional[ParameterCount]:
    """Extract parameter count from text using regex patterns.

    Args:
        text: Text to search (model name or repo ID).

    Returns:
        ParameterCount if pattern found, None otherwise.

    Example:
        >>> params = extract_parameter_pattern("Qwen2.5-7B-Instruct")
        >>> print(params.to_string())  # "7B"
    """
    text_lower = text.lower()

    # Pattern for billions: 7b, 7.5b, 70b, etc.
    billion_pattern = r"(\d+\.?\d*)b(?:illion)?"
    billion_matches = re.findall(billion_pattern, text_lower)

    if billion_matches:
        count = float(billion_matches[0])
        return ParameterCount(count=count, scale=ParameterScale.BILLION, is_estimate=True)

    # Pattern for millions: 350m, 125m, etc.
    million_pattern = r"(\d+)m(?:illion)?"
    million_matches = re.findall(million_pattern, text_lower)

    if million_matches:
        count = float(million_matches[0])
        return ParameterCount(count=count, scale=ParameterScale.MILLION, is_estimate=True)

    return None


# ============ Public API Functions ============


def get_disk_size_after_quant(link_hf_quant_repo: str, hf_api=None) -> ModelSize:
    """Get actual disk size of quantized model from HuggingFace Hub API.

    Fetches repository metadata via HF API and sums file sizes to get accurate
    total size. Falls back to estimates based on quantization level and parameter
    count if API call fails.

    Args:
        link_hf_quant_repo: HuggingFace repo ID for quantized model.
            Format: "mlx-community/Model-Name-4bit" or similar.
        hf_api: The HF client to ask (the catalog build passes its own retrying
            client so a snapshot never opens a second session); None falls back
            to the module-level ``get_hf_api()``.

    Returns:
        ModelSize object with actual size from API (``size_bytes`` set to the
        exact chosen-artifact byte count) or estimated size (``size_bytes`` None).

    Raises:
        HFAPIError: If API call fails and estimation is impossible.

    Examples:
        >>> # Get actual size via API
        >>> size = get_disk_size_after_quant("mlx-community/Mistral-7B-v0.3-4bit")
        >>> print(size.to_string())  # "~3.2 GB" (actual from API)
        >>>
        >>> # Fallback to estimate on error
        >>> size = get_disk_size_after_quant("mlx-community/Model-4bit")
        >>> print(size.to_string())  # "~3.0-4.0 GB" (estimated for 4-bit ~7B)

    Notes:
        - Uses get_hf_api().repo_info with files_metadata=True
        - Sums all file sizes in repo (weights, config, tokenizer)
        - Fallback logic: Detects quantization type and parameter count
        - Error handling: Logs error and returns best estimate
        - Precision: Returns size with 0.1 GB precision
    """
    try:
        logger.debug(f"Fetching disk size for quantized repo: {link_hf_quant_repo}")

        if hf_api is None:
            hf_api = get_hf_api()
        repo_info = hf_api.repo_info(link_hf_quant_repo, files_metadata=True)
        # Sum ONLY the artifact the downloader would actually fetch, not the whole
        # repo (#220/#170): a GGUF multi-quant repo can be 20-40 GB while we pull a
        # single ~1.8 GB quant. Single-artifact repos (mlx-community) select every
        # file, so the whole-repo sum is preserved for them.
        total_size_bytes = _chosen_artifact_bytes(repo_info)

        # Convert to GB with high precision. Decimal GB (#316) so the number
        # matches what Hugging Face itself shows for the very same file.
        size_gb = total_size_bytes / BYTES_PER_GB

        logger.info(f"Retrieved actual size for {link_hf_quant_repo}: {size_gb:.2f} GB")

        return ModelSize(
            size_gb=size_gb,
            min_gb=size_gb,
            max_gb=size_gb,
            is_estimate=False,
            source="api",
            # A listing with no sizes at all sums to 0: that is "unknown", not a
            # measured empty artifact.
            size_bytes=int(total_size_bytes) if total_size_bytes > 0 else None,
        )

    except Exception as e:
        logger.warning(f"Failed to fetch size from HF API for {link_hf_quant_repo}: {e}")
        logger.debug("Falling back to estimate based on quantization type")

        # Fallback: Estimate based on quantization type and parameter count
        quant_type = parse_quantization_type(link_hf_quant_repo)
        param_count = get_parameter_count_from_name("", link_hf_quant_repo)

        if param_count != "Unknown":
            # Have parameter count, calculate based on that
            try:
                params = extract_parameter_pattern(link_hf_quant_repo)
                if params:
                    estimated_size = calculate_size_from_parameters(params, quant_type)
                    logger.info(
                        f"Estimated size for {link_hf_quant_repo}: {estimated_size.to_string()}"
                    )
                    return estimated_size
            except Exception as calc_error:
                logger.debug(f"Failed to calculate from parameters: {calc_error}")

        # Rough fallback estimates based on common patterns
        if quant_type == QuantizationType.INT4:
            # 4-bit quantization, assume ~7B model
            return ModelSize(
                size_gb=3.5, min_gb=3.0, max_gb=4.0, is_estimate=True, source="fallback"
            )
        elif quant_type == QuantizationType.INT8:
            # 8-bit quantization, check for size hints
            repo_lower = link_hf_quant_repo.lower()
            if "1b" in repo_lower:
                return ModelSize(
                    size_gb=1.5, min_gb=1.0, max_gb=2.0, is_estimate=True, source="fallback"
                )
            elif "2b" in repo_lower:
                return ModelSize(
                    size_gb=2.5, min_gb=2.0, max_gb=3.0, is_estimate=True, source="fallback"
                )
            elif "4b" in repo_lower:
                return ModelSize(
                    size_gb=4.5, min_gb=4.0, max_gb=5.0, is_estimate=True, source="fallback"
                )
            else:
                # Default to ~7B 8-bit
                return ModelSize(
                    size_gb=7.0, min_gb=6.0, max_gb=8.0, is_estimate=True, source="fallback"
                )
        else:
            # Unknown quantization, very rough estimate
            logger.warning(f"Could not determine size for {link_hf_quant_repo}, returning unknown")
            return ModelSize(
                size_gb=0.0, min_gb=0.0, max_gb=0.0, is_estimate=True, source="unknown"
            )


def get_parameter_count_from_name(model_name: str, link: str) -> str:
    """Extract parameter count from model name or HuggingFace link.

    Uses regex to find common parameter count patterns in model names and
    repo IDs. Supports billion (B) and million (M) scale models.

    **DEPRECATED**: This function returns strings for backward compatibility.
    New code should use extract_parameter_pattern() which returns typed
    ParameterCount objects.

    Patterns Matched:
    - Billions: "7b", "7B", "7.5b", "70b" → "7B", "7.5B", "70B"
    - Millions: "350m", "125M" → "350M", "125M"
    - Variants: "billion", "million" words also matched

    Args:
        model_name: Human-readable model name (e.g., "Mistral 7B Instruct").
        link: HuggingFace repo ID (e.g., "mistralai/Mistral-7B-Instruct-v0.3").
            Both are lowercased and searched together.

    Returns:
        Parameter count string like "7B", "1.5B", "350M", or "Unknown" if
        no pattern matched.

    Examples:
        >>> # Standard cases
        >>> params = get_parameter_count_from_name(
        ...     "Qwen 2.5 7B Instruct",
        ...     "Qwen/Qwen2.5-7B-Instruct"
        ... )
        >>> print(params)  # "7B"
        >>>
        >>> # Decimal parameters
        >>> params = get_parameter_count_from_name(
        ...     "Phi 3.5 1.5B",
        ...     "microsoft/phi-3.5-mini-1.5b"
        ... )
        >>> print(params)  # "1.5B"

    Notes:
        - Case-insensitive: Converts to lowercase before matching
        - First match wins: If multiple patterns found, uses first
        - Backward compatibility: Returns string instead of ParameterCount
        - Deprecated: Use extract_parameter_pattern() for new code
    """
    combined_text = f"{model_name} {link}"
    param_count = extract_parameter_pattern(combined_text)

    if param_count:
        return param_count.to_string()
    else:
        return "Unknown"


def format_model_info_metadata(
    model_info: ModelInfo, size_estimate: Optional[ModelSize] = None, quantized: bool = False
) -> str:
    """Format HuggingFace ModelInfo object into structured string for storage.

    Converts a huggingface_hub.ModelInfo object into a multi-line string
    containing all relevant metadata fields. Includes size estimates and
    parameter counts for display in UI and database storage.

    Formatted Fields:
    - Model ID, Author, Created, Downloads, Likes
    - Library, Pipeline, Size, Parameters, Quantized status
    - Private/Gated flags, Tags (first 10), SHA, Last Modified

    Args:
        model_info: huggingface_hub.ModelInfo object from HF API. Contains
            all metadata fields from HuggingFace Hub.
        size_estimate: Optional ModelSize object. If None, shows "Unknown".
            Get from get_disk_size_after_quant.
        quantized: Boolean indicating if model is quantized (default: False).
            Shows "True" or "False" in output.

    Returns:
        Multi-line formatted string with all metadata fields.

    Raises:
        Whatever reading the ``model_info`` fields raises. The formatter does
        NOT catch its own failures (#354): a caught error used to come back as
        the string ``"Error formatting metadata: ..."``, which the catalog
        builders then stored as ``model_metadata``, so a broken row looked
        populated and the error was frozen into the bundled snapshot. The
        catalog builders' per-model failure paths log the repo id and skip
        the model instead.

    Examples:
        >>> from src.utils.hf_model_metadata import format_model_info_metadata
        >>> from huggingface_hub import HfApi
        >>>
        >>> # Fetch model info and format
        >>> api = HfApi()
        >>> model_info = api.model_info("mistralai/Mistral-7B-Instruct-v0.3")
        >>> size = get_disk_size_after_quant(model_info.id)
        >>>
        >>> metadata_str = format_model_info_metadata(
        ...     model_info,
        ...     size_estimate=size,
        ...     quantized=False
        ... )
        >>> print(metadata_str)
        # Model ID: mistralai/Mistral-7B-Instruct-v0.3
        # Author: mistralai
        # Created: 2024-05-22T14:00:00.000Z
        # Downloads: 1234567
        # Likes: 5678
        # Library: transformers
        # Pipeline: text-generation
        # Size: ~13.5 GB
        # Parameters: 7B
        # Quantized: False
        # ...

    Notes:
        - Parameter extraction: Uses extract_parameter_pattern internally
        - Tag limit: Shows first 10 tags, adds "..." if more exist
        - Use case: Store in Llm.model_metadata field for UI display
        - Timestamps: ISO format from HuggingFace Hub
    """
    # Extract parameter count from model ID
    param_count = extract_parameter_pattern(model_info.id)
    param_str = param_count.to_string() if param_count else "Unknown"

    # Format size estimate
    size_str = size_estimate.to_string() if size_estimate else "Unknown"

    metadata_str = f"""Model ID: {model_info.id}
Author: {model_info.author or 'Unknown'}
Created: {model_info.created_at or 'Unknown'}
Downloads: {model_info.downloads or 0} 
Likes: {model_info.likes or 0}
Library: {model_info.library_name or 'Unknown'}
Pipeline: {model_info.pipeline_tag or 'Unknown'}
Size: {size_str}
Parameters: {param_str}
Quantized: {quantized}
Private: {model_info.private}
Gated: {model_info.gated}
Tags: {', '.join(model_info.tags[:10]) if model_info.tags else 'None'}{'...' if model_info.tags and len(model_info.tags) > 10 else ''}
SHA: {model_info.sha or 'Unknown'}
Last Modified: {model_info.last_modified or 'Unknown'}"""

    logger.debug(f"Formatted metadata for {model_info.id}")
    return metadata_str


# ============ On-disk size (measured reality, #220) ============


def _chosen_artifact_bytes(repo_info) -> int:
    """Total bytes of the files the downloader would actually fetch from a repo.

    Mirrors the download path's file selection (``_select_download_files`` /
    ``pick_best_gguf`` in ``domains.llms.services``, the #170 fix) so catalog-time
    size matches the installed size: for GGUF multi-quant repos this is the single
    best quant (+ mmproj + small aux), NOT the whole repo. For single-artifact
    repos (e.g. mlx-community) the selection is every file, so the whole-repo sum
    is preserved. Falls back to the whole-repo sum when nothing is selectable
    (e.g. a "gguf" repo with no .gguf, or one whose weights are raw byte chunks)
    rather than reporting zero. No catalog row is built on that fallback for a
    GGUF engine: the resolver and the derived catalog only keep repos from which
    the downloader selects an artefact (``has_downloadable_gguf``).

    The selection helpers are imported lazily to keep this utility module free of a
    top-level dependency on the llms domain (utils sits below domains in the layering).
    """
    from src.domains.llms.services import _select_download_files, FILES_TO_EXCLUDE
    from src.core import config

    file_sizes = {
        s.rfilename: s.size
        for s in repo_info.siblings
        if getattr(s, "size", None) and s.rfilename not in FILES_TO_EXCLUDE
    }
    all_repo_files = [s.rfilename for s in repo_info.siblings]
    uses_gguf = bool(getattr(config.LLM_Engine, "USES_GGUF", False))
    selection = _select_download_files(all_repo_files, file_sizes, uses_gguf)
    chosen = [f for f in selection.files if f in file_sizes]
    if not chosen:
        return sum(file_sizes.values())
    return sum(file_sizes.get(f, 0) for f in chosen)


def measure_dir_size_bytes(path: Union[str, Path]) -> int:
    """Measure the real on-disk footprint of a model directory, recursively, in BYTES.

    The unit-free primitive: callers comparing a footprint against another byte
    count (a job's recorded ``total_bytes``, a manifest size) must use this and
    never round-trip through ``measure_dir_size_gb``. A bytes -> GB -> bytes
    round-trip silently changes meaning the day the GB divisor changes, and the
    error lands on whichever safety check does the comparing (#314/#316).

    Sums the size of every regular file under ``path``. Defensive by design: a
    missing path returns 0 and a per-file stat error is skipped, so an
    orphaned/removed model dir never crashes the caller (orphans are legitimate
    since #225/#208).

    Args:
        path: Directory (or single file) to measure.

    Returns:
        Size in bytes (0 if the path does not exist or is empty).
    """
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        try:
            return root.stat().st_size
        except OSError:
            return 0
    total_bytes = 0
    for entry in root.rglob("*"):
        try:
            if entry.is_file():
                total_bytes += entry.stat().st_size
        except OSError:
            continue
    return total_bytes


def measure_dir_size_gb(path: Union[str, Path]) -> float:
    """Measure the real on-disk footprint of a model directory, recursively, in GB.

    Display-layer wrapper over ``measure_dir_size_bytes``: it owns the divisor,
    i.e. what a "GB" means on screen. That is DECIMAL GB (#316), the unit the
    catalog and Hugging Face quote, so a model does not appear to shrink the
    moment it finishes downloading. Anything comparing against a byte count
    wants the primitive instead.

    Args:
        path: Directory (or single file) to measure.

    Returns:
        Size in gigabytes (0.0 if the path does not exist or is empty).
    """
    return measure_dir_size_bytes(path) / BYTES_PER_GB


def rewrite_size_in_metadata(metadata_str: Optional[str], size_gb: float) -> str:
    """Return ``metadata_str`` with its size rewritten to a measured value (#220).

    The metadata is the multi-line "Key: value" STRING the frontend parses
    (``parseMetadata`` in LandingPage.jsx). This replaces the ``Size:`` line in
    place with ``Size: ~X.X GB`` (same human format), and maintains a numeric
    ``Disk Size GB: X.XX`` line for future consumers. Both lines are appended when
    absent; every other line is preserved in order.

    Args:
        metadata_str: Existing metadata string (may be None/empty).
        size_gb: Measured on-disk size in gigabytes.

    Returns:
        The rewritten metadata string.
    """
    size_line = f"Size: ~{size_gb:.1f} GB"
    disk_line = f"Disk Size GB: {size_gb:.2f}"
    if not metadata_str:
        return f"{size_line}\n{disk_line}"

    out = []
    saw_size = False
    saw_disk = False
    for line in metadata_str.split("\n"):
        key = line.split(":", 1)[0].strip().lower() if ":" in line else ""
        if key == "size":
            out.append(size_line)
            saw_size = True
        elif key == "disk size gb":
            out.append(disk_line)
            saw_disk = True
        else:
            out.append(line)
    if not saw_size:
        out.append(size_line)
    if not saw_disk:
        out.append(disk_line)
    return "\n".join(out)


# ============ Module Exports ============

__all__ = [
    # Units
    "BYTES_PER_GB",
    # Data structures
    "ModelSize",
    "ParameterCount",
    "QuantizationType",
    "ParameterScale",
    # Main API functions
    "get_disk_size_after_quant",
    "format_model_info_metadata",
    # On-disk size (measured reality)
    "measure_dir_size_bytes",
    "measure_dir_size_gb",
    "rewrite_size_in_metadata",
    # Helper functions
    "parse_quantization_type",
    "calculate_size_from_parameters",
    "extract_parameter_pattern",
    # Exceptions
    "HFMetadataError",
    "HFAPIError",
    "ParameterExtractionError",
]
