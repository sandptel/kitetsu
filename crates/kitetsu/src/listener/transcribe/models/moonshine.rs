//! Moonshine v2 (streaming) ONNX backend via transcribe-rs.
//!
//! `MoonshineBackend` wraps `transcribe_rs::onnx::moonshine::StreamingModel` —
//! the 5-session streaming pipeline (frontend → encoder → adapter → cross_kv →
//! decoder_kv) — and adapts it to the `LoadedModel` interface. Inference is
//! synchronous and run over the whole buffer at once here; a single `100` is
//! sent on completion (no incremental callback).
//!
//! The `#[cfg(feature = "onnx")]` / `#[cfg(not(feature = "onnx"))]` twin pattern
//! keeps call sites in `models/mod.rs` unconditional — the stub returns
//! `BackendNotCompiled` when the feature is off.

use std::path::PathBuf;
#[cfg(feature = "onnx")]
use std::sync::mpsc::Sender;

use crate::listener::ListenerError;

use super::{LoadedModel, MoonshineVariant, Quantization};

// ── Model handle (feature-gated) ───────────────────────────────────────────────

/// Wraps a loaded Moonshine v2 streaming model ready for inference.
///
/// `StreamingModel` is `Send` but not `Sync` (inference takes `&mut self`).
/// For two concurrent streams, load two separate `MoonshineBackend` instances
/// rather than sharing one behind a `Mutex` — the parallel-inference gain
/// outweighs the extra memory.
#[cfg(feature = "onnx")]
pub(crate) struct MoonshineBackend {
    model: transcribe_rs::onnx::moonshine::StreamingModel,
}

#[cfg(feature = "onnx")]
impl MoonshineBackend {
    /// Run synchronous Moonshine inference on `samples` (16 kHz mono f32).
    ///
    /// Moonshine does not expose incremental decode steps through this path;
    /// `progress_tx` receives a single `100` when the call returns. Latency on a
    /// Zen 5 CPU is tens of milliseconds for a few seconds of audio.
    pub(crate) fn transcribe(
        &mut self,
        samples: &[f32],
        progress_tx: Option<Sender<u8>>,
    ) -> Result<String, ListenerError> {
        use transcribe_rs::{SpeechModel as _, TranscribeOptions};

        let options = TranscribeOptions {
            // Pin to English; the streaming variants are English-only anyway.
            language: Some("en".to_string()),
            ..Default::default()
        };

        let result = self
            .model
            .transcribe(samples, &options)
            .map_err(|e| ListenerError::Transcription(e.to_string()))?;

        // Signal completion — the caller's polling loop expects a final 100.
        if let Some(tx) = progress_tx {
            let _ = tx.send(100);
        }

        Ok(result.text.trim().to_owned())
    }
}

// ── Backend entry point (cfg-gated twin) ──────────────────────────────────────

/// Load a Moonshine v2 streaming model from `model_dir` and return a live handle.
///
/// `model_dir` must contain the 5 ONNX sessions (`frontend`, `encoder`,
/// `adapter`, `cross_kv`, `decoder_kv`), `tokenizer.bin`, and
/// `streaming_config.json`. `quantization` selects the preferred precision file
/// (`{name}.int8.onnx` etc.), falling back to FP32 if that file is absent.
///
/// `variant` is informational — the streaming engine reads its real dimensions
/// from `streaming_config.json`. Threads default to the host's parallelism.
#[cfg(feature = "onnx")]
pub(crate) fn load(
    model_dir: PathBuf,
    variant: MoonshineVariant,
    quantization: Quantization,
) -> Result<LoadedModel, ListenerError> {
    use transcribe_rs::onnx::{Quantization as TrQuant, moonshine::StreamingModel};

    let tr_quant = match quantization {
        Quantization::Fp32 => TrQuant::FP32,
        Quantization::Fp16 => TrQuant::FP16,
        Quantization::Int8 => TrQuant::Int8,
        Quantization::Int4 => TrQuant::Int4,
    };

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    tracing::info!(
        dir = %model_dir.display(),
        ?variant,
        ?quantization,
        num_threads,
        "loading moonshine v2 streaming model",
    );

    let model = StreamingModel::load(&model_dir, num_threads, &tr_quant)
        .map_err(|e| ListenerError::ModelUnavailable(e.to_string()))?;

    tracing::info!(dir = %model_dir.display(), "moonshine v2 model ready");

    Ok(LoadedModel::Moonshine(MoonshineBackend { model }))
}

/// Stub: returns `BackendNotCompiled` when the `onnx` feature is off.
#[cfg(not(feature = "onnx"))]
pub(crate) fn load(
    model_dir: PathBuf,
    variant: MoonshineVariant,
    quantization: Quantization,
) -> Result<LoadedModel, ListenerError> {
    let _ = (model_dir, variant, quantization);
    Err(ListenerError::BackendNotCompiled(
        "onnx — enable the 'onnx' feature",
    ))
}
