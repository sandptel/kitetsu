//! Moonshine ONNX backend via transcribe-rs.
//!
//! `MoonshineBackend` wraps `transcribe_rs::onnx::moonshine::MoonshineModel` and
//! adapts it to the `LoadedModel` interface. Inference is synchronous; there is
//! no incremental progress callback — a single `100` is sent on completion.
//!
//! The `#[cfg(feature = "onnx")]` / `#[cfg(not(feature = "onnx"))]` twin pattern
//! keeps call sites in `models/mod.rs` unconditional — the stub returns
//! `BackendNotCompiled` when the feature is off.

use std::path::PathBuf;
#[cfg(feature = "onnx")]
use std::sync::mpsc::Sender;

use crate::listener::ListenerError;

use super::{LoadedModel, MoonshineVariant};

// ── Model handle (feature-gated) ───────────────────────────────────────────────

/// Wraps a loaded Moonshine ONNX model ready for inference.
///
/// `MoonshineModel` is `Send` but not `Sync` (inference takes `&mut self`).
/// For two concurrent streams, load two separate `MoonshineBackend` instances
/// rather than sharing one behind a `Mutex` — model files are small and the
/// parallel-inference gain outweighs the extra memory.
#[cfg(feature = "onnx")]
pub(crate) struct MoonshineBackend {
    model: transcribe_rs::onnx::moonshine::MoonshineModel,
}

#[cfg(feature = "onnx")]
impl MoonshineBackend {
    /// Run synchronous Moonshine inference on `samples` (16 kHz mono f32).
    ///
    /// Moonshine does not expose incremental decode steps; `progress_tx` receives
    /// a single `100` when the call returns. The typical latency on a Zen 5
    /// 10-core CPU is ~50 ms for a 5-second chunk — fast enough that a progress
    /// bar adds little value, but the channel is kept for API compatibility.
    ///
    /// Audio must be between 0.1 s and 64 s; shorter or longer inputs are
    /// rejected by the ONNX model itself with a `Transcription` error.
    pub(crate) fn transcribe(
        &mut self,
        samples: &[f32],
        progress_tx: Option<Sender<u8>>,
    ) -> Result<String, ListenerError> {
        use transcribe_rs::{SpeechModel as _, TranscribeOptions};

        let options = TranscribeOptions {
            // Pin to English; Moonshine tiny/base are English-only models anyway.
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

/// Load a Moonshine ONNX model from `model_dir` and return a live engine handle.
///
/// `model_dir` must contain `encoder_model.onnx`, `decoder_model_merged.onnx`,
/// and `tokenizer.json`. Download from `onnx-community/moonshine-base-ONNX` (or
/// `moonshine-tiny-ONNX`) on HuggingFace.
///
/// Maps our `MoonshineVariant` to transcribe-rs's internal variant enum.
#[cfg(feature = "onnx")]
pub(crate) fn load(
    model_dir: PathBuf,
    variant: MoonshineVariant,
) -> Result<LoadedModel, ListenerError> {
    use transcribe_rs::onnx::{Quantization, moonshine::MoonshineVariant as TrVariant};

    let tr_variant = match variant {
        MoonshineVariant::Tiny => TrVariant::Tiny,
        MoonshineVariant::Base => TrVariant::Base,
    };

    tracing::info!(
        dir = %model_dir.display(),
        ?variant,
        "loading moonshine model",
    );

    let model = transcribe_rs::onnx::moonshine::MoonshineModel::load(
        &model_dir,
        tr_variant,
        &Quantization::FP32,
    )
    .map_err(|e| ListenerError::ModelUnavailable(e.to_string()))?;

    tracing::info!(dir = %model_dir.display(), "moonshine model ready");

    Ok(LoadedModel::Moonshine(MoonshineBackend { model }))
}

/// Stub: returns `BackendNotCompiled` when the `onnx` feature is off.
#[cfg(not(feature = "onnx"))]
pub(crate) fn load(
    model_dir: PathBuf,
    variant: MoonshineVariant,
) -> Result<LoadedModel, ListenerError> {
    let _ = (model_dir, variant);
    Err(ListenerError::BackendNotCompiled(
        "onnx — enable the 'onnx' feature",
    ))
}
