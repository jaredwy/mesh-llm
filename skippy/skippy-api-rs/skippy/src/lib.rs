//! Safe API over skippy_api's llama.cpp backend.
//!
//! Ownership follows the native layer:
//!   [`Runtime`]  owns every model and session (one native runtime behind a mutex)
//!   [`Model`]    borrows its `Runtime`; dropping it unloads the model
//!   [`Session`]  borrows its `Model`; dropping it releases the native context
//!
//! The borrows make the native rules static: a session cannot outlive its
//! model, a model cannot outlive the runtime, and `ModelInUse` cannot happen.
//! Calls are serialised by the runtime's mutex, so everything is `Send + Sync`.

use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard, PoisonError};

use skippy_sys as sys;
use sys::cxx::UniquePtr;

/// A vocabulary token id; layout-compatible with llama's `int32_t`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Token(pub i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum RuntimeError {
    #[error("model handle does not resolve")]
    InvalidModel,
    #[error("session handle does not resolve")]
    InvalidSession,
    #[error("model still has live sessions")]
    ModelInUse,
    #[error("native allocation failed")]
    AllocationFailed,
    #[error("model could not be loaded")]
    LoadFailed,
    #[error("native context could not be created")]
    SessionCreationFailed,
    #[error("session configuration not supported by the backend")]
    UnsupportedConfiguration,
    #[error("runtime capacity exhausted")]
    CapacityExceeded,
    #[error("backend returned an invalid resource")]
    InvalidBackendResult,
    #[error("native layer failed unexpectedly")]
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum WorkloadError {
    #[error("session handle does not resolve")]
    InvalidSession,
    #[error("model does not support this workload")]
    UnsupportedWorkload,
    #[error("rerank needs exactly one classifier output")]
    UnsupportedClassifierCount,
    #[error("input is empty")]
    EmptyInput,
    #[error("input exceeds the session's max_tokens")]
    TooManyTokens,
    #[error("more inputs than the session was created for")]
    TooManySequences,
    #[error("text could not be tokenized")]
    TokenizationFailed,
    #[error("model advertises no embedding width")]
    NoEmbeddingDimensions,
    #[error("output holds fewer than {required} values")]
    OutputTooSmall { required: usize },
    #[error("native allocation failed")]
    AllocationFailed,
    #[error("native decode failed")]
    DecodeFailed,
    #[error("native encode failed")]
    EncodeFailed,
    #[error("model has no decoder start token")]
    NoDecoderStartToken,
    #[error("decode produced no output")]
    NoOutput,
    #[error("native output width differs from the advertised width")]
    DimensionMismatch,
    #[error("native layer failed unexpectedly")]
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadKind {
    CausalGeneration,
    Embedding,
    Rerank,
    EncoderDecoder,
}

/// Pooling modes the embedding path can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pooling {
    None,
    Mean,
    Cls,
    Last,
}

/// Pooling as llama reports it. `Other` is a value from a newer llama.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativePooling {
    Unspecified,
    None,
    Mean,
    Cls,
    Last,
    Rank,
    Other(i8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkloadInfo {
    pub kind: WorkloadKind,
    pub pooling: NativePooling,
    /// Set iff the session can embed.
    pub embedding_pooling: Option<Pooling>,
    /// Embedding only, else 0.
    pub output_dimensions: usize,
    /// Rerank only, else 0.
    pub classifier_outputs: usize,
    pub has_encoder: bool,
    pub has_decoder: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionConfig {
    /// Longest single input, in tokens. The whole input goes through in one
    /// physical batch, so a single-sequence session's context is this size.
    pub max_tokens: u32,
    /// How many inputs one batch call may decode together. A batch session's
    /// context holds `sequences * max_tokens` tokens and costs accordingly.
    pub sequences: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self { max_tokens: 512, sequences: 1 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rerank {
    pub score: f32,
    /// Tokens the composed query/document prompt produced.
    pub token_count: usize,
}

pub struct Runtime {
    bridge: Mutex<UniquePtr<sys::Bridge>>,
}

impl Runtime {
    pub fn new() -> Result<Self, RuntimeError> {
        let bridge = sys::new_bridge();
        if bridge.is_null() {
            return Err(RuntimeError::AllocationFailed);
        }
        Ok(Self { bridge: Mutex::new(bridge) })
    }

    /// Loads a GGUF model. The path is passed to llama as UTF-8.
    pub fn load_model(&self, path: &str) -> Result<Model<'_>, RuntimeError> {
        let result = self.lock().pin_mut().load_model(path);
        runtime_result(result.status)?;
        Ok(Model { runtime: self, handle: result.model })
    }

    fn lock(&self) -> MutexGuard<'_, UniquePtr<sys::Bridge>> {
        self.bridge.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

pub struct Model<'rt> {
    runtime: &'rt Runtime,
    handle: UniquePtr<sys::Model>,
}

impl Model<'_> {
    pub fn create_session(&self, config: SessionConfig) -> Result<Session<'_>, RuntimeError> {
        let mut bridge = self.runtime.lock();
        let created = bridge.pin_mut().create_session(
            &self.handle,
            sys::SessionConfig { max_tokens: config.max_tokens, sequences: config.sequences },
        );
        runtime_result(created.status)?;
        let mut handle = created.session;

        let workload = bridge.workload(&handle);
        if let Err(error) = runtime_result(workload.status) {
            let _ = bridge.pin_mut().release_session(handle.pin_mut());
            return Err(error);
        }
        Ok(Session { runtime: self.runtime, _model: PhantomData, handle, workload: workload_info(workload.info) })
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        // Sessions borrow the model, so none are live: this cannot fail with
        // ModelInUse, and a stale handle after shutdown is nothing to report.
        let _ = self.runtime.lock().pin_mut().unload_model(self.handle.pin_mut());
    }
}

pub struct Session<'m> {
    runtime: &'m Runtime,
    _model: PhantomData<&'m ()>,
    handle: UniquePtr<sys::Session>,
    workload: WorkloadInfo,
}

impl Session<'_> {
    #[must_use]
    pub fn workload(&self) -> &WorkloadInfo {
        &self.workload
    }

    pub fn tokenize(&mut self, text: &str) -> Result<Vec<Token>, WorkloadError> {
        let mut tokens = Vec::new();
        self.tokenize_into(text, &mut tokens)?;
        Ok(tokens)
    }

    /// Replaces the contents of `tokens`, reusing its allocation.
    pub fn tokenize_into(&mut self, text: &str, tokens: &mut Vec<Token>) -> Result<(), WorkloadError> {
        let mut native = tokens_into_native(std::mem::take(tokens));
        let status = self.runtime.lock().pin_mut().tokenize(&self.handle, text, &mut native);
        *tokens = tokens_from_native(native);
        workload_result(status)
    }

    /// One unit-normalised embedding of `tokens`.
    pub fn embed(&mut self, tokens: &[Token]) -> Result<Vec<f32>, WorkloadError> {
        let mut output = vec![0.0f32; self.workload.output_dimensions];
        let written = self.embed_into(tokens, &mut output)?;
        output.truncate(written);
        Ok(output)
    }

    /// One score per document for `query`, decoded in a single native pass.
    /// Needs a session created with `SessionConfig::sequences` at least
    /// `documents.len()`.
    pub fn rerank_batch(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>, WorkloadError> {
        let mut scores = vec![0.0f32; documents.len()];
        let status = self.runtime.lock().pin_mut().rerank_batch(&self.handle, query, documents, &mut scores);
        workload_result(status)?;
        Ok(scores)
    }

    /// One unit-normalised embedding per group, decoded in a single native
    /// pass. Needs a session created with `SessionConfig::sequences` at least
    /// `groups.len()`.
    pub fn embed_batch(&mut self, groups: &[&[Token]]) -> Result<Vec<Vec<f32>>, WorkloadError> {
        let mut flat: Vec<i32> = Vec::with_capacity(groups.iter().map(|group| group.len()).sum());
        let mut lengths: Vec<u32> = Vec::with_capacity(groups.len());
        for group in groups {
            flat.extend(group.iter().map(|token| token.0));
            lengths.push(u32::try_from(group.len()).map_err(|_| WorkloadError::TooManyTokens)?);
        }

        let dimensions = self.workload.output_dimensions;
        let mut output = vec![0.0f32; dimensions * groups.len()];
        let result = self.runtime.lock().pin_mut().embed_batch(&self.handle, &flat, &lengths, &mut output);
        workload_result(result.status)?;
        Ok(output.chunks_exact(dimensions).map(<[f32]>::to_vec).collect())
    }

    /// Writes the embedding into the front of `output` and returns its width.
    pub fn embed_into(&mut self, tokens: &[Token], output: &mut [f32]) -> Result<usize, WorkloadError> {
        // The native check would report the zero width first; the kind is the
        // better answer and is already known here.
        if self.workload.kind != WorkloadKind::Embedding {
            return Err(WorkloadError::UnsupportedWorkload);
        }
        let result = self.runtime.lock().pin_mut().embed(&self.handle, tokens_as_native(tokens), output);
        if result.status == sys::WorkloadStatus::OutputTooSmall {
            return Err(WorkloadError::OutputTooSmall { required: result.dimensions });
        }
        workload_result(result.status)?;
        Ok(result.dimensions)
    }

    pub fn rerank(&mut self, query: &str, document: &str) -> Result<Rerank, WorkloadError> {
        let result = self.runtime.lock().pin_mut().rerank(&self.handle, query, document);
        workload_result(result.status)?;
        Ok(Rerank { score: result.score, token_count: result.token_count })
    }

    /// Runs the encoder over `tokens`; returns the token the decoder starts from.
    pub fn encode(&mut self, tokens: &[Token]) -> Result<Token, WorkloadError> {
        let result = self.runtime.lock().pin_mut().encode(&self.handle, tokens_as_native(tokens));
        workload_result(result.status)?;
        Ok(Token(result.decoder_start))
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        let _ = self.runtime.lock().pin_mut().release_session(self.handle.pin_mut());
    }
}

fn runtime_result(status: sys::RuntimeStatus) -> Result<(), RuntimeError> {
    use sys::RuntimeStatus as S;
    Err(match status {
        S::Ok => return Ok(()),
        S::InvalidModel => RuntimeError::InvalidModel,
        S::InvalidSession => RuntimeError::InvalidSession,
        S::ModelInUse => RuntimeError::ModelInUse,
        S::AllocationFailed => RuntimeError::AllocationFailed,
        S::LoadFailed => RuntimeError::LoadFailed,
        S::SessionCreationFailed => RuntimeError::SessionCreationFailed,
        S::UnsupportedConfiguration => RuntimeError::UnsupportedConfiguration,
        S::CapacityExceeded => RuntimeError::CapacityExceeded,
        S::InvalidBackendResult => RuntimeError::InvalidBackendResult,
        _ => RuntimeError::Internal,
    })
}

fn workload_result(status: sys::WorkloadStatus) -> Result<(), WorkloadError> {
    use sys::WorkloadStatus as S;
    Err(match status {
        S::Ok => return Ok(()),
        S::InvalidSession => WorkloadError::InvalidSession,
        S::UnsupportedWorkload => WorkloadError::UnsupportedWorkload,
        S::UnsupportedClassifierCount => WorkloadError::UnsupportedClassifierCount,
        S::EmptyInput => WorkloadError::EmptyInput,
        S::TooManyTokens => WorkloadError::TooManyTokens,
        S::TooManySequences => WorkloadError::TooManySequences,
        S::TokenizationFailed => WorkloadError::TokenizationFailed,
        S::NoEmbeddingDimensions => WorkloadError::NoEmbeddingDimensions,
        // The width travels with the result, not the status; see embed_into.
        S::OutputTooSmall => WorkloadError::OutputTooSmall { required: 0 },
        S::AllocationFailed => WorkloadError::AllocationFailed,
        S::DecodeFailed => WorkloadError::DecodeFailed,
        S::EncodeFailed => WorkloadError::EncodeFailed,
        S::NoDecoderStartToken => WorkloadError::NoDecoderStartToken,
        S::NoOutput => WorkloadError::NoOutput,
        S::DimensionMismatch => WorkloadError::DimensionMismatch,
        _ => WorkloadError::Internal,
    })
}

fn workload_info(info: sys::WorkloadInfo) -> WorkloadInfo {
    use sys::WorkloadKind as K;
    let kind = match info.kind {
        K::CausalGeneration => WorkloadKind::CausalGeneration,
        K::Embedding => WorkloadKind::Embedding,
        K::Rerank => WorkloadKind::Rerank,
        K::EncoderDecoder => WorkloadKind::EncoderDecoder,
        other => unreachable!("facade produced WorkloadKind {}", other.repr),
    };
    let pooling = native_pooling(info.pooling);
    let embedding_pooling = match native_pooling(info.embedding_pooling) {
        NativePooling::None => Some(Pooling::None),
        NativePooling::Mean => Some(Pooling::Mean),
        NativePooling::Cls => Some(Pooling::Cls),
        NativePooling::Last => Some(Pooling::Last),
        NativePooling::Unspecified | NativePooling::Rank | NativePooling::Other(_) => None,
    };
    WorkloadInfo {
        kind,
        pooling,
        embedding_pooling,
        output_dimensions: info.output_dimensions,
        classifier_outputs: info.classifier_outputs,
        has_encoder: info.has_encoder,
        has_decoder: info.has_decoder,
    }
}

fn native_pooling(pooling: sys::NativePooling) -> NativePooling {
    use sys::NativePooling as P;
    match pooling {
        P::Unspecified => NativePooling::Unspecified,
        P::None => NativePooling::None,
        P::Mean => NativePooling::Mean,
        P::Cls => NativePooling::Cls,
        P::Last => NativePooling::Last,
        P::Rank => NativePooling::Rank,
        other => NativePooling::Other(other.repr),
    }
}

// Token is repr(transparent) over i32: same size, alignment and allocation
// layout, so a Vec or slice of one is a Vec or slice of the other.
fn tokens_as_native(tokens: &[Token]) -> &[i32] {
    // SAFETY: see above; the slice keeps its length and lifetime.
    unsafe { std::slice::from_raw_parts(tokens.as_ptr().cast::<i32>(), tokens.len()) }
}

fn tokens_into_native(mut tokens: Vec<Token>) -> Vec<i32> {
    let (ptr, len, capacity) = (tokens.as_mut_ptr(), tokens.len(), tokens.capacity());
    std::mem::forget(tokens);
    // SAFETY: see above; the allocation was made for `capacity` elements of
    // the same size and alignment, and `tokens` no longer owns it.
    unsafe { Vec::from_raw_parts(ptr.cast::<i32>(), len, capacity) }
}

fn tokens_from_native(mut tokens: Vec<i32>) -> Vec<Token> {
    let (ptr, len, capacity) = (tokens.as_mut_ptr(), tokens.len(), tokens.capacity());
    std::mem::forget(tokens);
    // SAFETY: as tokens_into_native, in the other direction.
    unsafe { Vec::from_raw_parts(ptr.cast::<Token>(), len, capacity) }
}

const _: () = {
    assert!(size_of::<Token>() == size_of::<i32>());
    assert!(align_of::<Token>() == align_of::<i32>());
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_vectors_round_trip_without_reallocating() {
        let mut tokens = Vec::with_capacity(16);
        tokens.extend([Token(1), Token(2)]);
        let before = tokens.as_ptr();

        let native = tokens_into_native(tokens);
        assert_eq!(native, [1, 2]);
        let tokens = tokens_from_native(native);

        assert_eq!(tokens, [Token(1), Token(2)]);
        assert_eq!(tokens.capacity(), 16);
        assert_eq!(tokens.as_ptr(), before);
    }

    #[test]
    fn status_mapping_is_total() {
        assert_eq!(runtime_result(sys::RuntimeStatus::Ok), Ok(()));
        assert_eq!(runtime_result(sys::RuntimeStatus::InternalError), Err(RuntimeError::Internal));
        assert_eq!(runtime_result(sys::RuntimeStatus { repr: 200 }), Err(RuntimeError::Internal));
        assert_eq!(workload_result(sys::WorkloadStatus::Ok), Ok(()));
        assert_eq!(workload_result(sys::WorkloadStatus::DecodeFailed), Err(WorkloadError::DecodeFailed));
        assert_eq!(workload_result(sys::WorkloadStatus { repr: 200 }), Err(WorkloadError::Internal));
    }

    #[test]
    fn unknown_pooling_is_preserved() {
        assert_eq!(native_pooling(sys::NativePooling { repr: 9 }), NativePooling::Other(9));
        assert_eq!(native_pooling(sys::NativePooling::Rank), NativePooling::Rank);
    }
}
