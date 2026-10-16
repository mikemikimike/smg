//! Media-reference forwarding: decide per request whether the router
//! preprocesses media itself or forwards references to a vLLM gRPC worker that
//! advertises worker-side processing, and build the wire payload.

use std::{
    collections::{BTreeSet, HashSet},
    sync::Arc,
};

use axum::response::Response;
use llm_multimodal::{data_url::data_url_payload_bytes, media, MediaContentPart, Modality};
use openai_protocol::worker::MmProcessingMode;
use smg_grpc_client::{common_proto as common, vllm_proto as vllm};

use super::{
    capability::{runtime_supports_modality, SUPPORTS_VISION_LABEL},
    config::MultimodalComponents,
    plan::{MediaPlan, PlaceholderTokens},
};
use crate::{
    observability::metrics::Metrics,
    routers::{
        common::placement::{self, NO_MEDIA_REF_CAPABLE_WORKER_CODE},
        error,
        grpc::context::WorkerSelection,
    },
    worker::{ConnectionMode, RuntimeType, Worker, WorkerRegistry},
};

/// Worker label carrying the advertised backend (`GetServerInfo.mm_processor`).
pub(crate) const MM_PROCESSOR_LABEL: &str = "mm_processor";
/// Worker label carrying accepted URL schemes (`GetServerInfo.mm_media_ref_schemes`).
pub(crate) const MM_MEDIA_REF_SCHEMES_LABEL: &str = "mm_media_ref_schemes";

/// Where one request's media is fetched and preprocessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MmProcessing {
    Router,
    Worker,
}

impl MmProcessing {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Router => "router",
            Self::Worker => "worker",
        }
    }
}

/// Why a request cannot be forwarded as media references.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MmRefsError {
    /// A part carries per-item processing hints (the long-side cap, the
    /// sampling rate) no worker honors, and the router cannot apply them
    /// itself: it has no spec for the model, or the request carries a
    /// modality only the worker processes (audio).
    HintUnsupported,
    /// The model's placeholder anchor is not one the worker can expand.
    ModelNotOptedIn(Modality),
    /// vLLM does not accept this modality at all.
    ModalityUnsupported(Modality),
    /// A part kind with no reference form (inline bytes or embeddings).
    UnsupportedPart(&'static str),
    /// An inline `data:` payload above the router's byte cap.
    RefTooLarge {
        index: usize,
        bytes: usize,
        limit: usize,
    },
    /// The selected worker does not fetch this URL scheme.
    SchemeNotAccepted { scheme: String, accepted: String },
    /// EPD encode workers never process references.
    EncodeNotSupported,
    /// A selected leg does not advertise worker-side processing.
    WorkerNotCapable,
    /// The model has no registry spec, so the router cannot process its media
    /// itself, and the placement did not give it a worker that does.
    RouterSpecMissing,
    /// The model has registered workers and none of them processes media
    /// references, under the placement that forwards every request's media
    /// to the worker (`--mm-processing worker`). A verdict over the fleet,
    /// not a fault of the request: a terminal 503 ([`Self::into_response`]).
    NoCapableWorker { model: String, workers: usize },
}

impl MmRefsError {
    /// Stable error code for the HTTP response body.
    pub(crate) fn code(&self) -> &'static str {
        match self {
            Self::HintUnsupported => "multimodal_hint_unsupported_in_worker_mode",
            Self::ModelNotOptedIn(_) => "multimodal_worker_processing_unsupported_model",
            Self::ModalityUnsupported(_) | Self::UnsupportedPart(_) => "multimodal_not_supported",
            Self::RefTooLarge { .. } => "media_ref_too_large",
            Self::SchemeNotAccepted { .. } => "media_ref_scheme_not_accepted",
            Self::EncodeNotSupported | Self::WorkerNotCapable | Self::RouterSpecMissing => {
                "multimodal_not_supported"
            }
            Self::NoCapableWorker { .. } => NO_MEDIA_REF_CAPABLE_WORKER_CODE,
        }
    }

    /// The client's response: a fault of the request is a 400 under the
    /// variant's code; the fleet verdict ([`Self::NoCapableWorker`]) is the
    /// 503 worker selection has always answered that condition with
    /// ([`placement::no_media_ref_capable_worker`]), terminal for the retry
    /// layer and reported once, rate-limited, like the pool's other refusals.
    pub(crate) fn into_response(self) -> Response {
        let code = self.code();
        let message = self.to_string();
        match self {
            Self::NoCapableWorker { .. } => placement::no_media_ref_capable_worker(message),
            _ => error::bad_request(code, message),
        }
    }
}

impl std::fmt::Display for MmRefsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HintUnsupported => f.write_str(
                "per-item media hints (max_long_side_pixel, fps) cannot be forwarded to a worker, \
                 and the router cannot apply them itself: it has no media spec for this model, \
                 or an item is of a modality only the worker processes",
            ),
            Self::ModelNotOptedIn(modality) => write!(
                f,
                "this model's {modality} placeholder is not expandable by a vLLM worker"
            ),
            Self::ModalityUnsupported(modality) => {
                write!(f, "vLLM workers do not accept {modality} inputs")
            }
            Self::UnsupportedPart(kind) => {
                write!(f, "{kind} cannot be forwarded as a media reference")
            }
            Self::RefTooLarge {
                index,
                bytes,
                limit,
            } => write!(
                f,
                "media item {index}: inline payload is {bytes} bytes, above the {limit}-byte cap"
            ),
            Self::SchemeNotAccepted { scheme, accepted } => write!(
                f,
                "the selected worker does not fetch '{scheme}' URLs (accepts: {accepted}); \
                 file:// needs --allowed-local-media-path on the worker"
            ),
            Self::EncodeNotSupported => {
                f.write_str("media references cannot be routed through encode workers")
            }
            Self::WorkerNotCapable => f.write_str(
                "the selected worker does not advertise worker-side multimodal processing",
            ),
            Self::RouterSpecMissing => f.write_str(
                "the router has no media spec for this model's family and cannot process its \
                 media itself; a vLLM worker with --mm-processor inprocess|redis processes it \
                 under --mm-processing worker (or auto, when every worker of the model does)",
            ),
            Self::NoCapableWorker { model, workers } => write!(
                f,
                "model '{model}' has {workers} registered worker(s) and none processes media \
                 references (no worker advertises an mm_processor), so under --mm-processing \
                 worker its multimodal requests cannot be served: start the workers with \
                 worker-side media processing, or run the router with --mm-processing auto or \
                 router"
            ),
        }
    }
}

/// Whether `worker` can fetch and process media references itself.
pub(crate) fn worker_accepts_media_refs(worker: &dyn Worker) -> bool {
    let spec = &worker.metadata().spec;
    spec.runtime_type == RuntimeType::Vllm
        && *worker.connection_mode() == ConnectionMode::Grpc
        && spec
            .labels
            .get(MM_PROCESSOR_LABEL)
            .is_some_and(|value| !value.is_empty())
}

/// Whether `worker` runs an engine with a vision encoder (`supports_vision`
/// `"true"`, see [`SUPPORTS_VISION_LABEL`]) on the gRPC pipeline and does not
/// process media references: under worker-side placement no image or video
/// ever reaches it. A worker without the label is unknown, never this.
pub(crate) fn vision_worker_without_media_processing(worker: &dyn Worker) -> bool {
    worker.connection_mode().uses_grpc_pipeline()
        && worker
            .metadata()
            .spec
            .labels
            .get(SUPPORTS_VISION_LABEL)
            .is_some_and(|value| value == "true")
        && !worker_accepts_media_refs(worker)
}

/// The models among `workers` whose multimodal requests worker-side placement
/// cannot serve: a model with a worker that reports a vision encoder and no
/// worker that processes media references, judged over every registered
/// worker (healthy or not: the capability is configuration). What readiness
/// is held on, sorted for a stable line and body. Empty under any other
/// placement: `auto` processes such a model's media on the router, `router`
/// always does.
pub(crate) fn models_without_media_processing(
    processing: MmProcessingMode,
    workers: &[Arc<dyn Worker>],
) -> Vec<String> {
    if processing != MmProcessingMode::Worker {
        return Vec::new();
    }
    let mut vision_only = BTreeSet::new();
    let mut capable = HashSet::new();
    for worker in workers {
        if worker_accepts_media_refs(worker.as_ref()) {
            capable.insert(worker.model_id());
        } else if vision_worker_without_media_processing(worker.as_ref()) {
            vision_only.insert(worker.model_id());
        }
    }
    vision_only
        .into_iter()
        .filter(|model| !capable.contains(model))
        .map(str::to_string)
        .collect()
}

/// URL schemes `worker` advertised; empty when it advertised none.
pub(crate) fn worker_media_ref_schemes(worker: &dyn Worker) -> HashSet<String> {
    worker
        .metadata()
        .spec
        .labels
        .get(MM_MEDIA_REF_SCHEMES_LABEL)
        .map(|value| {
            value
                .split(',')
                .map(|scheme| scheme.trim().to_ascii_lowercase())
                .filter(|scheme| !scheme.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

fn ensure_worker_expandable(
    plan: &MediaPlan,
    placeholders: &PlaceholderTokens,
) -> Result<(), MmRefsError> {
    for &modality in plan.modalities() {
        // Audio uses the native worker processor; preprocessed audio tensors
        // remain unsupported on vLLM's separate router-processing path.
        if modality != Modality::Audio && !runtime_supports_modality(RuntimeType::Vllm, modality) {
            return Err(MmRefsError::ModalityUnsupported(modality));
        }
        if !placeholders.worker_expandable(modality) {
            return Err(MmRefsError::ModelNotOptedIn(modality));
        }
    }
    Ok(())
}

/// Whether the router's own pipeline can serve `plan` for a vLLM worker: it has
/// a spec for the model, and every modality is one it preprocesses for that
/// runtime (audio is the worker's own processor's alone). Checked before the
/// hint fallback, so a request the pipeline would refuse at worker selection
/// keeps the cheap refusal instead of a fetch and a decode first.
fn router_serves(plan: &MediaPlan, placeholders: &PlaceholderTokens) -> bool {
    !placeholders.is_worker_only()
        && plan
            .modalities()
            .iter()
            .all(|&modality| runtime_supports_modality(RuntimeType::Vllm, modality))
}

/// Under worker-side placement, a model whose registered workers include none
/// that processes media references cannot serve a forwarded request whichever
/// worker selection would pick: refused here, in preparation, before the
/// request is weighed against any worker and ahead of the selection stage's
/// own refusal of the condition (its backstop for a pool that changed in
/// between). Every registered worker counts, healthy or not, as under `auto`:
/// the capability is configuration, not health. A model with no worker at all
/// is left to selection, which answers for it (the 404 of a model nobody
/// serves, or the disaggregated leg that is missing).
fn ensure_fleet_processes_media_refs(
    registry: &WorkerRegistry,
    model_id: &str,
) -> Result<(), MmRefsError> {
    let workers = registry.get_by_model(model_id);
    if workers.is_empty()
        || workers
            .iter()
            .any(|worker| worker_accepts_media_refs(worker.as_ref()))
    {
        return Ok(());
    }
    Err(MmRefsError::NoCapableWorker {
        model: model_id.to_string(),
        workers: workers.len(),
    })
}

/// Decide where this request's media is processed. Runs in preparation, before
/// worker selection, so `auto` consults every registered worker of the model,
/// and `worker` refuses a model none of whose workers can take the request
/// ([`ensure_fleet_processes_media_refs`]).
pub(crate) fn resolve_mm_processing(
    components: &MultimodalComponents,
    registry: &WorkerRegistry,
    model_id: &str,
    plan: &MediaPlan,
    placeholders: &PlaceholderTokens,
) -> Result<MmProcessing, MmRefsError> {
    let (resolved, reason) = match components.processing {
        MmProcessingMode::Router => (MmProcessing::Router, "config"),
        MmProcessingMode::Worker => match plan_carries_hints(plan)? {
            // The hints are knobs of the router's own pipeline (the long-side
            // cap, the sampling rate): a hinted request that pipeline can serve
            // is preprocessed here, as under `auto`; every other request keeps
            // forwarding references.
            true if router_serves(plan, placeholders) => (MmProcessing::Router, "hint_fallback"),
            true => return Err(MmRefsError::HintUnsupported),
            false => {
                ensure_worker_expandable(plan, placeholders)?;
                ensure_fleet_processes_media_refs(registry, model_id)?;
                (MmProcessing::Worker, "config")
            }
        },
        MmProcessingMode::Auto => {
            if !plan.is_forwardable() {
                (MmProcessing::Router, "plan_not_forwardable")
            } else if ensure_worker_expandable(plan, placeholders).is_err() {
                (MmProcessing::Router, "model_not_opted_in")
            } else {
                // Every registered worker, healthy or not: a health flap must
                // not flip a model between modes request to request.
                let workers = registry.get_by_model(model_id);
                if workers.is_empty() {
                    (MmProcessing::Router, "auto_none")
                } else if workers
                    .iter()
                    .all(|worker| worker_accepts_media_refs(worker.as_ref()))
                {
                    (MmProcessing::Worker, "auto_uniform")
                } else {
                    (MmProcessing::Router, "auto_mixed")
                }
            }
        }
    };

    if resolved == MmProcessing::Router && placeholders.is_worker_only() {
        // No spec: the router has nothing to process the media with.
        return Err(MmRefsError::RouterSpecMissing);
    }

    // Where a model's media is processed is worker configuration, reported in
    // the worker's labels and countable on this metric. It is not a per-request
    // event, so it is not logged as one.
    Metrics::record_mm_processing(
        registry.metric_model_label(model_id),
        resolved.as_str(),
        reason,
    );
    Ok(resolved)
}

/// Post-selection check: every leg must accept references, and every URL
/// scheme in the plan must be one every leg that advertised schemes fetches
/// (the prefill leg processes them; the decode leg does too whenever the
/// identity relay does not apply, as with n>1).
pub(crate) fn ensure_selection_supports_media_refs(
    workers: &WorkerSelection,
    plan: &MediaPlan,
) -> Result<(), MmRefsError> {
    let legs: Vec<&dyn Worker> = match workers {
        WorkerSelection::Single { worker } => vec![worker.as_ref()],
        WorkerSelection::Disaggregated {
            prefill, decode, ..
        } => vec![prefill.as_ref(), decode.as_ref()],
    };
    if !legs.iter().all(|leg| worker_accepts_media_refs(*leg)) {
        return Err(MmRefsError::WorkerNotCapable);
    }
    let mut accepted: Option<HashSet<String>> = None;
    for leg in &legs {
        let schemes = worker_media_ref_schemes(*leg);
        if schemes.is_empty() {
            continue;
        }
        accepted = Some(match accepted {
            Some(current) => current.intersection(&schemes).cloned().collect(),
            None => schemes,
        });
    }
    let Some(accepted) = accepted else {
        return Ok(());
    };
    for part in plan.parts() {
        let Some(url) = part_url(part) else {
            continue;
        };
        let scheme = url_scheme(url);
        if !accepted.contains(&scheme) {
            let mut accepted: Vec<_> = accepted.into_iter().collect();
            accepted.sort();
            return Err(MmRefsError::SchemeNotAccepted {
                scheme,
                accepted: accepted.join(","),
            });
        }
    }
    Ok(())
}

fn part_url(part: &MediaContentPart) -> Option<&str> {
    match part {
        MediaContentPart::ImageUrl { url, .. }
        | MediaContentPart::VideoUrl { url, .. }
        | MediaContentPart::AudioUrl { url, .. } => Some(url),
        _ => None,
    }
}

/// Lower-cased URL scheme; `data` for data URLs, empty for bare paths.
fn url_scheme(url: &str) -> String {
    url.split_once(':')
        .map(|(scheme, _)| scheme)
        .filter(|scheme| {
            !scheme.is_empty()
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        })
        .map(str::to_ascii_lowercase)
        .unwrap_or_default()
}

/// Classify one part as a forwardable reference: its modality, URL and the
/// router's byte cap for inline payloads.
fn forwardable_ref(
    part: &MediaContentPart,
) -> Result<(common::Modality, &str, usize), MmRefsError> {
    match part {
        MediaContentPart::ImageUrl {
            url,
            max_long_side_pixel: None,
            ..
        } => Ok((common::Modality::Image, url, media::image_max_input_bytes())),
        MediaContentPart::VideoUrl {
            url,
            fps: None,
            max_long_side_pixel: None,
            ..
        } => Ok((common::Modality::Video, url, media::video_max_input_bytes())),
        MediaContentPart::ImageUrl { .. } | MediaContentPart::VideoUrl { .. } => {
            Err(MmRefsError::HintUnsupported)
        }
        MediaContentPart::ImageData { .. } => {
            Err(MmRefsError::UnsupportedPart("inline image bytes"))
        }
        MediaContentPart::VideoData { .. } => {
            Err(MmRefsError::UnsupportedPart("inline video bytes"))
        }
        MediaContentPart::AudioUrl { url, .. } => {
            Ok((common::Modality::Audio, url, media::audio_max_input_bytes()))
        }
        MediaContentPart::AudioData { .. } => {
            Err(MmRefsError::UnsupportedPart("inline audio bytes"))
        }
        MediaContentPart::ImageEmbeds { .. } => {
            Err(MmRefsError::UnsupportedPart("image embeddings"))
        }
        MediaContentPart::Text { .. } => Err(MmRefsError::UnsupportedPart("text")),
    }
}

/// Whether this plan is forwardable apart from per-item hints: `Ok(true)` when
/// a part carries one, `Ok(false)` when none does; a part with no reference
/// form at all is the error, whatever its position.
fn plan_carries_hints(plan: &MediaPlan) -> Result<bool, MmRefsError> {
    let mut hinted = false;
    for part in plan.parts() {
        match forwardable_ref(part) {
            Ok(_) => {}
            Err(MmRefsError::HintUnsupported) => hinted = true,
            Err(blocker) => return Err(blocker),
        }
    }
    Ok(hinted)
}

/// Build the wire payload from the plan, in authored (prompt) order.
pub(crate) fn assemble_media_refs(plan: MediaPlan) -> Result<vllm::MediaRefs, MmRefsError> {
    let mut items = Vec::with_capacity(plan.parts().len());
    for (index, part) in plan.parts().iter().enumerate() {
        let (modality, url, limit) = forwardable_ref(part)?;
        if let Some(bytes) = data_url_payload_bytes(url) {
            if bytes > limit {
                return Err(MmRefsError::RefTooLarge {
                    index,
                    bytes,
                    limit,
                });
            }
        }
        items.push(vllm::MediaRef {
            modality: modality as i32,
            url: url.to_string(),
        });
    }
    Ok(vllm::MediaRefs { items })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openai_protocol::worker::HealthCheckConfig;

    use super::*;
    use crate::{
        routers::grpc::multimodal::{
            config::MultimodalConfigRegistry, settings::MultimodalSettings,
        },
        worker::{BasicWorkerBuilder, ModelCard, WorkerType},
    };

    const MODEL: &str = "qwen3-vl";

    fn image_url(url: &str) -> MediaContentPart {
        MediaContentPart::ImageUrl {
            url: url.to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: None,
        }
    }

    fn plan(parts: Vec<MediaContentPart>) -> MediaPlan {
        MediaPlan::new(parts)
    }

    fn expandable_placeholders() -> PlaceholderTokens {
        let mut placeholders = PlaceholderTokens::default();
        placeholders.insert(Modality::Image, "<|image_pad|>".to_string());
        placeholders.set_worker_expandable(Modality::Image, true);
        placeholders
    }

    fn components(mode: MmProcessingMode) -> MultimodalComponents {
        let mut components = MultimodalComponents::new(
            Arc::new(MultimodalConfigRegistry::new()),
            None,
            None,
            &MultimodalSettings::default(),
        )
        .expect("components");
        components.processing = mode;
        components
    }

    fn worker(
        url: &str,
        runtime: RuntimeType,
        connection: ConnectionMode,
        labels: &[(&str, &str)],
    ) -> Arc<dyn Worker> {
        let mut builder = BasicWorkerBuilder::new(url)
            .model(ModelCard::new(MODEL))
            .worker_type(WorkerType::Regular)
            .runtime_type(runtime)
            .connection_mode(connection)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            });
        for (key, value) in labels {
            builder = builder.label(*key, *value);
        }
        Arc::new(builder.build())
    }

    fn capable(url: &str) -> Arc<dyn Worker> {
        worker(
            url,
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[
                (MM_PROCESSOR_LABEL, "inprocess"),
                (MM_MEDIA_REF_SCHEMES_LABEL, "http,https,data"),
            ],
        )
    }

    fn registry_with(workers: Vec<Arc<dyn Worker>>) -> WorkerRegistry {
        let registry = WorkerRegistry::new();
        for worker in workers {
            registry.register(worker).expect("register");
        }
        registry
    }

    type AcceptCase = (
        RuntimeType,
        ConnectionMode,
        &'static [(&'static str, &'static str)],
        bool,
    );

    #[test]
    fn accepts_only_labeled_vllm_grpc_workers() {
        let cases: [AcceptCase; 5] = [
            (
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[(MM_PROCESSOR_LABEL, "inprocess")],
                true,
            ),
            (
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[(MM_PROCESSOR_LABEL, "")],
                false,
            ),
            (RuntimeType::Vllm, ConnectionMode::Grpc, &[], false),
            (
                RuntimeType::Vllm,
                ConnectionMode::Zmq,
                &[(MM_PROCESSOR_LABEL, "inprocess")],
                false,
            ),
            (
                RuntimeType::Sglang,
                ConnectionMode::Grpc,
                &[(MM_PROCESSOR_LABEL, "inprocess")],
                false,
            ),
        ];
        for (i, (runtime, connection, labels, expected)) in cases.into_iter().enumerate() {
            let scheme = if connection == ConnectionMode::Zmq {
                format!("ipc:///tmp/smg-refs-{i}.ipc")
            } else {
                format!("grpc://127.0.0.1:{}", 9100 + i)
            };
            let w = worker(&scheme, runtime, connection, labels);
            assert_eq!(worker_accepts_media_refs(w.as_ref()), expected, "case {i}");
        }
    }

    #[test]
    fn schemes_label_is_parsed_case_insensitively() {
        let w = worker(
            "grpc://127.0.0.1:9200",
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[(MM_MEDIA_REF_SCHEMES_LABEL, "HTTP, https ,data,")],
        );
        let schemes = worker_media_ref_schemes(w.as_ref());
        assert_eq!(schemes.len(), 3);
        assert!(schemes.contains("http") && schemes.contains("https") && schemes.contains("data"));
    }

    /// A model without a registry spec: worker placement forwards it, auto
    /// forwards it when every worker of the model processes media itself and
    /// refuses it otherwise, router placement refuses it; the refusal names
    /// the router's missing spec, not the model.
    #[test]
    fn a_model_without_a_spec_goes_to_a_capable_worker_or_is_refused() {
        let worker_only = PlaceholderTokens::worker_only();
        let plan_ok = plan(vec![image_url("https://a/1.png")]);
        assert!(worker_only.worker_expandable(Modality::Image));

        assert_eq!(
            resolve_mm_processing(
                &components(MmProcessingMode::Worker),
                &registry_with(vec![]),
                MODEL,
                &plan_ok,
                &worker_only
            )
            .expect("worker placement forwards"),
            MmProcessing::Worker
        );
        let auto = components(MmProcessingMode::Auto);
        assert_eq!(
            resolve_mm_processing(
                &auto,
                &registry_with(vec![capable("grpc://127.0.0.1:9450")]),
                MODEL,
                &plan_ok,
                &worker_only
            )
            .expect("a uniform capable fleet forwards"),
            MmProcessing::Worker
        );
        for (registry, placement) in [
            (registry_with(vec![]), &auto),
            (
                registry_with(vec![capable("grpc://127.0.0.1:9451")]),
                &components(MmProcessingMode::Router),
            ),
        ] {
            let err = resolve_mm_processing(placement, &registry, MODEL, &plan_ok, &worker_only)
                .expect_err("nothing processes the media");
            assert_eq!(err, MmRefsError::RouterSpecMissing);
            assert_eq!(err.code(), "multimodal_not_supported");
            assert!(err.to_string().contains("no media spec"), "{err}");
        }
    }

    #[test]
    fn router_mode_never_forwards() {
        let registry = registry_with(vec![capable("grpc://127.0.0.1:9300")]);
        let resolved = resolve_mm_processing(
            &components(MmProcessingMode::Router),
            &registry,
            MODEL,
            &plan(vec![image_url("https://a/1.png")]),
            &expandable_placeholders(),
        )
        .expect("router mode resolves");
        assert_eq!(resolved, MmProcessing::Router);
    }

    #[test]
    fn auto_forwards_only_for_a_uniform_capable_fleet() {
        let placeholders = expandable_placeholders();
        let plan_ok = plan(vec![image_url("https://a/1.png")]);
        let components = components(MmProcessingMode::Auto);

        let uniform = registry_with(vec![
            capable("grpc://127.0.0.1:9400"),
            capable("grpc://127.0.0.1:9401"),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &uniform, MODEL, &plan_ok, &placeholders)
                .expect("resolves"),
            MmProcessing::Worker
        );

        let mixed = registry_with(vec![
            capable("grpc://127.0.0.1:9402"),
            worker(
                "grpc://127.0.0.1:9403",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[],
            ),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &mixed, MODEL, &plan_ok, &placeholders)
                .expect("resolves"),
            MmProcessing::Router
        );

        let empty = registry_with(vec![]);
        assert_eq!(
            resolve_mm_processing(&components, &empty, MODEL, &plan_ok, &placeholders)
                .expect("resolves"),
            MmProcessing::Router
        );
    }

    #[test]
    fn auto_falls_back_for_hints_and_unopted_models() {
        let components = components(MmProcessingMode::Auto);
        let registry = registry_with(vec![capable("grpc://127.0.0.1:9500")]);

        let hinted = plan(vec![MediaContentPart::ImageUrl {
            url: "https://a/1.png".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: Some(512),
        }]);
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &hinted,
                &expandable_placeholders()
            )
            .expect("resolves"),
            MmProcessing::Router
        );

        let mut unopted = PlaceholderTokens::default();
        unopted.insert(Modality::Image, "<|image|>".to_string());
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![image_url("https://a/1.png")]),
                &unopted
            )
            .expect("resolves"),
            MmProcessing::Router
        );
    }

    /// Under worker-side placement a model with workers and none that
    /// processes media references is refused in preparation, as the fleet
    /// verdict it is: a terminal 503 under the code selection has always
    /// answered the condition with, whose message names the flag and the
    /// fix. One capable worker lifts it (selection picks it); an empty pool
    /// is left to selection; a hinted request the router can serve keeps its
    /// router fallback; a fault of the request keeps its 400.
    #[test]
    fn worker_mode_refuses_a_fleet_that_processes_no_media_references() {
        use crate::routers::common::retry::is_retryable_response;

        let components = components(MmProcessingMode::Worker);
        let plan_ok = plan(vec![image_url("https://a/1.png")]);
        let placeholders = expandable_placeholders();
        let plain = |url: &str| worker(url, RuntimeType::Vllm, ConnectionMode::Grpc, &[]);

        let none_capable = registry_with(vec![
            plain("grpc://127.0.0.1:9800"),
            plain("grpc://127.0.0.1:9801"),
        ]);
        let err = resolve_mm_processing(&components, &none_capable, MODEL, &plan_ok, &placeholders)
            .expect_err("no worker processes media references");
        assert_eq!(
            err,
            MmRefsError::NoCapableWorker {
                model: MODEL.to_string(),
                workers: 2,
            }
        );
        assert_eq!(err.code(), "no_media_ref_capable_worker");
        let message = err.to_string();
        assert!(
            message.contains("--mm-processing worker") && message.contains("2 registered worker"),
            "{message}"
        );
        let response = err.into_response();
        assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response.headers()[error::HEADER_X_SMG_ERROR_CODE],
            "no_media_ref_capable_worker"
        );
        assert!(
            !is_retryable_response(&response),
            "a capability verdict is terminal for the retry layer"
        );
        assert!(
            placement::refusal_reason(&response).is_some(),
            "the refusal's line names its reason"
        );

        // An engine of another family never processes references either.
        let other_engine = registry_with(vec![worker(
            "grpc://127.0.0.1:9802",
            RuntimeType::TokenSpeed,
            ConnectionMode::Grpc,
            &[],
        )]);
        assert!(matches!(
            resolve_mm_processing(&components, &other_engine, MODEL, &plan_ok, &placeholders),
            Err(MmRefsError::NoCapableWorker { workers: 1, .. })
        ));

        let mixed = registry_with(vec![
            plain("grpc://127.0.0.1:9803"),
            capable("grpc://127.0.0.1:9804"),
        ]);
        assert_eq!(
            resolve_mm_processing(&components, &mixed, MODEL, &plan_ok, &placeholders)
                .expect("one capable worker is enough: selection picks it"),
            MmProcessing::Worker
        );

        let hinted = plan(vec![MediaContentPart::ImageUrl {
            url: "https://a/1.png".to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: Some(512),
        }]);
        assert_eq!(
            resolve_mm_processing(&components, &none_capable, MODEL, &hinted, &placeholders)
                .expect("the router applies the hint itself"),
            MmProcessing::Router
        );

        let mut unopted = PlaceholderTokens::default();
        unopted.insert(Modality::Image, "<|image|>".to_string());
        let fault = resolve_mm_processing(&components, &mixed, MODEL, &plan_ok, &unopted)
            .expect_err("an unopted model is the request's fault");
        assert_eq!(
            fault.code(),
            "multimodal_worker_processing_unsupported_model"
        );
        assert_eq!(
            fault.into_response().status(),
            http::StatusCode::BAD_REQUEST
        );
    }

    /// What readiness is held on: under worker-side placement, a model with
    /// a worker that reports a vision encoder and no worker that processes
    /// media references. A pool without the label (unknown) or with a
    /// `false` (a text-only engine) is not it, one capable worker lifts it
    /// for the model, and the other placements never hold readiness.
    #[test]
    fn models_without_media_processing_names_vision_pools_with_no_capable_worker() {
        fn worker_of(model: &str, url: &str, labels: &[(&str, &str)]) -> Arc<dyn Worker> {
            let mut builder = BasicWorkerBuilder::new(url)
                .model(ModelCard::new(model))
                .worker_type(WorkerType::Regular)
                .runtime_type(RuntimeType::Vllm)
                .connection_mode(ConnectionMode::Grpc)
                .health_config(HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                });
            for (key, value) in labels {
                builder = builder.label(*key, *value);
            }
            Arc::new(builder.build())
        }
        let vision = [(SUPPORTS_VISION_LABEL, "true")];
        let vision_capable = [
            (SUPPORTS_VISION_LABEL, "true"),
            (MM_PROCESSOR_LABEL, "inprocess"),
        ];
        let text_only = [(SUPPORTS_VISION_LABEL, "false")];

        let workers: Vec<Arc<dyn Worker>> = vec![
            // Two vision workers, neither processes references: held.
            worker_of("vl-b", "grpc://127.0.0.1:9900", &vision),
            worker_of("vl-b", "grpc://127.0.0.1:9901", &vision),
            // A vision worker without, beside one with: the model is served.
            worker_of("vl-a", "grpc://127.0.0.1:9902", &vision),
            worker_of("vl-a", "grpc://127.0.0.1:9903", &vision_capable),
            // Text-only and unlabeled pools say nothing about media.
            worker_of("text", "grpc://127.0.0.1:9904", &text_only),
            worker_of("legacy", "grpc://127.0.0.1:9905", &[]),
        ];
        assert_eq!(
            models_without_media_processing(MmProcessingMode::Worker, &workers),
            vec!["vl-b".to_string()]
        );
        for placement in [MmProcessingMode::Auto, MmProcessingMode::Router] {
            assert!(models_without_media_processing(placement, &workers).is_empty());
        }
        assert!(models_without_media_processing(MmProcessingMode::Worker, &[]).is_empty());

        // The predicate behind it, per worker.
        assert!(vision_worker_without_media_processing(workers[0].as_ref()));
        assert!(!vision_worker_without_media_processing(workers[3].as_ref()));
        assert!(!vision_worker_without_media_processing(workers[4].as_ref()));
        assert!(!vision_worker_without_media_processing(workers[5].as_ref()));
    }

    #[test]
    fn worker_mode_is_strict() {
        let components = components(MmProcessingMode::Worker);
        // No registered workers: the decision is still Worker; selection enforces.
        let registry = registry_with(vec![]);
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![image_url("https://a/1.png")]),
                &expandable_placeholders()
            )
            .expect("resolves"),
            MmProcessing::Worker
        );

        let mut unopted = PlaceholderTokens::default();
        unopted.insert(Modality::Image, "<|image|>".to_string());
        let err = resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![image_url("https://a/1.png")]),
            &unopted,
        )
        .expect_err("unopted model is refused");
        assert_eq!(err.code(), "multimodal_worker_processing_unsupported_model");

        let inline = plan(vec![MediaContentPart::ImageData {
            data: vec![0, 1, 2],
            mime_type: None,
            uuid: None,
            detail: None,
        }]);
        let err = resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &inline,
            &expandable_placeholders(),
        )
        .expect_err("inline bytes are refused by name");
        assert_eq!(err, MmRefsError::UnsupportedPart("inline image bytes"));
    }

    fn hinted_image(url: &str, max_long_side_pixel: u32) -> MediaContentPart {
        MediaContentPart::ImageUrl {
            url: url.to_string(),
            detail: None,
            uuid: None,
            max_long_side_pixel: Some(max_long_side_pixel),
        }
    }

    fn hinted_video(
        url: &str,
        fps: Option<f64>,
        max_long_side_pixel: Option<u32>,
    ) -> MediaContentPart {
        MediaContentPart::VideoUrl {
            url: url.to_string(),
            uuid: None,
            fps,
            max_long_side_pixel,
        }
    }

    #[test]
    fn worker_mode_preprocesses_hinted_items_itself() {
        let components = components(MmProcessingMode::Worker);
        let registry = registry_with(vec![capable("grpc://127.0.0.1:9700")]);
        let placeholders = expandable_placeholders();

        // The hints are the router pipeline's knobs: a request carrying one
        // is preprocessed here, whichever item carries it.
        for (name, hinted) in [
            (
                "image long side",
                plan(vec![hinted_image("https://a/1.png", 504)]),
            ),
            (
                "video fps",
                plan(vec![hinted_video("https://a/c.mp4", Some(1.0), None)]),
            ),
            (
                "video long side",
                plan(vec![hinted_video("https://a/c.mp4", None, Some(504))]),
            ),
            (
                "one hinted item among plain ones",
                plan(vec![
                    image_url("https://a/1.png"),
                    hinted_video("https://a/c.mp4", Some(2.0), None),
                ]),
            ),
        ] {
            assert_eq!(
                resolve_mm_processing(&components, &registry, MODEL, &hinted, &placeholders)
                    .unwrap_or_else(|err| panic!("{name}: {err}")),
                MmProcessing::Router,
                "{name}"
            );
        }

        // Plain items still forward.
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![image_url("https://a/1.png")]),
                &placeholders
            )
            .expect("plain items forward"),
            MmProcessing::Worker
        );

        // The router expands the placeholders itself, so the model's worker
        // expansion is not consulted for a hinted request.
        let mut unopted = PlaceholderTokens::default();
        unopted.insert(Modality::Image, "<|image|>".to_string());
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![hinted_image("https://a/1.png", 252)]),
                &unopted
            )
            .expect("the router processes it"),
            MmProcessing::Router
        );

        // A part with no reference form is still refused by name, before or
        // after the hinted one.
        let inline = MediaContentPart::ImageData {
            data: vec![0, 1, 2],
            mime_type: None,
            uuid: None,
            detail: None,
        };
        for parts in [
            vec![hinted_image("https://a/1.png", 504), inline.clone()],
            vec![inline.clone(), hinted_image("https://a/1.png", 504)],
        ] {
            assert_eq!(
                resolve_mm_processing(&components, &registry, MODEL, &plan(parts), &placeholders)
                    .expect_err("inline bytes are refused"),
                MmRefsError::UnsupportedPart("inline image bytes")
            );
        }

        // Without a spec the router cannot apply the hints: the request is
        // refused for the hint, not for the missing spec.
        let err = resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![hinted_image("https://a/1.png", 504)]),
            &PlaceholderTokens::worker_only(),
        )
        .expect_err("no pipeline applies the hint");
        assert_eq!(err.code(), "multimodal_hint_unsupported_in_worker_mode");
        assert!(
            err.to_string().contains("cannot apply them itself"),
            "{err}"
        );

        // Audio is the worker's alone on this runtime: a hinted request that
        // also carries it keeps the cheap refusal instead of a fetch and a
        // decode that worker selection would refuse, while the same audio
        // without a hint still forwards.
        let audio = MediaContentPart::AudioUrl {
            url: "https://a/x.wav".to_string(),
            uuid: None,
        };
        let mut with_audio = expandable_placeholders();
        with_audio.insert(Modality::Audio, "<|audio_pad|>".to_string());
        with_audio.set_worker_expandable(Modality::Audio, true);
        let err = resolve_mm_processing(
            &components,
            &registry,
            MODEL,
            &plan(vec![hinted_image("https://a/1.png", 504), audio.clone()]),
            &with_audio,
        )
        .expect_err("the router cannot serve audio to this worker");
        assert_eq!(err.code(), "multimodal_hint_unsupported_in_worker_mode");
        assert_eq!(
            resolve_mm_processing(
                &components,
                &registry,
                MODEL,
                &plan(vec![audio]),
                &with_audio
            )
            .expect("plain audio forwards"),
            MmProcessing::Worker
        );
    }

    #[test]
    fn selection_check_requires_every_leg_and_scheme() {
        let plan_https = plan(vec![image_url("https://a/1.png")]);
        let single = WorkerSelection::Single {
            worker: capable("grpc://127.0.0.1:9600"),
        };
        ensure_selection_supports_media_refs(&single, &plan_https).expect("capable single");

        let plain = WorkerSelection::Single {
            worker: worker(
                "grpc://127.0.0.1:9601",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[],
            ),
        };
        assert_eq!(
            ensure_selection_supports_media_refs(&plain, &plan_https),
            Err(MmRefsError::WorkerNotCapable)
        );

        let file_plan = plan(vec![image_url("file:///media/1.png")]);
        let err = ensure_selection_supports_media_refs(&single, &file_plan)
            .expect_err("file scheme not advertised");
        assert_eq!(err.code(), "media_ref_scheme_not_accepted");

        let pd = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: capable("grpc://127.0.0.1:9602"),
            decode: worker(
                "grpc://127.0.0.1:9603",
                RuntimeType::Vllm,
                ConnectionMode::Grpc,
                &[],
            ),
            runtime_type: RuntimeType::Vllm,
        };
        assert_eq!(
            ensure_selection_supports_media_refs(&pd, &plan_https),
            Err(MmRefsError::WorkerNotCapable)
        );

        // Both legs fetch: the accepted schemes are the intersection of what
        // each advertising leg fetches.
        let file_prefill = worker(
            "grpc://127.0.0.1:9604",
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[
                (MM_PROCESSOR_LABEL, "inprocess"),
                (MM_MEDIA_REF_SCHEMES_LABEL, "http,https,data,file"),
            ],
        );
        let mixed = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: file_prefill,
            decode: capable("grpc://127.0.0.1:9605"),
            runtime_type: RuntimeType::Vllm,
        };
        ensure_selection_supports_media_refs(&mixed, &plan_https).expect("https on both legs");
        assert_eq!(
            ensure_selection_supports_media_refs(&mixed, &file_plan),
            Err(MmRefsError::SchemeNotAccepted {
                scheme: "file".to_string(),
                accepted: "data,http,https".to_string(),
            })
        );

        // A leg that advertises no schemes leaves the other leg's set in force.
        let silent_decode = worker(
            "grpc://127.0.0.1:9606",
            RuntimeType::Vllm,
            ConnectionMode::Grpc,
            &[(MM_PROCESSOR_LABEL, "inprocess")],
        );
        let half = WorkerSelection::Disaggregated {
            encode_assignments: None,
            prefill: capable("grpc://127.0.0.1:9607"),
            decode: silent_decode,
            runtime_type: RuntimeType::Vllm,
        };
        assert_eq!(
            ensure_selection_supports_media_refs(&half, &file_plan)
                .expect_err("prefill's set still applies")
                .code(),
            "media_ref_scheme_not_accepted"
        );
    }

    #[test]
    fn assemble_keeps_prompt_order_and_maps_modalities() {
        let refs = assemble_media_refs(plan(vec![
            image_url("https://a/1.png"),
            MediaContentPart::VideoUrl {
                url: "data:video/mp4;base64,AAAA".to_string(),
                uuid: None,
                fps: None,
                max_long_side_pixel: None,
            },
            image_url("https://a/2.png"),
        ]))
        .expect("assembles");
        let items: Vec<(i32, &str)> = refs
            .items
            .iter()
            .map(|item| (item.modality, item.url.as_str()))
            .collect();
        assert_eq!(
            items,
            vec![
                (common::Modality::Image as i32, "https://a/1.png"),
                (common::Modality::Video as i32, "data:video/mp4;base64,AAAA"),
                (common::Modality::Image as i32, "https://a/2.png"),
            ]
        );
    }

    #[test]
    fn audio_refs_require_an_expandable_model_and_stay_in_prompt_order() {
        let plan = plan(vec![
            MediaContentPart::AudioUrl {
                url: "data:audio/wav;base64,AAAA".to_string(),
                uuid: None,
            },
            image_url("https://a/1.png"),
        ]);
        assert!(plan.is_forwardable());
        let mut placeholders = expandable_placeholders();
        assert_eq!(
            ensure_worker_expandable(&plan, &placeholders),
            Err(MmRefsError::ModelNotOptedIn(Modality::Audio))
        );
        placeholders.set_worker_expandable(Modality::Audio, true);
        ensure_worker_expandable(&plan, &placeholders).unwrap();
        let refs = assemble_media_refs(plan).unwrap();
        assert_eq!(refs.items[0].modality, common::Modality::Audio as i32);
        assert_eq!(refs.items[0].url, "data:audio/wav;base64,AAAA");
        assert_eq!(refs.items[1].modality, common::Modality::Image as i32);
    }

    #[test]
    fn assemble_rejects_unsupported_parts_and_oversized_data_urls() {
        let err = assemble_media_refs(plan(vec![MediaContentPart::AudioData {
            data: vec![0; 4],
            mime_type: Some("audio/wav".to_string()),
            uuid: None,
        }]))
        .expect_err("raw audio bytes have no reference form");
        assert_eq!(err, MmRefsError::UnsupportedPart("inline audio bytes"));

        let oversized = format!(
            "data:image/png;base64,{}",
            "A".repeat(media::image_max_input_bytes() * 4 / 3 + 8)
        );
        let err =
            assemble_media_refs(plan(vec![image_url(&oversized)])).expect_err("oversized data url");
        assert_eq!(err.code(), "media_ref_too_large");
    }

    #[test]
    fn url_scheme_and_data_url_size() {
        assert_eq!(url_scheme("HTTPS://a/1.png"), "https");
        assert_eq!(url_scheme("data:image/png;base64,AAAA"), "data");
        assert_eq!(url_scheme("/local/path.png"), "");
        assert_eq!(data_url_payload_bytes("https://a/1.png"), None);
        assert_eq!(
            data_url_payload_bytes("data:image/png;base64,AAAAAAAA"),
            Some(6)
        );
        assert_eq!(data_url_payload_bytes("DATA:text/plain,hello"), Some(5));
    }
}
