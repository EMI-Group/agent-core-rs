//! Multimodal documents plus offline image and table ingestion.
//!
//! Port of the multimodal parts of the Python
//! `openjiuwen/core/retrieval/` package that can run without a vision model:
//! [`MultimodalDocument`] with its `text`/`image`/`audio`/`video` content parts
//! (`common/document.py`), the image captioning seam with the offline fallback
//! (`indexing/processor/parser/captioner.py`), the image loader
//! (`image_parser.py`), and the row/column/Markdown table conversion of
//! `excel_parser.py`.
//!
//! Offline degradation is explicit: [`OfflineCaptioner`] describes an image by
//! its file name (never by its pixels) and marks the document metadata with
//! `caption_status: "offline"`. A model-backed captioner is injected through
//! [`ImageCaptioner`]; it belongs to an upper crate because this crate does not
//! depend on `openjiuwen-foundation`.
//!
//! Deviations from Python, all deliberate: `MultimodalDocument` stores its
//! content parts in a public serializable list instead of a private `_data`
//! cache, `strip()` keeps a non-empty `text` (Python only inspects the parts),
//! and the image loader reports `NotFound`/`Unsupported` errors instead of
//! silently returning an empty document list.
//!
//! Loader registration is not wired here: [`ImageLoader`] implements
//! [`DocumentLoader`](crate::loaders::DocumentLoader) and is registered with
//! `LoaderRegistry`/`AutoLoader` by a later round.

pub mod caption;
pub mod document;
pub mod image;
pub mod modality;
pub mod table;

pub use caption::{ImageCaptioner, OfflineCaptioner, DEFAULT_CAPTION_PREFIX, IMAGE_CAPTION_PROMPT};
pub use document::{parse_data_uri, to_data_uri, ContentPart, MultimodalDocument};
pub use image::{ImageLoader, IMAGE_EXTENSIONS};
pub use modality::{guess_mime_type, Modality};
pub use table::TableData;
