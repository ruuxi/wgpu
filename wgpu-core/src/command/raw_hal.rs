//! Deferred raw HAL commands: backend commands recorded by a callback at a
//! fixed point of a wgpu command stream.
//!
//! [`CommandEncoder::as_hal_mut`] gives direct access to the HAL encoder, but
//! only on an encoder that is never used with the wgpu encoding API, and the
//! resources it touches are not tracked. [`CommandEncoder::as_hal_deferred`]
//! instead records a command in the normal (deferred) command list: when the
//! command buffer is encoded, wgpu-core first transitions the declared
//! resources to the declared states (as [`CommandEncoder::transition_resources`]
//! does), records their initialization, and then runs the callback with the
//! open HAL encoder. The callback may record anything, but must leave every
//! declared resource in the state (for Vulkan: the image layout wgpu-hal derives
//! for that usage) it was given, and must not keep the encoder.
//!
//! Retention: what the callback returns is kept with the command buffer and
//! dropped only once that command buffer can no longer run: after the
//! submission that contains it has finished executing on the GPU, or when the
//! command buffer is dropped without being submitted (or fails to submit). The
//! callback moves into it whatever its recorded commands refer to that wgpu
//! does not track (native objects, extra image views). It is dropped inside
//! wgpu's submission bookkeeping (`Device::poll`, `Queue::submit`), so its
//! `Drop` must neither call back into wgpu nor block.
//!
//! Initialization: a texture declared with a state that includes a write usage
//! is treated as fully written by the callback (implicitly initialized); a
//! texture declared read-only must hold initialized contents (wgpu-core clears
//! it first when it does not). Buffers are handled the same way over their
//! whole size.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::{any::Any, fmt};

use crate::{
    command::{
        clear_texture, encoder::EncodingState, transition_resources, ArcCommand, CommandEncoder,
        CommandEncoderError, EncoderStateError, TransitionResourcesError,
    },
    device::queue::TempResource,
    init_tracker::{MemoryInitKind, TextureInitRange, TextureInitTrackerAction},
    resource::{Buffer, Labeled as _, Texture},
};

/// The callback of a deferred raw HAL command. What it returns is retained
/// until its command buffer can no longer run (see the [module docs](self)).
pub type RawHalFn =
    Box<dyn FnOnce(&mut dyn hal::DynCommandEncoder) -> Box<dyn Any + Send> + Send + 'static>;

/// What a [`RawHalFn`] returned, kept with its command buffer (among its
/// temporary resources) until that command buffer has finished executing or is
/// dropped unsubmitted.
pub struct RawHalRetained(#[allow(dead_code)] Box<dyn Any + Send>);

impl fmt::Debug for RawHalRetained {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RawHalRetained")
    }
}

/// A [`RawHalFn`] shared by the (cloneable) command that carries it. It runs at
/// most once; a clone of the command (trace recording) carries no callback.
#[derive(Clone)]
pub struct RawHalCallback(Arc<wgpu_sync::Mutex<Option<RawHalFn>>>);

impl RawHalCallback {
    pub fn new(f: RawHalFn) -> Self {
        Self(Arc::new(wgpu_sync::Mutex::new(Some(f))))
    }

    fn take(&self) -> Option<RawHalFn> {
        self.0.lock().take()
    }
}

impl fmt::Debug for RawHalCallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RawHalCallback")
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for RawHalCallback {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_unit()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for RawHalCallback {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        <() as serde::Deserialize>::deserialize(deserializer)?;
        Ok(Self(Arc::new(wgpu_sync::Mutex::new(None))))
    }
}

const WRITE_USES: wgt::TextureUses = wgt::TextureUses::COPY_DST
    .union(wgt::TextureUses::COLOR_TARGET)
    .union(wgt::TextureUses::DEPTH_WRITE)
    .union(wgt::TextureUses::STENCIL_WRITE)
    .union(wgt::TextureUses::STORAGE_WRITE_ONLY)
    .union(wgt::TextureUses::STORAGE_READ_WRITE)
    .union(wgt::TextureUses::STORAGE_ATOMIC);

const BUFFER_WRITE_USES: wgt::BufferUses =
    wgt::BufferUses::COPY_DST.union(wgt::BufferUses::STORAGE_READ_WRITE);

impl CommandEncoder {
    /// Record `callback` at this point of the command stream, after
    /// transitioning the given resources. See the [module docs](self).
    pub fn as_hal_deferred(
        self: &Arc<Self>,
        buffer_transitions: impl Iterator<Item = wgt::BufferTransition<Arc<Buffer>>>,
        texture_transitions: impl Iterator<Item = wgt::TextureTransition<Arc<Texture>>>,
        callback: RawHalFn,
    ) {
        profiling::scope!("CommandEncoder::as_hal_deferred");
        let res = {
            let mut cmd_buf_data = self.data.lock();
            cmd_buf_data.push_with(|| -> Result<_, TransitionResourcesError> {
                Ok(ArcCommand::RawHal {
                    buffer_transitions: buffer_transitions
                        .map(|t| {
                            t.buffer.check_is_valid()?;
                            Ok(t)
                        })
                        .collect::<Result<_, TransitionResourcesError>>()?,
                    texture_transitions: texture_transitions
                        .map(|t| {
                            t.texture.check_valid()?;
                            Ok(t)
                        })
                        .collect::<Result<_, TransitionResourcesError>>()?,
                    callback: RawHalCallback::new(callback),
                })
            })
        };
        if let Err(err) = res {
            let err: EncoderStateError = err;
            self.device
                .handle_error(err, Some(self.label()), "CommandEncoder::as_hal_deferred");
        }
    }
}

/// Encode a [`ArcCommand::RawHal`]: initialization, transitions, the callback.
pub(crate) fn encode_raw_hal(
    state: &mut EncodingState,
    buffer_transitions: Vec<wgt::BufferTransition<Arc<Buffer>>>,
    texture_transitions: Vec<wgt::TextureTransition<Arc<Texture>>>,
    callback: RawHalCallback,
) -> Result<(), CommandEncoderError> {
    for t in &buffer_transitions {
        let kind = if t.state.intersects(BUFFER_WRITE_USES) {
            MemoryInitKind::ImplicitlyInitialized
        } else {
            MemoryInitKind::NeedsInitializedMemory
        };
        let size = t.buffer.size;
        state.buffer_memory_init_actions.extend(
            t.buffer
                .initialization_status
                .read()
                .create_action(&t.buffer, 0..size, kind),
        );
    }
    for t in &texture_transitions {
        let selector = t
            .selector
            .clone()
            .unwrap_or_else(|| t.texture.full_range.clone());
        let kind = if t.state.intersects(WRITE_USES) {
            MemoryInitKind::ImplicitlyInitialized
        } else {
            MemoryInitKind::NeedsInitializedMemory
        };
        let layer_range = if t.texture.desc.dimension == wgt::TextureDimension::D3 {
            0..1
        } else {
            selector.layers.clone()
        };
        let action = TextureInitTrackerAction {
            texture: t.texture.clone(),
            range: TextureInitRange {
                mip_range: selector.mips.clone(),
                layer_range,
            },
            kind,
        };
        let immediate = state
            .texture_memory_actions
            .register_init_action(&action, None);
        for init in immediate {
            let index = init.layer_or_depth_slice;
            let (layer_range, depth_slice) =
                if init.texture.desc.dimension == wgt::TextureDimension::D3 {
                    (0..1, Some(index))
                } else {
                    (index..(index + 1), None)
                };
            clear_texture(
                &init.texture,
                TextureInitRange {
                    mip_range: init.mip_level..(init.mip_level + 1),
                    layer_range,
                },
                depth_slice,
                state.raw_encoder,
                &mut state.tracker.textures,
                &state.device.alignments,
                state.device.zero_buffer.as_ref(),
                state.snatch_guard,
                state.device.instance_flags,
            )?;
        }
    }
    transition_resources::transition_resources(state, buffer_transitions, texture_transitions)?;
    if let Some(f) = callback.take() {
        let retained = f(state.raw_encoder);
        state
            .temp_resources
            .push(TempResource::RawHalRetained(RawHalRetained(retained)));
    }
    Ok(())
}
