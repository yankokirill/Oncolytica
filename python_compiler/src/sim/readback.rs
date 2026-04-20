//! # Шаг 7 — Обратная связь (Readback)

use bevy::{
    prelude::*,
    render::{
        render_resource::{Buffer, BufferDescriptor, BufferUsages, MapMode},
        renderer::RenderDevice,
    },
};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU8, Ordering};

use crate::sim::data::METRICS_OUT_COUNT;

pub const STAGING_BUFFER_BYTES: u64 = METRICS_OUT_COUNT as u64 * 4;

/// Состояния конечного автомата для избежания конфликтов WGPU
pub const STAGING_STATE_READY: u8         = 0; // Буфер свободен, можно писать
pub const STAGING_STATE_COPY_RECORDED: u8 = 1; // Записана команда copy_buffer, ждем map_async
pub const STAGING_STATE_MAPPING: u8       = 2; // map_async вызван, ждем GPU
pub const STAGING_STATE_MAPPED: u8        = 3; // Данные готовы, можно читать на CPU

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetricIndex {
    TotalLive       = 0,
    NProlif         = 1,
    NQuiesc         = 2,
    NNecr           = 3,
    FracProlif      = 4,
    FracQuiesc      = 5,
    FracNecr        = 6,
    FracHypoxic     = 7,
    CscCount        = 8,
    CscFrac         = 9,
    CentroidX       = 10,
    CentroidY       = 11,
    CentroidZ       = 12,
    RGyration       = 13,
    RInvasive       = 14,
    RNecrotic       = 15,
    RimThickness    = 16,
    MeanEnergy      = 17,
    MeanO2          = 18,
    MeanGlu         = 19,
    MeanSpeed       = 20,
    CrowdFrac       = 21,
    StepId          = 22,
}

impl MetricIndex {
    pub const ALL: &'static[MetricIndex] = &[
        Self::TotalLive, Self::NProlif, Self::NQuiesc, Self::NNecr,
        Self::FracProlif, Self::FracQuiesc, Self::FracNecr, Self::FracHypoxic,
        Self::CscCount, Self::CscFrac, Self::CentroidX, Self::CentroidY, Self::CentroidZ,
        Self::RGyration, Self::RInvasive, Self::RNecrotic, Self::RimThickness,
        Self::MeanEnergy, Self::MeanO2, Self::MeanGlu, Self::MeanSpeed, Self::CrowdFrac,
    ];
}

#[derive(Resource, Clone)]
pub struct MetricsReadback {
    pub latest_values: Arc<Mutex<Option<Vec<f32>>>>,
}

impl Default for MetricsReadback {
    fn default() -> Self {
        Self { latest_values: Arc::new(Mutex::new(None)) }
    }
}

impl MetricsReadback {
    pub fn take_data(&self) -> Option<Vec<f32>> {
        let mut guard = self.latest_values.lock().ok()?;
        guard.take() // Это стандартный метод Rust: забирает значение, кладет None
    }
}

#[derive(Resource)]
pub struct StagingBuffers {
    pub metrics: Buffer,
    pub state: Arc<AtomicU8>,
}

impl StagingBuffers {
    pub fn new(device: &RenderDevice) -> Self {
        let metrics = device.create_buffer(&BufferDescriptor {
            label:              Some("readback::metrics_staging"),
            size:               STAGING_BUFFER_BYTES,
            usage:              BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { 
            metrics, 
            state: Arc::new(AtomicU8::new(STAGING_STATE_READY)) 
        }
    }
}

pub fn read_mapped_data_system(
    staging: Res<StagingBuffers>,
    sink: Res<MetricsReadback>,
) {
    if staging.state.load(Ordering::SeqCst) == STAGING_STATE_MAPPED {
        let slice = staging.metrics.slice(..);
        
        {
            let mapped = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&mapped);
            
            if let Ok(mut guard) = sink.latest_values.lock() {
                *guard = Some(floats.to_vec());
            }
        } // mapped view уничтожается здесь

        // Немедленно освобождаем буфер для следующих копирований!
        staging.metrics.unmap();
        staging.state.store(STAGING_STATE_READY, Ordering::SeqCst);
    }
}

pub fn trigger_readback_system(
    staging: Res<StagingBuffers>,
) {
    // Инициируем чтение ТОЛЬКО если копия была записана в этом кадре
    if staging.state.compare_exchange(
        STAGING_STATE_COPY_RECORDED, 
        STAGING_STATE_MAPPING, 
        Ordering::SeqCst, 
        Ordering::SeqCst
    ).is_ok() {
        let slice = staging.metrics.slice(..);
        let state_clone = Arc::clone(&staging.state);
        
        slice.map_async(MapMode::Read, move |res| {
            if res.is_ok() {
                state_clone.store(STAGING_STATE_MAPPED, Ordering::SeqCst);
            } else {
                state_clone.store(STAGING_STATE_READY, Ordering::SeqCst);
            }
        });
    }
}
