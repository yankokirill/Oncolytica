//! # Шаг 1 — Подготовка данных
//!
//! Rust-структуры с WGSL-совместимым layout (bytemuck) и создание
//! GPU-буферов / текстур через RenderDevice.

use bevy::{
    render::{
        render_resource::{
            AddressMode, Buffer, BufferDescriptor, BufferUsages, Extent3d, FilterMode,
            Sampler, SamplerDescriptor, StorageTextureAccess, Texture, TextureDescriptor,
            TextureDimension, TextureFormat, TextureUsages, TextureView,
            TextureViewDescriptor, TextureViewDimension,
        },
        renderer::RenderDevice,
    },
    prelude::*,
    render::extract_resource::ExtractResource,
    render::storage::ShaderStorageBuffer,
};
use bytemuck::{Pod, Zeroable};
use std::mem::size_of;
use std::sync::atomic::AtomicBool;

// ─── константы размерностей ──────────────────────────────────────────────────

pub const MAX_AGENTS:          u32 = 100000;
pub const GRID_DIM:            u32 = 64;
pub const MAX_CELL_PER_VOXEL:  u32 = 16;
pub const MAX_SPECIES:         u32 = 8;
pub const METRICS_ACCUM_COUNT: u32 = 19;
pub const METRICS_OUT_COUNT:   u32 = 23;
pub const VOXEL_COUNT:         u32 = GRID_DIM * GRID_DIM * GRID_DIM;
pub const WARMUP_STEPS:        u32 = 50;
pub const DIVISION_COUNT:      u32 = 18;

// ─── структуры данных (должны совпадать с WGSL-layout) ──────────────────────

/// Одна клетка в агент-буфере.  
/// Размер = 16 × f32 = 64 байта (кратно 16 для std430).
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[repr(C)]
pub struct CellData {
    pub position:        [f32; 3],
    pub species_id:      i32,
    pub custom_params:   [f32; 3],
    pub state:           i32,
    pub alive_age:       f32,
    pub dead_age:        f32,
    pub energy:          f32,
    pub prolif_capacity: i32,
    pub rng_state:       u32,
    pub _reserved:       [i32; 3],
}

/// Параметры одного биологического вида (16 × f32 = 64 байта).
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct SpeciesParam {
    pub consume_o2:         f32,
    pub consume_glu:        f32,
    pub hypoxia_death_thr:  f32,
    pub base_move_prob:     f32,
    pub chemo_taxis_o2:     f32,
    pub chemo_taxis_glu:    f32,
    pub base_divide_time:   f32,
    pub prob_symmetric_div: f32,
    pub max_proliferations: f32,
    pub spontaneous_death:  f32,
    pub _pad1:              f32,
    pub _pad2:              f32,
    pub _pad3:              [f32; 4],
}

impl Default for SpeciesParam {
    fn default() -> Self {
        Self {
            consume_o2: 0.005, consume_glu: 0.05, hypoxia_death_thr: 0.05,
            base_move_prob: 0.5, chemo_taxis_o2: 1.0, chemo_taxis_glu: 0.5,
            base_divide_time: 10.0, prob_symmetric_div: 1.0, max_proliferations: 20.0,
            spontaneous_death: 0.001, _pad1: 0.0, _pad2: 0.0, _pad3: [0.0; 4],
        }
    }
}

/// Глобальные uniform-параметры симуляции.
/// Должны совпадать с `GlobalUniforms` в WGSL шейдере.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct GlobalUniforms {
    pub delta_time:     f32,
    pub step_count:     i32,
    pub total_agents:   i32,
    pub initial_agents: i32,

    pub necrotic_decay:       f32,
    pub diffusion_rate_o2:    f32,
    pub diffusion_rate_glu:   f32,
    pub diffusion_rate_chemo: f32,

    pub oxygen_boundary_value:  f32,
    pub glu_boundary_value:     f32,
    pub chemo_source_level:     f32,
    pub repulsion_stiffness:    f32,

    pub adhesion_stiffness: f32,
    pub adhesion_range:     f32,
    pub damping_coeff:      f32,
    pub vessel_radius:      f32,

    pub vessel_center_world:   [f32; 3],
    pub request_count:          u32,
    pub live_agent_count_base:  u32,
    pub _pad1:                  [u8; 12],

    pub domain_center: [f32; 3],
    pub _pad2:         [u8; 4],

    pub domain_min:       [f32; 3],
    pub oxygen_grid_size: f32,

    pub domain_max:       [f32; 3],
    pub spatial_grid_dim: i32,

    pub spatial_voxel_size:  f32,
    pub spatial_voxel_count: i32,
    pub num_species:         i32,
    pub pad:                 i32,
}

impl Default for GlobalUniforms {
    fn default() -> Self {
        let half = GRID_DIM as f32 * 0.5;
        Self {
            delta_time: 0.05, step_count: 0, total_agents: 512, initial_agents: 512,
            necrotic_decay: 50.0, diffusion_rate_o2: 0.10, diffusion_rate_glu: 0.005,
            diffusion_rate_chemo: 0.05, oxygen_boundary_value: 1.0, glu_boundary_value: 1.0,
            chemo_source_level: 0.5, repulsion_stiffness: 15.0, adhesion_stiffness: 1.0,
            adhesion_range: 0.4, damping_coeff: 1.0, vessel_radius: 6.0,
            vessel_center_world: [0.0; 3], request_count: 0, live_agent_count_base: 0,
            _pad1: [0; 12], domain_center: [0.0; 3], _pad2: [0; 4],
            domain_min: [-half, -half, -half], oxygen_grid_size: GRID_DIM as f32,
            domain_max: [half, half, half], spatial_grid_dim: GRID_DIM as i32,
            spatial_voxel_size: 1.0, spatial_voxel_count: VOXEL_COUNT as i32,
            num_species: 1, pad: 0,
        }
    }
}

// ─── ресурс конфигурации (живёт в App + Extract → RenderApp) ────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SimMode {
    Fixed(u32), // Целевая скорость (шагов/сек)
    Max,        // Headless / Турбо-режим
}

#[derive(Resource, Clone, ExtractResource)]
pub struct SimConfig {
    pub uniforms:      GlobalUniforms,
    pub species:       Vec<SpeciesParam>,
    pub running:       bool,
    pub mode:          SimMode,
    pub steps_to_run:  u32,
    pub step_count:    u32,
    pub render_buffer: Handle<ShaderStorageBuffer>,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            uniforms:      GlobalUniforms::default(),
            species:       vec![SpeciesParam::default()],
            running:       true,
            mode:          SimMode::Fixed(144),
            steps_to_run:  0,
            step_count:    0,
            render_buffer: Handle::default(),
        }
    }
}

// ─── все GPU-буферы и текстуры симуляции ────────────────────────────────────

/// Все GPU-ресурсы симуляции.  
/// Создаётся один раз в `SimPlugin::finish` и живёт в `RenderApp`.
#[derive(Resource)]
pub struct SimBuffers {
    // uniforms
    pub uniform_buf: Buffer,

    // агент-буферы (пинг-понг A/B)
    pub cells_a: Buffer,
    pub cells_b: Buffer,

    // таблица видов и матрица взаимодействий
    pub species_table:      Buffer,
    pub interaction_matrix: Buffer,

    // пространственная сетка
    pub spatial_count: Buffer,
    pub spatial_slots:  Buffer,

    // метрики
    pub metrics_accum: Buffer,
    pub metrics_out:   Buffer,

    // диффузионные поля (пинг-понг A/B)
    pub field_tex_a: Texture,
    pub field_tex_b: Texture,
    pub field_sampled_a: TextureView,
    pub field_sampled_b: TextureView,
    pub field_storage_a: TextureView,
    pub field_storage_b: TextureView,

    // маска окружения + заглушка для init-прохода
    pub env_mask_tex:   Texture,
    pub env_mask_read:  TextureView,
    pub env_mask_write: TextureView,

    pub dummy_env_mask_tex:   Texture,
    pub dummy_env_mask_read:  TextureView,
    pub dummy_env_mask_write: TextureView,

    pub field_sampler: Sampler,

    // состояние пинг-понга (атомики для thread-safety в render graph)
    pub field_ping: AtomicBool,
    pub cells_ping: AtomicBool,
}

impl SimBuffers {
    pub fn new(device: &RenderDevice) -> Self {
        // ── вспомогательные замыкания ──────────────────────────────────────

        let make_buf = |label: &'static str, size: u64, usage: BufferUsages| {
            device.create_buffer(&BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
        };

        let field_size = Extent3d { width: GRID_DIM, height: GRID_DIM, depth_or_array_layers: GRID_DIM };

        let make_field_tex = |label: &'static str| {
            device.create_texture(&TextureDescriptor {
                label: Some(label), size: field_size,
                mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D3,
                format: TextureFormat::Rgba32Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
        };

        let rgba_view = |tex: &Texture, label: &'static str| {
            tex.create_view(&TextureViewDescriptor {
                label: Some(label),
                format:    Some(TextureFormat::Rgba32Float),
                dimension: Some(TextureViewDimension::D3),
                ..default()
            })
        };

        // ── буферы ────────────────────────────────────────────────────────

        let cell_size = size_of::<CellData>() as u64;
        let cells_usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;

        let uniform_buf      = make_buf("sim::uniform",      size_of::<GlobalUniforms>() as u64, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let cells_a          = make_buf("sim::cells_a",      MAX_AGENTS as u64 * cell_size,       cells_usage);
        let cells_b          = make_buf("sim::cells_b",      MAX_AGENTS as u64 * cell_size,       cells_usage);
        let species_table    = make_buf("sim::species_table",MAX_SPECIES as u64 * size_of::<SpeciesParam>() as u64, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let interaction_matrix = make_buf("sim::interaction_matrix",(MAX_SPECIES * MAX_SPECIES) as u64 * 4, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let spatial_count    = make_buf("sim::spatial_count",VOXEL_COUNT as u64 * 4,              BufferUsages::STORAGE);
        let spatial_slots    = make_buf("sim::spatial_slots", VOXEL_COUNT as u64 * MAX_CELL_PER_VOXEL as u64 * 4, BufferUsages::STORAGE);
        let metrics_accum    = make_buf("sim::metrics_accum",METRICS_ACCUM_COUNT as u64 * 4,     BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let metrics_out = make_buf("sim::metrics_out", METRICS_OUT_COUNT as u64 * 4, BufferUsages::STORAGE | BufferUsages::COPY_SRC);

        // ── текстуры полей ────────────────────────────────────────────────

        let field_tex_a = make_field_tex("sim::field_tex_a");
        let field_tex_b = make_field_tex("sim::field_tex_b");

        let field_sampled_a = rgba_view(&field_tex_a, "sim::sampled_a");
        let field_sampled_b = rgba_view(&field_tex_b, "sim::sampled_b");
        let field_storage_a = rgba_view(&field_tex_a, "sim::storage_a");
        let field_storage_b = rgba_view(&field_tex_b, "sim::storage_b");

        // ── маска окружения ───────────────────────────────────────────────

        let r32_view = |tex: &Texture, label: &'static str| {
            tex.create_view(&TextureViewDescriptor {
                label: Some(label),
                format:    Some(TextureFormat::R32Sint),
                dimension: Some(TextureViewDimension::D3),
                ..default()
            })
        };

        let env_mask_tex = device.create_texture(&TextureDescriptor {
            label: Some("sim::env_mask"), size: field_size,
            mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D3,
            format: TextureFormat::R32Sint,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let env_mask_read  = r32_view(&env_mask_tex, "sim::env_mask_read");
        let env_mask_write = r32_view(&env_mask_tex, "sim::env_mask_write");

        let dummy_size = Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
        let dummy_env_mask_tex = device.create_texture(&TextureDescriptor {
            label: Some("sim::dummy_env_mask"), size: dummy_size,
            mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D3,
            format: TextureFormat::R32Sint,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let dummy_env_mask_read  = r32_view(&dummy_env_mask_tex, "sim::dummy_read");
        let dummy_env_mask_write = r32_view(&dummy_env_mask_tex, "sim::dummy_write");

        // ── сэмплер ──────────────────────────────────────────────────────

        let field_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("sim::field_sampler"),
            mag_filter: FilterMode::Linear, min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            ..default()
        });

        Self {
            uniform_buf, cells_a, cells_b, species_table, interaction_matrix,
            spatial_count, spatial_slots, metrics_accum, metrics_out,
            field_tex_a, field_tex_b,
            field_sampled_a, field_sampled_b, field_storage_a, field_storage_b,
            env_mask_tex, env_mask_read, env_mask_write,
            dummy_env_mask_tex, dummy_env_mask_read, dummy_env_mask_write,
            field_sampler,
            field_ping: AtomicBool::new(true),
            cells_ping: AtomicBool::new(true),
        }
    }
}

// ─── unit-тесты ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── размеры структур ──────────────────────────────────────────────────

    /// CellData должен быть ровно 64 байта (16 × f32), выравнивание 4.
    #[test]
    fn cell_data_size_is_64_bytes() {
        assert_eq!(size_of::<CellData>(), 64,
            "CellData size mismatch: WGSL ожидает 64 байта (std430)");
    }

    /// SpeciesParam — 64 байта.
    #[test]
    fn species_param_size_is_64_bytes() {
        assert_eq!(size_of::<SpeciesParam>(), 64,
            "SpeciesParam size mismatch");
    }

    /// GlobalUniforms — кратно 16 байтам (выравнивание uniform-буфера).
    #[test]
    fn global_uniforms_aligned_to_16() {
        assert_eq!(size_of::<GlobalUniforms>() % 16, 0,
            "GlobalUniforms должен быть кратен 16 для std140/uniform layout");
    }

    // ── Pod / Zeroable ────────────────────────────────────────────────────

    /// bytemuck::cast_slice не должен паниковать.
    #[test]
    fn cell_data_pod_cast() {
        let cells = vec![CellData::default(); 4];
        let bytes = bytemuck::cast_slice::<CellData, u8>(&cells);
        assert_eq!(bytes.len(), 4 * 64);
    }

    #[test]
    fn species_param_pod_cast() {
        let sp = SpeciesParam::default();
        let bytes = bytemuck::bytes_of(&sp);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn global_uniforms_pod_cast() {
        let u = GlobalUniforms::default();
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), size_of::<GlobalUniforms>());
    }

    // ── константы и default-значения ──────────────────────────────────────

    #[test]
    fn voxel_count_correct() {
        assert_eq!(VOXEL_COUNT, 64 * 64 * 64);
    }

    #[test]
    fn global_uniforms_default_domain_symmetry() {
        let u = GlobalUniforms::default();
        let half = GRID_DIM as f32 * 0.5;
        assert_eq!(u.domain_min, [-half, -half, -half]);
        assert_eq!(u.domain_max, [half,  half,  half]);
    }

    #[test]
    fn species_param_default_base_divide_time() {
        let sp = SpeciesParam::default();
        assert!((sp.base_divide_time - 10.0).abs() < f32::EPSILON);
    }

    // ── workgroup dispatch math ───────────────────────────────────────────

    /// Диспетчеризация агентов делится без остатка.
    #[test]
    fn agent_dispatch_256_no_remainder() {
        assert_eq!(MAX_AGENTS % 256, 0,
            "MAX_AGENTS должен делиться на 256 для dispatch_workgroups(MAX_AGENTS/256, 1, 1)");
    }

    #[test]
    fn agent_dispatch_128_no_remainder() {
        assert_eq!(MAX_AGENTS % 128, 0);
    }

    /// Диспетчеризация сетки делится без остатка.
    #[test]
    fn grid_dispatch_8_no_remainder() {
        assert_eq!(GRID_DIM % 8, 0,
            "GRID_DIM должен делиться на 8 для dispatch_workgroups(GRID_DIM/8, ...)");
    }

    #[test]
    fn voxel_dispatch_256_no_remainder() {
        assert_eq!(VOXEL_COUNT % 256, 0);
    }

    // ── SimConfig ─────────────────────────────────────────────────────────

    #[test]
    fn sim_config_default_running() {
        let cfg = SimConfig::default();
        assert!(cfg.running);
        assert_eq!(cfg.step_count, 0);
        assert_eq!(cfg.species.len(), 1);
    }
}
