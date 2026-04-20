//! # Шаг 3 — Конвейеры (Pipelines)
//!
//! Загрузка WGSL-шейдера и компиляция всех compute-ядер
//! (entry points) в `CachedComputePipelineId` через `PipelineCache`.

use bevy::{
    asset::AssetServer,
    prelude::*,
    render::{
        render_resource::{
            CachedComputePipelineId, ComputePipelineDescriptor, PipelineCache,
        },
    },
};
use std::borrow::Cow;

use super::layouts::SimBindGroupLayouts;

/// Список всех entry point-ов шейдера `tumor_sim.wgsl`.
/// Используется как единственный source-of-truth для строк.
pub mod entry_points {
    pub const INIT_FIELD:               &str = "Kernel_InitField";
    pub const INIT_ENVIRONMENT:         &str = "Kernel_InitEnvironment";
    pub const CLEAR_METRICS:            &str = "Kernel_ClearMetrics";
    pub const COLLECT_METRICS:          &str = "Kernel_CollectMetrics";
    pub const FINALIZE_METRICS:         &str = "Kernel_FinalizeMetrics";
    pub const SPATIAL_CLEAR:            &str = "SpatialGrid_Clear";
    pub const SPATIAL_INSERT_INTERACT:  &str = "SpatialGrid_Insert_Interaction";
    pub const SPATIAL_INSERT_FIELD:     &str = "SpatialGrid_Insert_Field";
    pub const CELL_INTERACTION:         &str = "Kernel_Interaction";
    pub const CELL_UPDATE:              &str = "Kernel_CellUpdate";
    pub const CELL_DIVISION:            &str = "Kernel_Division";
    pub const APPLY_REACTION:           &str = "Kernel_ApplyReaction";
    pub const DIFFUSION_SOLVER:         &str = "Kernel_DiffusionSolver";
    pub const COPY_TO_MAT:              &str = "Kernel_CopyToMat";

    /// Все entry-point строки в одном массиве (для тестирования уникальности).
    pub const ALL: &[&str] = &[
        INIT_FIELD, INIT_ENVIRONMENT,
        CLEAR_METRICS, COLLECT_METRICS, FINALIZE_METRICS,
        SPATIAL_CLEAR, SPATIAL_INSERT_INTERACT, SPATIAL_INSERT_FIELD,
        CELL_INTERACTION, CELL_UPDATE, CELL_DIVISION,
        APPLY_REACTION, DIFFUSION_SOLVER,
        COPY_TO_MAT,
    ];
}

/// Кэшированные ID всех compute-конвейеров симуляции.
#[derive(Resource)]
pub struct SimPipelines {
    // инициализация
    pub init_f: CachedComputePipelineId,
    pub init_e: CachedComputePipelineId,

    // метрики
    pub metrics_clear: CachedComputePipelineId,
    pub metrics_col:   CachedComputePipelineId,
    pub metrics_fin:   CachedComputePipelineId,

    // пространственная сетка
    pub spatial_clear:    CachedComputePipelineId,
    pub spatial_ins_int:  CachedComputePipelineId,
    pub spatial_ins_fld:  CachedComputePipelineId,

    // агент-ядра
    pub cell_int: CachedComputePipelineId,
    pub cell_upd: CachedComputePipelineId,
    pub cell_div: CachedComputePipelineId,

    // поля
    pub apply_reac: CachedComputePipelineId,
    pub diff_solv:  CachedComputePipelineId,

    // копирование в материал (4 группы)
    pub copy_to_mat: CachedComputePipelineId,
}

impl FromWorld for SimPipelines {
    fn from_world(world: &mut World) -> Self {
        let asset_server   = world.resource::<AssetServer>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let layouts        = world.resource::<SimBindGroupLayouts>();

        let shader: Handle<Shader> = asset_server.load("shaders/tumor_sim.wgsl");

        // Layout для инициализации (group 0)
        let init_layout = vec![layouts.group_0.clone()];

        // Стандартный layout (groups 0–2)
        let base_layout = vec![
            layouts.group_0.clone(),
            layouts.group_1.clone(),
            layouts.group_2.clone(),
        ];

        // Layout для CopyToMat (groups 0–3)
        let copy_layout = vec![
            layouts.group_0.clone(),
            layouts.group_1.clone(),
            layouts.group_2.clone(),
            layouts.group_copy.clone(),
        ];

        let queue = |ep: &'static str, layout: Vec<_>| -> CachedComputePipelineId {
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label:                        Some(Cow::Borrowed(ep)),
                layout,
                push_constant_ranges:         Vec::new(),
                shader:                       shader.clone(),
                shader_defs:                  Vec::new(),
                entry_point:                  Cow::Borrowed(ep),
                zero_initialize_workgroup_memory: false,
            })
        };

        let q = |ep: &'static str| queue(ep, base_layout.clone());

        use entry_points::*;
        Self {
            init_f: queue(INIT_FIELD, init_layout.clone()),
            init_e: queue(INIT_ENVIRONMENT, init_layout),
            metrics_clear: q(CLEAR_METRICS),
            metrics_col:   q(COLLECT_METRICS),
            metrics_fin:   q(FINALIZE_METRICS),
            spatial_clear:   q(SPATIAL_CLEAR),
            spatial_ins_int: q(SPATIAL_INSERT_INTERACT),
            spatial_ins_fld: q(SPATIAL_INSERT_FIELD),
            cell_int: q(CELL_INTERACTION),
            cell_upd: q(CELL_UPDATE),
            cell_div: q(CELL_DIVISION),
            apply_reac: q(APPLY_REACTION),
            diff_solv:  q(DIFFUSION_SOLVER),
            copy_to_mat: queue(COPY_TO_MAT, copy_layout),
        }
    }
}

// ─── unit-тесты ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::entry_points::*;

    /// Каждая строка entry point непустая.
    #[test]
    fn all_entry_points_non_empty() {
        for ep in ALL {
            assert!(!ep.is_empty(), "Entry point не должен быть пустой строкой");
        }
    }

    /// Все entry point строки уникальны.
    #[test]
    fn all_entry_points_unique() {
        let mut seen = std::collections::HashSet::new();
        for ep in ALL {
            assert!(seen.insert(*ep), "Дублирующийся entry point: {ep}");
        }
    }

    /// Ожидаемое общее количество ядер.
    #[test]
    fn total_kernel_count() {
        assert_eq!(ALL.len(), 14,
            "Ожидается 14 ядер: 2 init + 3 metrics + 3 spatial + 3 cells + 2 field + 1 copy");
    }

    /// Все ядра начинаются с заглавной буквы (соглашение WGSL).
    #[test]
    fn entry_points_start_with_capital() {
        for ep in ALL {
            let first = ep.chars().next().unwrap();
            assert!(first.is_uppercase(), "Entry point '{ep}' должен начинаться с заглавной буквы");
        }
    }

    /// CopyToMat — единственное ядро, использующее 4 группы.
    #[test]
    fn copy_to_mat_needs_extra_group() {
        // Остальные ядра используют 3 группы (0-2), CopyToMat — 4 (0-3).
        // Тестируем что строка корректна.
        assert_eq!(COPY_TO_MAT, "Kernel_CopyToMat");
    }

    /// Ядра инициализации объявлены с правильными именами.
    #[test]
    fn init_kernels_naming() {
        assert!(INIT_FIELD.contains("Field"));
        assert!(INIT_ENVIRONMENT.contains("Environment"));
    }

    /// Ядра пространственной сетки имеют префикс "SpatialGrid_".
    #[test]
    fn spatial_kernels_prefix() {
        for ep in &[SPATIAL_CLEAR, SPATIAL_INSERT_INTERACT, SPATIAL_INSERT_FIELD] {
            assert!(ep.starts_with("SpatialGrid_"), "'{ep}' должен начинаться с 'SpatialGrid_'");
        }
    }

    /// Основные вычислительные ядра имеют префикс "Kernel_".
    #[test]
    fn main_kernels_prefix() {
        let kernel_prefixed = ALL.iter().filter(|ep| ep.starts_with("Kernel_")).count();
        assert_eq!(kernel_prefixed, 11,
            "11 из 14 ядер должны иметь префикс Kernel_");
    }
}
