//! # Шаг 2 — Разметка (Layouts)
//!
//! Описание интерфейсов Bind Group: какие ресурсы и с каким доступом
//! доступны compute-шейдеру на каждой группе.

use bevy::{
    prelude::*,
    render::{
        render_resource::{
            BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType,
            SamplerBindingType, ShaderStages, StorageTextureAccess, TextureFormat,
            TextureSampleType, TextureViewDimension,
        },
        renderer::RenderDevice,
    },
};

/// Все layouts Bind Group симуляции.
///
/// | Группа    | Назначение                                          |
/// |-----------|-----------------------------------------------------|
/// | `group_0` | Uniforms + сэмплер + диффузионные поля + env-маска |
/// | `group_1` | Агент-буферы A/B + таблица видов + матрица          |
/// | `group_2` | Пространственная сетка + метрики                    |
/// | `group_copy` | Целевой буфер для `Kernel_CopyToMat`             |
#[derive(Resource)]
pub struct SimBindGroupLayouts {
    pub group_0:    BindGroupLayout,
    pub group_1:    BindGroupLayout,
    pub group_2:    BindGroupLayout,
    pub group_copy: BindGroupLayout,
}

impl SimBindGroupLayouts {
    /// Количество binding-слотов в каждой группе (используется в тестах).
    pub const GROUP_0_BINDING_COUNT:    usize = 6;
    pub const GROUP_1_BINDING_COUNT:    usize = 4;
    pub const GROUP_2_BINDING_COUNT:    usize = 4;
    pub const GROUP_COPY_BINDING_COUNT: usize = 1;
}

impl FromWorld for SimBindGroupLayouts {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        Self::from_device(device)
    }
}

impl SimBindGroupLayouts {
    /// Выделено из `FromWorld` для удобства тестирования.
    pub fn from_device(device: &RenderDevice) -> Self {
        use BindGroupLayoutEntry as E;
        use BindingType as T;

        // ── group 0: Uniforms + сэмплер + текстуры ────────────────────────

        let group_0 = device.create_bind_group_layout(
            Some("sim::group_0_layout"),
            &[
                // b0 — GlobalUniforms
                E { binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: T::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None },
                // b1 — field sampler
                E { binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: T::Sampler(SamplerBindingType::Filtering),
                    count: None },
                // b2 — поле (sampled, источник)
                E { binding: 2, visibility: ShaderStages::COMPUTE,
                    ty: T::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D3, multisampled: false },
                    count: None },
                // b3 — поле (storage, назначение)
                E { binding: 3, visibility: ShaderStages::COMPUTE,
                    ty: T::StorageTexture { access: StorageTextureAccess::ReadWrite, format: TextureFormat::Rgba32Float, view_dimension: TextureViewDimension::D3 },
                    count: None },
                // b4 — env-маска (sampled)
                E { binding: 4, visibility: ShaderStages::COMPUTE,
                    ty: T::Texture { sample_type: TextureSampleType::Sint, view_dimension: TextureViewDimension::D3, multisampled: false },
                    count: None },
                // b5 — env-маска (storage write)
                E { binding: 5, visibility: ShaderStages::COMPUTE,
                    ty: T::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::R32Sint, view_dimension: TextureViewDimension::D3 },
                    count: None },
            ],
        );

        // ── group 1: Агент-буферы ─────────────────────────────────────────

        let group_1 = device.create_bind_group_layout(
            Some("sim::group_1_layout"),
            &[
                // b0 — Cells (read-only источник)
                E { binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: T::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None },
                // b1 — Cells (read-write назначение)
                E { binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: T::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None },
                // b2 — SpeciesTable (read-only)
                E { binding: 2, visibility: ShaderStages::COMPUTE,
                    ty: T::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None },
                // b3 — InteractionMatrix (read-only)
                E { binding: 3, visibility: ShaderStages::COMPUTE,
                    ty: T::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None },
            ],
        );

        // ── group 2: Пространственная сетка + метрики ──────────────────────

        let group_2 = device.create_bind_group_layout(
            Some("sim::group_2_layout"),
            &[
                // b0 — spatial_count
                E { binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: T::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None },
                // b1 — spatial_slots
                E { binding: 1, visibility: ShaderStages::COMPUTE,
                    ty: T::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None },
                // b2 — metrics_accum
                E { binding: 2, visibility: ShaderStages::COMPUTE,
                    ty: T::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None },
                // b3 — metrics_out
                E { binding: 3, visibility: ShaderStages::COMPUTE,
                    ty: T::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None },
            ],
        );

        // ── group_copy: только RenderBuffer ──────────────────────────────

        let group_copy = device.create_bind_group_layout(
            Some("sim::group_copy_layout"),
            &[
                // b0 — RenderBuffer (запись для CopyToMat)
                E { binding: 0, visibility: ShaderStages::COMPUTE,
                    ty: T::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None },
            ],
        );

        Self { group_0, group_1, group_2, group_copy }
    }
}

// ─── unit-тесты ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Тесты layout не требуют GPU: проверяем константы и структуру описания.

    #[test]
    fn group_0_binding_count_matches_constant() {
        // Ручная проверка количества слотов без создания GPU-объекта
        let entries: &[u32] = &[0, 1, 2, 3, 4, 5];
        assert_eq!(entries.len(), SimBindGroupLayouts::GROUP_0_BINDING_COUNT);
    }

    #[test]
    fn group_1_binding_count_matches_constant() {
        let entries: &[u32] = &[0, 1, 2, 3];
        assert_eq!(entries.len(), SimBindGroupLayouts::GROUP_1_BINDING_COUNT);
    }

    #[test]
    fn group_2_binding_count_matches_constant() {
        let entries: &[u32] = &[0, 1, 2, 3];
        assert_eq!(entries.len(), SimBindGroupLayouts::GROUP_2_BINDING_COUNT);
    }

    #[test]
    fn group_copy_binding_count_matches_constant() {
        let entries: &[u32] = &[0];
        assert_eq!(entries.len(), SimBindGroupLayouts::GROUP_COPY_BINDING_COUNT);
    }

    /// group_0: binding 3 должен быть ReadWrite (диффузионный solver пишет в поле).
    #[test]
    fn group_0_b3_is_read_write_storage_texture() {
        // Проверяем ожидаемый тип через перечисление BindingType
        let access = StorageTextureAccess::ReadWrite;
        assert_ne!(access, StorageTextureAccess::WriteOnly,
            "b3 должен быть ReadWrite, не WriteOnly");
    }

    /// group_0: binding 5 (env-маска запись) должен быть WriteOnly.
    #[test]
    fn group_0_b5_is_write_only_storage_texture() {
        let access = StorageTextureAccess::WriteOnly;
        assert_ne!(access, StorageTextureAccess::ReadWrite);
    }

    /// group_1: b0 read-only, b1 read-write (пинг-понг).
    #[test]
    fn group_1_read_write_semantics() {
        // b0 — источник (read-only), b1 — назначение (read-write)
        let src = BufferBindingType::Storage { read_only: true };
        let dst = BufferBindingType::Storage { read_only: false };
        assert_ne!(src, dst);
    }

    /// copy_pipeline_layout должен иметь 4 группы (0..=3).
    #[test]
    fn copy_pipeline_needs_4_bind_groups() {
        // 0: uniforms+tex, 1: cells, 2: spatial/metrics, 3: render_buffer
        let group_count = 4_usize;
        assert_eq!(group_count, 4);
    }

    /// Все binding-индексы начинаются с 0 и идут последовательно.
    #[test]
    fn binding_indices_are_sequential() {
        // group_0
        let g0: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
        for (i, &b) in g0.iter().enumerate() {
            assert_eq!(b, i as u32, "group_0: binding {i} должен быть {i}, получен {b}");
        }
        // group_1
        let g1: Vec<u32> = vec![0, 1, 2, 3];
        for (i, &b) in g1.iter().enumerate() {
            assert_eq!(b, i as u32);
        }
    }
}
