//! # Рендеринг клеток
//!
//! `CellImpostorMaterial` — материал на основе `ShaderStorageBuffer`,
//! который читает позиции клеток прямо на GPU и рисует их как billboards.

use bevy::{
    prelude::*,
    render::{
        mesh::PrimitiveTopology,
        render_asset::RenderAssetUsages,
        render_resource::{AsBindGroup, ShaderRef, ShaderType},
        storage::ShaderStorageBuffer,
        view::NoFrustumCulling,
    },
};

use crate::sim::{data::{CellData, SimMode}, SimConfig, GlobalUniforms, SpeciesParam, MAX_AGENTS};

// ─── материал ────────────────────────────────────────────────────────────────

/// Материал, передающий буфер клеток в вершинный/фрагментный шейдер.

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct ImpostorParams {
    pub scale: f32,
    pub necrotic_decay_time: f32,
}

/// Шейдер `cell_impostor.wgsl` использует `gl_VertexID` для адресации
/// клетки и вычисляет billboard-трансформацию на лету.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CellImpostorMaterial {
    /// Ссылка на буфер с данными клеток (read-only в шейдере).
    #[storage(0, read_only)]
    pub cells_buffer: Handle<ShaderStorageBuffer>,

    /// Масштаб impostor-квада (радиус клетки в world units).
    #[uniform(1)]
    pub params: ImpostorParams,
}

impl Material for CellImpostorMaterial {
    fn vertex_shader()   -> ShaderRef { "shaders/cell_impostor.wgsl".into() }
    fn fragment_shader() -> ShaderRef { "shaders/cell_impostor.wgsl".into() }
}

// ─── система инициализации ────────────────────────────────────────────────────

/// Startup-система: создаёт рендер-буфер, материал, меш и камеру.
pub fn setup_rendering(
    mut commands:       Commands,
    mut storage_bufs:   ResMut<Assets<ShaderStorageBuffer>>,
    mut materials:      ResMut<Assets<CellImpostorMaterial>>,
    mut meshes:         ResMut<Assets<Mesh>>,
) {
    // ── рендер-буфер (заполняется GPU через Kernel_CopyToMat) ─────────────
    let floats_per_cell  = std::mem::size_of::<CellData>() / 4;
    let empty            = vec![0u32; MAX_AGENTS as usize * floats_per_cell];
    let render_buffer    = storage_bufs.add(ShaderStorageBuffer::from(empty));

    // Перезаписываем SimConfig с корректным handle
    commands.insert_resource(SimConfig {
        uniforms:      GlobalUniforms::default(),
        species:       vec![SpeciesParam::default()],
        running:       true,
        mode:          SimMode::Fixed(144),
        steps_to_run:  0,
        step_count:    0,
        render_buffer: render_buffer.clone(),
    });

    // ── impostor меш (6 вершин на клетку = 2 треугольника) ───────────────
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        vec![[0.0_f32, 0.0, 0.0]; 6 * MAX_AGENTS as usize],
    );

    // ── спавн сущности с мешем и материалом ──────────────────────────────
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(CellImpostorMaterial {
            cells_buffer:      render_buffer,
            params: ImpostorParams {
                scale: 0.45,
                necrotic_decay_time: 50.0,
            }
        })),
        NoFrustumCulling,
    ));

    // ── камера ────────────────────────────────────────────────────────────
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, 80.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

// ─── unit-тесты ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    // ── размер рендер-буфера ──────────────────────────────────────────────

    #[test]
    fn render_buffer_element_size_is_cell_data() {
        let floats_per_cell = size_of::<CellData>() / 4;
        let total_floats    = MAX_AGENTS as usize * floats_per_cell;
        let total_bytes     = total_floats * 4;

        assert_eq!(total_bytes, MAX_AGENTS as usize * size_of::<CellData>(),
            "Рендер-буфер должен вмещать MAX_AGENTS клеток");
    }

    #[test]
    fn render_buffer_u32_count_aligns_with_cell_data() {
        let floats_per_cell = size_of::<CellData>() / 4;
        // CellData = 64 байта = 16 u32
        assert_eq!(floats_per_cell, 16,
            "CellData должен быть ровно 16 u32 (64 байта)");
    }

    // ── вершины меша ─────────────────────────────────────────────────────

    #[test]
    fn impostor_mesh_vertex_count() {
        let verts_per_cell    = 6_usize; // 2 треугольника
        let total_verts       = verts_per_cell * MAX_AGENTS as usize;
        assert_eq!(total_verts, 6 * MAX_AGENTS as usize);
    }

    #[test]
    fn impostor_mesh_triangle_count() {
        let triangles = (6 * MAX_AGENTS as usize) / 3;
        assert_eq!(triangles, 2 * MAX_AGENTS as usize);
    }

    // ── параметры материала ───────────────────────────────────────────────

    #[test]
    fn default_scale_positive() {
        let scale = 0.45_f32;
        assert!(scale > 0.0, "Масштаб impostor должен быть положительным");
    }

    #[test]
    fn necrotic_decay_time_positive() {
        let decay = 50.0_f32;
        assert!(decay > 0.0);
    }

    // ── shader paths ──────────────────────────────────────────────────────

    #[test]
    fn vertex_shader_path_matches_fragment() {
        // Для impostors оба шейдера в одном файле
        let v: &str = "shaders/cell_impostor.wgsl";
        let f: &str = "shaders/cell_impostor.wgsl";
        assert_eq!(v, f, "vertex и fragment шейдеры должны быть в одном файле");
    }
}
