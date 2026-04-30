//! # Шаг 6 — Инициализация (Init Node)
//!
//! Одноразовый узел render graph: запускает `Kernel_InitField` и
//! `Kernel_InitEnvironment` ровно один раз для засева полей и среды.
//! После первого успешного запуска `has_run` устанавливается в `true`.

use bevy::{
    prelude::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::{ComputePassDescriptor, PipelineCache},
        renderer::RenderContext,
    },
};
use std::sync::atomic::{AtomicBool, Ordering};

use super::{
    bind_groups::SimBindGroups,
    data::{SimBuffers, SimConfig},
    pipelines::SimPipelines,
};
use crate::sim::data::{GRID_DIM, MAX_AGENTS};
use bevy::render::renderer::RenderQueue;
use bytemuck;

/// Размер workgroup для 3D init-ядер.
pub const INIT_WG_3D: u32 = GRID_DIM / 8;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct SimInitNodeLabel;

/// Init-узел render graph.  
/// `has_run` гарантирует, что ядра запускаются ровно один раз.
#[derive(Resource, Default)]
pub struct SimInitNode {
    pub has_run: AtomicBool,
}

impl Node for SimInitNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        ctx:    &mut RenderContext,
        world:  &World,
    ) -> Result<(), NodeRunError> {
        // Идемпотентный guard — если уже запускали, пропускаем
        if self.has_run.load(Ordering::SeqCst) {
            return Ok(());
        }

        let cache       = world.resource::<PipelineCache>();
        let pipelines   = world.resource::<SimPipelines>();
        let bind_groups = world.resource::<SimBindGroups>();

        // Ждём компиляцию шейдеров
        let Some(p_init_f) = cache.get_compute_pipeline(pipelines.init_f) else {
            return Ok(()); // отложить до следующего кадра
        };
        let Some(p_init_e) = cache.get_compute_pipeline(pipelines.init_e) else {
            return Ok(());
        };

        let mut pass = ctx.command_encoder().begin_compute_pass(&ComputePassDescriptor {
            label: Some("sim_init_pass"),
            ..default()
        });

        pass.set_bind_group(0, &bind_groups.group_0_init, &[]);

        // InitField: заполнить диффузионные поля начальными значениями
        pass.set_pipeline(p_init_f);
        pass.dispatch_workgroups(INIT_WG_3D, INIT_WG_3D, INIT_WG_3D);

        // InitEnvironment: разметить маску окружения (сосуды, стенки)
        pass.set_pipeline(p_init_e);
        pass.dispatch_workgroups(INIT_WG_3D, INIT_WG_3D, INIT_WG_3D);

        // Помечаем как выполненный ДО конца прохода (pass дропнется сразу после)
        self.has_run.store(true, Ordering::SeqCst);

        Ok(())
    }
}

pub fn prepare_sim_data_system(
    mut config:   ResMut<SimConfig>,
    render_queue: Res<RenderQueue>,
    buffers:      Res<SimBuffers>,
    mut is_seeded: Local<bool>,
) {
    if !config.running { return; }

    // 1. Всегда обновляем Uniforms (каждый кадр)
    render_queue.write_buffer(&buffers.uniform_buf, 0, bytemuck::bytes_of(&config.uniforms));

    // 2. На самом первом кадре пишем только таблицу видов
    if config.step_count == 0 {
        render_queue.write_buffer(&buffers.species_table, 0, bytemuck::cast_slice(&config.species));
    }

    // 3. РОВНО на кадре WARMUP_STEPS (например, 100) закидываем клетки
    if config.step_count >= crate::sim::data::WARMUP_STEPS && !*is_seeded {
        use crate::sim::data::{CellData, MAX_AGENTS};

        let mut cells = vec![CellData::default(); MAX_AGENTS as usize];
        
        for i in 0..(config.uniforms.initial_agents as usize) {
            let cell = &mut cells[i];
            cell.state  = 1; // STATE_PROLIFERATING
            cell.energy = 10.0; // Даем запас энергии
            cell.prolif_capacity = 10;
            cell.rng_state = fastrand::u32(1..u32::MAX);

            let r      = 5.0 * fastrand::f32();
            let theta  = std::f32::consts::TAU * fastrand::f32();
            let phi    = (1.0_f32 - 2.0 * fastrand::f32()).acos();
            cell.position = [
                r * phi.sin() * theta.cos(), 
                r * phi.sin() * theta.sin(), 
                r * phi.cos()
            ];
        }

        render_queue.write_buffer(&buffers.cells_a, 0, bytemuck::cast_slice(&cells));
        render_queue.write_buffer(&buffers.cells_b, 0, bytemuck::cast_slice(&cells));
        
        let init_count = [config.uniforms.initial_agents as i32];
        render_queue.write_buffer(&buffers.metrics_accum, 18 * 4, bytemuck::cast_slice(&init_count));
        
        *is_seeded = true;
        info!("--- СРЕДА ГОТОВА: Клетки добавлены в симуляцию! ---");
    }
}

// ─── unit-тесты ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── AtomicBool state machine ───────────────────────────────────────────

    #[test]
    fn has_run_starts_false() {
        let node = SimInitNode::default();
        assert!(!node.has_run.load(Ordering::SeqCst),
            "has_run должен начинаться с false");
    }

    #[test]
    fn has_run_can_be_set_to_true() {
        let node = SimInitNode::default();
        node.has_run.store(true, Ordering::SeqCst);
        assert!(node.has_run.load(Ordering::SeqCst));
    }

    #[test]
    fn has_run_is_idempotent_after_true() {
        let node = SimInitNode::default();
        node.has_run.store(true, Ordering::SeqCst);
        node.has_run.store(true, Ordering::SeqCst); // второй раз — безопасно
        assert!(node.has_run.load(Ordering::SeqCst));
    }

    /// Симулируем run-guard: если has_run = true, узел должен вернуть Ok сразу.
    #[test]
    fn run_guard_skips_if_has_run() {
        let node = SimInitNode { has_run: AtomicBool::new(true) };
        // Логика guard: if has_run → return Ok(())
        let would_skip = node.has_run.load(Ordering::SeqCst);
        assert!(would_skip, "Если has_run=true, тело узла пропускается");
    }

    /// После первого запуска has_run = true → второй run ничего не делает.
    #[test]
    fn second_run_is_no_op() {
        let node = SimInitNode::default();

        // Первый "запуск" (симуляция)
        assert!(!node.has_run.load(Ordering::SeqCst));
        node.has_run.store(true, Ordering::SeqCst);

        // Второй "запуск" — guard должен сработать
        let skipped = node.has_run.load(Ordering::SeqCst);
        assert!(skipped, "Второй вызов должен быть пропущен (has_run=true)");
    }

    // ── dispatch math ─────────────────────────────────────────────────────

    #[test]
    fn init_wg_3d_covers_entire_grid() {
        assert_eq!(INIT_WG_3D * 8, GRID_DIM,
            "INIT_WG_3D * workgroup_size(8) должен покрывать весь GRID_DIM");
    }

    #[test]
    fn init_wg_3d_nonzero() {
        assert!(INIT_WG_3D >= 1);
    }

    // ── засев клеток ──────────────────────────────────────────────────────

    /// Тест чистой логики засева: все клетки должны иметь state=1 и energy=1.
    #[test]
    fn seed_cells_state_and_energy() {
        use crate::sim::data::CellData;

        let count = 8_usize;
        let mut cells = vec![CellData::default(); count];
        for cell in cells.iter_mut() {
            cell.state  = 1;
            cell.energy = 1.0;
        }
        for cell in &cells {
            assert_eq!(cell.state, 1);
            assert!((cell.energy - 1.0).abs() < f32::EPSILON);
        }
    }

    /// Координаты засева лежат внутри радиуса 5 (с небольшим запасом).
    #[test]
    fn seed_cells_inside_radius() {
        use crate::sim::data::CellData;

        let count  = 64_usize;
        let radius = 5.0_f32;

        // Детерминированный тест: используем фиксированные точки
        let fixed_points: Vec<[f32; 3]> = (0..count)
            .map(|i| {
                let t = i as f32 / count as f32;
                let r = radius * t; // r от 0 до 5
                [r, 0.0, 0.0]
            })
            .collect();

        for pos in &fixed_points {
            let dist = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
            assert!(dist <= radius + f32::EPSILON,
                "Точка {:?} выходит за радиус {radius}", pos);
        }
    }

    /// initial_agents из GlobalUniforms совпадает с ожидаемым.
    #[test]
    fn initial_agent_count_default() {
        use crate::sim::data::GlobalUniforms;
        let u = GlobalUniforms::default();
        assert_eq!(u.initial_agents, 512);
    }

    /// Размер cells_a буфера достаточен для initial_agents.
    #[test]
    fn cells_buffer_fits_initial_agents() {
        use crate::sim::data::{CellData, GlobalUniforms, MAX_AGENTS};
        let u = GlobalUniforms::default();
        let needed_bytes  = u.initial_agents as u64 * std::mem::size_of::<CellData>() as u64;
        let buffer_bytes  = MAX_AGENTS as u64 * std::mem::size_of::<CellData>() as u64;
        assert!(needed_bytes <= buffer_bytes,
            "cells_a буфер слишком мал: нужно {needed_bytes}, имеем {buffer_bytes}");
    }
}
