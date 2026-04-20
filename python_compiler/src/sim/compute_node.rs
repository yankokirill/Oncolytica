//! # Шаг 5 — Узел графа (Main Node)

use bevy::{
    prelude::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::{
            BindGroupEntry, ComputePass, ComputePassDescriptor, ComputePipeline, PipelineCache,
        },
        renderer::RenderContext,
        render_asset::RenderAssets,
        storage::GpuShaderStorageBuffer,
    },
};
use std::sync::atomic::Ordering;

use super::{
    bind_groups::SimBindGroups,
    data::{SimBuffers, SimConfig},
    layouts::SimBindGroupLayouts,
    readback::StagingBuffers,
};

use crate::sim::METRICS_INTERVAL_FRAMES;
use crate::sim::data::{GRID_DIM, MAX_AGENTS, VOXEL_COUNT, WARMUP_STEPS};

// Размеры групп
pub const WG_AGENTS_256: u32 = MAX_AGENTS / 256;
pub const WG_AGENTS_128: u32 = MAX_AGENTS / 128;
pub const WG_VOXELS: u32     = VOXEL_COUNT / 256;
pub const WG_GRID_3D: u32    = GRID_DIM / 8;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct SimComputeNodeLabel;
pub struct SimComputeNode;

// ============================================================================
// 1. ВЫСОКОУРОВНЕВЫЙ ЦИКЛ ОБНОВЛЕНИЯ (GPU "UPDATE")
// ============================================================================

/// Аналог "Update" для GPU. Здесь задается строгий порядок вызова ядер.
fn execute_gpu_update(ctx: &mut ComputeContext, is_warmup: bool, should_collect_metrics: bool) {
    if is_warmup {
        ctx.dispatch(ctx.p.diff, WG_GRID_3D, WG_GRID_3D, WG_GRID_3D);
        ctx.swap_fields();
        return;
    }
    
    // --- СЕТКА ДЛЯ ПОЛЕЙ ---
    ctx.dispatch(ctx.p.spat_clr,     WG_VOXELS, 1, 1);
    ctx.dispatch(ctx.p.spat_ins_fld, WG_AGENTS_256, 1, 1);
    
    // --- ПОЛЯ И ДИФФУЗИЯ (Работает всегда) ---
    ctx.dispatch(ctx.p.reac, WG_GRID_3D, WG_GRID_3D, WG_GRID_3D);
    ctx.swap_fields(); // Пинг-понг полей
    
    ctx.dispatch(ctx.p.diff, WG_GRID_3D, WG_GRID_3D, WG_GRID_3D);
    ctx.swap_fields();
    
    // --- ПРОСТРАНСТВЕННАЯ СЕТКА И ВЗАИМОДЕЙСТВИЕ ---
    ctx.dispatch(ctx.p.spat_clr,     WG_VOXELS, 1, 1);
    ctx.dispatch(ctx.p.spat_ins_int, WG_AGENTS_256, 1, 1);
    
    ctx.dispatch(ctx.p.cell_int, WG_AGENTS_128, 1, 1);
    ctx.swap_cells();
    
    // --- МЕТАБОЛИЗМ И ДЕЛЕНИЕ ---
    ctx.dispatch(ctx.p.cell_upd, WG_AGENTS_256, 1, 1);
    ctx.swap_cells();
    
    ctx.dispatch(ctx.p.cell_div, WG_AGENTS_128, 1, 1);
    ctx.swap_cells();
    
    // --- МЕТРИКИ ---
    if (should_collect_metrics) {
        ctx.dispatch(ctx.p.metrics_clr, 1, 1, 1);
        ctx.dispatch(ctx.p.metrics_col, WG_AGENTS_256, 1, 1);
        ctx.dispatch(ctx.p.metrics_fin, 1, 1, 1);
    }
}

// ============================================================================
// 2. ОБЕРТКА ДЛЯ УДОБНОГО ВЫЗОВА ЯДЕР (DISPATCHER)
// ============================================================================

struct ComputeContext<'a> {
    pass:        ComputePass<'a>,
    p:           PipelineHandles<'a>,
    bind_groups: &'a SimBindGroups,
    g0_ping:     bool,
    g1_ping:     bool,
}

impl<'a> ComputeContext<'a> {
    /// Устанавливает ядро и сразу вызывает его (dispatch)
    #[inline]
    fn dispatch(&mut self, pipeline: &ComputePipeline, x: u32, y: u32, z: u32) {
        self.pass.set_pipeline(pipeline);
        self.pass.dispatch_workgroups(x, y, z);
    }

    /// Меняет местами буферы клеток (A <-> B)
    #[inline]
    fn swap_cells(&mut self) {
        self.g1_ping = !self.g1_ping;
        self.pass.set_bind_group(1, self.bind_groups.g1(self.g1_ping), &[]);
    }

    /// Меняет местами текстуры полей (A <-> B)
    #[inline]
    fn swap_fields(&mut self) {
        self.g0_ping = !self.g0_ping;
        self.pass.set_bind_group(0, self.bind_groups.g0(self.g0_ping), &[]);
    }
}

// ============================================================================
// 3. НИЗКОУРОВНЕВАЯ ИНТЕГРАЦИЯ С BEVY (RENDER GRAPH)
// ============================================================================

impl Node for SimComputeNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_ctx: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let cache       = world.resource::<PipelineCache>();
        let pipelines   = world.resource::<crate::sim::pipelines::SimPipelines>();
        let bind_groups = world.resource::<SimBindGroups>();
        let buffers     = world.resource::<SimBuffers>();
        let layouts     = world.resource::<SimBindGroupLayouts>();
        let staging     = world.resource::<StagingBuffers>();
        let config      = world.resource::<SimConfig>();

        // 1. Получаем скомпилированные шейдеры
        let Some(p) = PipelineHandles::resolve(cache, pipelines) else { return Ok(()) };

        let mut g0_ping = buffers.field_ping.load(Ordering::SeqCst);
        let mut g1_ping = buffers.cells_ping.load(Ordering::SeqCst);
        
        let mut metrics_were_collected_this_frame = false;
        let start_step = config.step_count.saturating_sub(config.steps_to_run);
        let is_warmup = start_step < WARMUP_STEPS;

        // --- ПРОХОД 1: Вычисления симуляции ---
        {
            let mut pass = render_ctx.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                label: Some("sim_main_pass"), ..default()
            });

            // Первоначальная установка бинд-групп
            pass.set_bind_group(0, bind_groups.g0(g0_ping), &[]);
            pass.set_bind_group(1, bind_groups.g1(g1_ping), &[]);
            pass.set_bind_group(2, &bind_groups.group_2, &[]);

            let mut ctx = ComputeContext {
                pass, p: p.clone(), bind_groups, g0_ping, g1_ping,
            };

            for i in 0..config.steps_to_run {
                let should_collect_metrics = (start_step + i)  % METRICS_INTERVAL_FRAMES == 0;
                metrics_were_collected_this_frame = metrics_were_collected_this_frame || should_collect_metrics;
                execute_gpu_update(&mut ctx, is_warmup, should_collect_metrics);
            }

            // Сохраняем состояние пинг-понга обратно
            g0_ping = ctx.g0_ping;
            g1_ping = ctx.g1_ping;
        } 

        // --- ПРОХОД 2 и 3: Копирование данных (Без изменений) ---
        if !is_warmup {
            // Копирование в материал рендера
            let storage_assets = world.resource::<RenderAssets<GpuShaderStorageBuffer>>();
            if let Some(mat_buf) = storage_assets.get(&config.render_buffer) {
                let copy_bg = render_ctx.render_device().create_bind_group(
                    Some("sim::copy_bg"), &layouts.group_copy,
                    &[BindGroupEntry { binding: 0, resource: mat_buf.buffer.as_entire_binding() }],
                );

                let mut copy_pass = render_ctx.command_encoder().begin_compute_pass(&ComputePassDescriptor {
                    label: Some("sim_copy_pass"), ..default()
                });
                copy_pass.set_pipeline(p.copy);
                copy_pass.set_bind_group(0, bind_groups.g0(g0_ping), &[]);
                copy_pass.set_bind_group(1, bind_groups.g1(g1_ping), &[]);
                copy_pass.set_bind_group(2, &bind_groups.group_2, &[]);
                copy_pass.set_bind_group(3, &copy_bg, &[]);
                copy_pass.dispatch_workgroups(WG_AGENTS_256, 1, 1);
            }

            // 3. Чтение метрик
            // Копируем только если метрики были собраны в этом кадре и буфер свободен
            let staging_ready = staging.state.load(Ordering::SeqCst) == crate::sim::readback::STAGING_STATE_READY;
            
            if metrics_were_collected_this_frame && staging_ready {
                render_ctx.command_encoder().copy_buffer_to_buffer(
                    &buffers.metrics_out,
                    0,
                    &staging.metrics,
                    0,
                    crate::sim::readback::STAGING_BUFFER_BYTES,
                );
                staging.state.store(crate::sim::readback::STAGING_STATE_COPY_RECORDED, Ordering::SeqCst);
            }

        }

        buffers.field_ping.store(g0_ping, Ordering::SeqCst);
        buffers.cells_ping.store(g1_ping, Ordering::SeqCst);

        Ok(())
    }
}

#[derive(Clone)]
struct PipelineHandles<'a> {
    metrics_clr:  &'a ComputePipeline,
    metrics_col:  &'a ComputePipeline,
    metrics_fin:  &'a ComputePipeline,
    spat_clr:     &'a ComputePipeline,
    spat_ins_int: &'a ComputePipeline,
    spat_ins_fld: &'a ComputePipeline,
    cell_int:     &'a ComputePipeline,
    cell_upd:     &'a ComputePipeline,
    cell_div:     &'a ComputePipeline,
    reac:         &'a ComputePipeline,
    diff:         &'a ComputePipeline,
    copy:         &'a ComputePipeline,
}

impl<'a> PipelineHandles<'a> {
    fn resolve(cache: &'a PipelineCache, pl: &crate::sim::pipelines::SimPipelines) -> Option<Self> {
        Some(Self {
            metrics_clr:  cache.get_compute_pipeline(pl.metrics_clear)?,
            metrics_col:  cache.get_compute_pipeline(pl.metrics_col)?,
            metrics_fin:  cache.get_compute_pipeline(pl.metrics_fin)?,
            spat_clr:     cache.get_compute_pipeline(pl.spatial_clear)?,
            spat_ins_int: cache.get_compute_pipeline(pl.spatial_ins_int)?,
            spat_ins_fld: cache.get_compute_pipeline(pl.spatial_ins_fld)?,
            cell_int:     cache.get_compute_pipeline(pl.cell_int)?,
            cell_upd:     cache.get_compute_pipeline(pl.cell_upd)?,
            cell_div:     cache.get_compute_pipeline(pl.cell_div)?,
            reac:         cache.get_compute_pipeline(pl.apply_reac)?,
            diff:         cache.get_compute_pipeline(pl.diff_solv)?,
            copy:         cache.get_compute_pipeline(pl.copy_to_mat)?,
        })
    }
}
