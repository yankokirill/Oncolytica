//! # Модуль симуляции опухоли
//!
//! Архитектура в 7 шагов:
//!
//! | Шаг | Файл              | Содержимое                                        |
//! |-----|-------------------|---------------------------------------------------|
//! | 1   | `data.rs`         | Структуры данных, константы, SimBuffers, SimConfig |
//! | 2   | `layouts.rs`      | BindGroupLayouts (интерфейсы шейдера)              |
//! | 3   | `pipelines.rs`    | CachedComputePipelineId + entry points             |
//! | 4   | `bind_groups.rs`  | BindGroups + ping-pong логика                      |
//! | 5   | `compute_node.rs` | SimComputeNode (dispatch каждый кадр)              |
//! | 6   | `init_node.rs`    | SimInitNode (одноразовая инициализация)            |
//! | 7   | `readback.rs`     | MetricsReadback + StagingBuffers                   |

pub mod data;
pub mod layouts;
pub mod pipelines;
pub mod bind_groups;
pub mod compute_node;
pub mod init_node;
pub mod readback;

// Переэкспортируем наиболее часто используемые типы
pub use data::{
    CellData, GlobalUniforms, SimConfig, SpeciesParam,
    MAX_AGENTS, GRID_DIM, VOXEL_COUNT, MAX_SPECIES,
};

// Псевдонимы структуры в том же пространстве имён для удобства
pub use data::SimBuffers;
pub use layouts::SimBindGroupLayouts;
pub use pipelines::SimPipelines;
pub use bind_groups::SimBindGroups;
pub use compute_node::SimComputeNode;
pub use init_node::SimInitNode;
pub use readback::{MetricsReadback, StagingBuffers};
pub use readback::{read_mapped_data_system, trigger_readback_system};

use crate::sim::data::{SimMode, METRICS_OUT_COUNT};
pub const METRICS_INTERVAL_FRAMES: u32 = 100;
pub const MAX_STEPS_PER_FRAME: u32 = 64;

use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResourcePlugin,
        render_graph::RenderGraph,
        renderer::RenderDevice,
        Render, RenderApp, RenderSet,
    },
};

use compute_node::SimComputeNodeLabel;
use init_node::{prepare_sim_data_system, SimInitNodeLabel};

use crate::sim::readback::MetricIndex;

/// Плагин симуляции — регистрирует все ресурсы и render graph узлы.
pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimConfig>()
           .init_resource::<MetricsReadback>()
           .add_plugins(ExtractResourcePlugin::<SimConfig>::default())
           .add_systems(Update, (
                simulation_rate_controller,
                collect_and_print_metrics_system.after(simulation_rate_controller),
            ));

        // Клонируем Arc<Mutex<...>>, чтобы RenderApp имел доступ к той же памяти
        let readback_clone = app.world().resource::<MetricsReadback>().clone();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else { return; };
        
        // Вставляем клон в мир рендера
        render_app.insert_resource(readback_clone);

        render_app.add_systems(
            Render,
            (
                prepare_sim_data_system.in_set(RenderSet::Prepare),
                read_mapped_data_system.in_set(RenderSet::Prepare),
                trigger_readback_system.in_set(RenderSet::Cleanup),
            ),
        );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);
        let device = render_app.world().resource::<RenderDevice>().clone();

        render_app
            .insert_resource(SimBuffers::new(&device))
            .insert_resource(StagingBuffers::new(&device))
            .init_resource::<SimBindGroupLayouts>()
            .init_resource::<SimPipelines>()
            .init_resource::<SimBindGroups>()
            .insert_resource(SimInitNode::default());

        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(SimInitNodeLabel, SimInitNode::default());
        graph.add_node(SimComputeNodeLabel, SimComputeNode);
        graph.add_node_edge(SimInitNodeLabel, SimComputeNodeLabel);
        graph.add_node_edge(
            SimComputeNodeLabel,
            bevy::render::graph::CameraDriverLabel,
        );
    }
}

fn simulation_rate_controller(
    time: Res<Time>,
    mut config: ResMut<SimConfig>,
    mut accumulator: Local<f32>,
) {
    if !config.running { return; }

    match config.mode {
        SimMode::Fixed(target_sps) => {
            *accumulator += time.delta_secs() * target_sps as f32;
            *accumulator = accumulator.min(MAX_STEPS_PER_FRAME as f32);

            let steps = (*accumulator as u32).min(MAX_STEPS_PER_FRAME);
            config.steps_to_run = steps;
            *accumulator -= steps as f32;
        }
        SimMode::Max => {
            config.steps_to_run = MAX_STEPS_PER_FRAME;
        }
    }

    config.step_count += config.steps_to_run;
    config.uniforms.step_count = config.step_count as i32;
}

fn collect_and_print_metrics_system(
    readback: Res<MetricsReadback>,
) {
    if let Some(data) = readback.take_data() {
        let step = data[MetricIndex::StepId as usize] as u32;

        // --- Категория: Популяция ---
        let total_live = data[MetricIndex::TotalLive as usize];
        let n_necr     = data[MetricIndex::NNecr as usize];
        let p_prolif   = data[MetricIndex::FracProlif as usize] * 100.0;
        let p_hypoxic  = data[MetricIndex::FracHypoxic as usize] * 100.0;

        // --- Категория: Среда и Энергия ---
        let mean_o2    = data[MetricIndex::MeanO2 as usize];
        let mean_glu   = data[MetricIndex::MeanGlu as usize];
        let mean_en    = data[MetricIndex::MeanEnergy as usize];

        // --- Категория: Морфология ---
        let r_inv      = data[MetricIndex::RInvasive as usize];
        let r_necr     = data[MetricIndex::RNecrotic as usize];
        let speed      = data[MetricIndex::MeanSpeed as usize];

        // 1. КОМПАКТНЫЙ ВЫВОД (Одной строкой для быстрого взгляда)
        info!("Step {:>5} | Live: {:>6} | O2: {:.2} | R: {:.1}", step, total_live, mean_o2, r_inv);

        // 2. КРАСИВЫЙ СТРУКТУРИРОВАННЫЙ ВЫВОД (Подробный)
        // Мы используем специальные символы для визуального разделения
        println!(
            "╔═════════════════════════════════════════════════════════════╗\n\
             ║  SIMULATION METRICS | STEP: {:<30}  ║\n\
             ╠═════════════════════════════════════════════════════════════╣\n\
             ║  POPULATION          │  ENVIRONMENT         │  SHAPE        ║\n\
             ║  Live:    {:>10.0} │  Mean O2:  {:>8.3}  │  R_inv: {:>5.1} ║\n\
             ║  Necro:   {:>10.0} │  Mean Glu: {:>8.3}  │  R_nec: {:>5.1} ║\n\
             ║  Prolif:  {:>9.1}% │  Energy:   {:>8.2}  │  Speed: {:>5.2} ║\n\
             ║  Hypoxic: {:>9.1}% │                      │               ║\n\
             ╚═════════════════════════════════════════════════════════════╝",
            step, total_live, mean_o2, r_inv, n_necr, mean_glu, r_necr, p_prolif, mean_en, speed, p_hypoxic
        ); // TODO: Step поточнее

    }
}
