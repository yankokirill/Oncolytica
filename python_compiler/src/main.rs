//! # Tumor Simulation — точка входа
//!
//! Связывает все модули согласно 7-шаговой архитектуре:
//!
//! | Шаг | Модуль                    | Файл                    |
//! |-----|---------------------------|-------------------------|
//! | 1   | Подготовка данных          | `sim/data.rs`           |
//! | 2   | Разметка (Layouts)         | `sim/layouts.rs`        |
//! | 3   | Конвейеры (Pipelines)      | `sim/pipelines.rs`      |
//! | 4   | Связки (Bind Groups)       | `sim/bind_groups.rs`    |
//! | 5   | Узел графа (Main Node)     | `sim/compute_node.rs`   |
//! | 6   | Инициализация (Init Node)  | `sim/init_node.rs`      |
//! | 7   | Обратная связь (Readback)  | `sim/readback.rs`       |

#![allow(unused)]

mod sim;
mod rendering;

use bevy::{
    prelude::*,
    render::{
        render_resource::WgpuFeatures,
        settings::{RenderCreation, WgpuSettings},
        RenderPlugin,
    },
};

use sim::SimPlugin;
use rendering::{setup_rendering, CellImpostorMaterial};

use bevy::winit::WinitSettings;
use bevy::window::PresentMode;

fn main() {
    let headless = std::env::args().any(|a| a == "--headless");

    let mut app = App::new();

    if headless {
        // Режим без окна: только RenderApp крутит GPU
        app.add_plugins(
            DefaultPlugins
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                                | WgpuFeatures::FLOAT32_FILTERABLE,
                        ..default()
                    }),
                    ..default()
                })
                // Отключаем окно полностью
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: bevy::window::ExitCondition::DontExit,
                    ..default()
                }),
        )
        // Убираем ограничение framerate
        .insert_resource(WinitSettings {
            focused_mode:   bevy::winit::UpdateMode::Continuous,
            unfocused_mode: bevy::winit::UpdateMode::Continuous,
        });
    } else {
        app.add_plugins(
            DefaultPlugins
                .set(RenderPlugin {
                    render_creation: RenderCreation::Automatic(WgpuSettings {
                        features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                                | WgpuFeatures::FLOAT32_FILTERABLE,
                        ..default()
                    }),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        // Immediate убирает v-sync — GPU рендерит настолько быстро,
                        // насколько позволяет железо (без привязки к 60 Hz монитора)
                        present_mode: PresentMode::Immediate,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .insert_resource(WinitSettings {
            // В фокусе — без ограничений; не в фокусе — тоже без ограничений
            focused_mode:   bevy::winit::UpdateMode::Continuous,
            unfocused_mode: bevy::winit::UpdateMode::Continuous,
        });
    }

    app.add_plugins(SimPlugin)
       .add_plugins(MaterialPlugin::<CellImpostorMaterial>::default())
       .add_systems(Startup, setup_rendering)
       .run();
}
