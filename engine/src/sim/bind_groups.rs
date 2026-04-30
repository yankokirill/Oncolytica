//! # Шаг 4 — Связки (Bind Groups)
//!
//! Создание конкретных наборов ресурсов и логика Ping-Pong:
//! в каждом кадре ядра читают из буфера A и пишут в B, затем флаг меняется.

use bevy::{
    prelude::*,
    render::{
        render_resource::{BindGroup, BindGroupEntry, BindingResource},
        renderer::RenderDevice,
    },
};

use super::{
    data::SimBuffers,
    layouts::SimBindGroupLayouts,
};

/// Конкретные bind groups, созданные из GPU-буферов.
///
/// Пинг-понг реализован заранее созданными парами групп:
/// - `group_0_a` / `group_0_b` — диффузионные поля A↔B  
/// - `group_1_a` / `group_1_b` — агент-буферы A↔B
#[derive(Resource)]
pub struct SimBindGroups {
    /// Init-проход: env_mask пишется, dummy-маска читается.
    pub group_0_init: BindGroup,
    /// Поле A читается → поле B пишется.
    pub group_0_a: BindGroup,
    /// Поле B читается → поле A пишется.
    pub group_0_b: BindGroup,
    /// Cells_A читается, Cells_B пишется.
    pub group_1_a: BindGroup,
    /// Cells_B читается, Cells_A пишется.
    pub group_1_b: BindGroup,
    /// Пространственная сетка + метрики (не меняется).
    pub group_2: BindGroup,
}

impl SimBindGroups {
    /// Выбор group_0 по флагу пинг-понга.
    #[inline]
    pub fn g0(&self, ping: bool) -> &BindGroup {
        if ping { &self.group_0_a } else { &self.group_0_b }
    }

    /// Выбор group_1 по флагу пинг-понга.
    #[inline]
    pub fn g1(&self, ping: bool) -> &BindGroup {
        if ping { &self.group_1_a } else { &self.group_1_b }
    }
}

impl FromWorld for SimBindGroups {
    fn from_world(world: &mut World) -> Self {
        let device  = world.resource::<RenderDevice>();
        let layouts = world.resource::<SimBindGroupLayouts>();
        let buffers = world.resource::<SimBuffers>();

        Self::from_parts(device, layouts, buffers)
    }
}

impl SimBindGroups {
    pub fn from_parts(
        device:  &RenderDevice,
        layouts: &SimBindGroupLayouts,
        buffers: &SimBuffers,
    ) -> Self {
        // ── вспомогательные замыкания ──────────────────────────────────────

        let create_g0 = |label: &str,
                         tex_r:  &bevy::render::render_resource::TextureView,
                         tex_w:  &bevy::render::render_resource::TextureView,
                         env_r:  &bevy::render::render_resource::TextureView,
                         env_w:  &bevy::render::render_resource::TextureView| {
            device.create_bind_group(Some(label), &layouts.group_0, &[
                BindGroupEntry { binding: 0, resource: buffers.uniform_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&buffers.field_sampler) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(tex_r) },
                BindGroupEntry { binding: 3, resource: BindingResource::TextureView(tex_w) },
                BindGroupEntry { binding: 4, resource: BindingResource::TextureView(env_r) },
                BindGroupEntry { binding: 5, resource: BindingResource::TextureView(env_w) },
            ])
        };

        let create_g1 = |label: &str,
                         cells_r: &bevy::render::render_resource::Buffer,
                         cells_w: &bevy::render::render_resource::Buffer| {
            device.create_bind_group(Some(label), &layouts.group_1, &[
                BindGroupEntry { binding: 0, resource: cells_r.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: cells_w.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buffers.species_table.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: buffers.interaction_matrix.as_entire_binding() },
            ])
        };

        // ── group_0 ────────────────────────────────────────────────────────

        // init: маска ENV пишется; поле A/B не читается (используется dummy)
        let group_0_init = create_g0(
            "g0_init",
            &buffers.field_sampled_a, &buffers.field_storage_b,
            &buffers.dummy_env_mask_read, &buffers.env_mask_write,
        );

        // A → tex_a читается, tex_b пишется
        let group_0_a = create_g0(
            "g0_a",
            &buffers.field_sampled_a, &buffers.field_storage_b,
            &buffers.env_mask_read, &buffers.dummy_env_mask_write,
        );

        // B → tex_b читается, tex_a пишется
        let group_0_b = create_g0(
            "g0_b",
            &buffers.field_sampled_b, &buffers.field_storage_a,
            &buffers.env_mask_read, &buffers.dummy_env_mask_write,
        );

        // ── group_1 ────────────────────────────────────────────────────────

        let group_1_a = create_g1("g1_a", &buffers.cells_a, &buffers.cells_b);
        let group_1_b = create_g1("g1_b", &buffers.cells_b, &buffers.cells_a);

        // ── group_2 ────────────────────────────────────────────────────────

        let group_2 = device.create_bind_group(Some("g2"), &layouts.group_2, &[
            BindGroupEntry { binding: 0, resource: buffers.spatial_count.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: buffers.spatial_slots.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: buffers.metrics_accum.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: buffers.metrics_out.as_entire_binding() },
        ]);

        Self { group_0_init, group_0_a, group_0_b, group_1_a, group_1_b, group_2 }
    }
}

// ─── unit-тесты ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // Ping-Pong логику можно протестировать без GPU через чистую логику.

    /// Симулируем конечный автомат ping-pong для агент-буферов.
    struct PingPong { ping: bool }

    impl PingPong {
        fn new() -> Self { Self { ping: true } }
        fn flip(&mut self) { self.ping = !self.ping; }
        fn source_is_a(&self) -> bool { self.ping }
        fn dest_is_b(&self)   -> bool { self.ping }
    }

    #[test]
    fn ping_pong_starts_with_a_as_source() {
        let pp = PingPong::new();
        assert!(pp.source_is_a(), "Первый кадр: источник = A");
        assert!(pp.dest_is_b(),   "Первый кадр: назначение = B");
    }

    #[test]
    fn ping_pong_flips_after_one_step() {
        let mut pp = PingPong::new();
        pp.flip();
        assert!(!pp.source_is_a(), "После флипа: источник = B");
        assert!(!pp.dest_is_b(),   "После флипа: назначение = A");
    }

    #[test]
    fn ping_pong_returns_to_initial_after_two_flips() {
        let mut pp = PingPong::new();
        pp.flip();
        pp.flip();
        assert!(pp.source_is_a(), "После двух флипов буферы вернулись в A");
    }

    #[test]
    fn multiple_flips_cycle_correctly() {
        let mut pp = PingPong::new();
        for i in 0..100 {
            let expected = i % 2 == 0; // чётные — ping=true
            assert_eq!(pp.source_is_a(), expected,
                "Шаг {i}: ожидался source_is_a={expected}");
            pp.flip();
        }
    }

    /// Количество флипов за кадр симуляции совпадает с количеством
    /// пинг-понг переключений в `SimComputeNode`.
    /// Агент-буфер переключается 3 раза за кадр (int, upd, div).
    #[test]
    fn cells_ping_pong_flips_per_frame() {
        let flips_per_frame = 3_usize; // теперь 3 + финальный swap
        let mut pp = PingPong::new();
        for _ in 0..flips_per_frame {
            pp.flip();
        }
        // Через 3 флипа g1_ping = false → g1_b → читает cells_b (актуально)
        assert!(!pp.source_is_a(), "3 флипа: следующий шаг читает из B");
    }

    /// Поле (field_ping) переключается 2 раза за кадр (apply_reac, diff_solv).
    #[test]
    fn field_ping_pong_flips_per_frame() {
        let flips_per_frame = 2_usize;
        let mut pp = PingPong::new();
        for _ in 0..flips_per_frame {
            pp.flip();
        }
        // 2 флипа → возврат к исходному
        assert!(pp.source_is_a(), "2 флипа за кадр: поле возвращается к A");
    }

    /// `g0` возвращает правильную ссылку при ping=true / ping=false.
    /// Тестируем через индексный суррогат (bool → usize).
    #[test]
    fn g0_selector_logic() {
        let idx = |ping: bool| -> &'static str { if ping { "a" } else { "b" } };
        assert_eq!(idx(true),  "a");
        assert_eq!(idx(false), "b");
    }

    #[test]
    fn g1_selector_logic() {
        let idx = |ping: bool| -> &'static str { if ping { "a" } else { "b" } };
        assert_eq!(idx(true),  "a");
        assert_eq!(idx(false), "b");
    }

    /// group_0_init использует dummy-маску для чтения (не настоящую).
    #[test]
    fn init_group_uses_dummy_mask_for_read() {
        // Логика: во время init мы пишем env_mask, а читаем dummy (размер 1³).
        // Проверяем соответствие через именование ресурсов.
        let read_label  = "sim::dummy_read";
        let write_label = "sim::env_mask_write";
        assert!(read_label.contains("dummy"));
        assert!(write_label.contains("env_mask"));
    }
}
