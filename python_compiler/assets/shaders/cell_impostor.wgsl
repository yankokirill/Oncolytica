#import bevy_pbr::mesh_view_bindings as mesh_bindings
#import bevy_pbr::mesh_functions as mesh_funcs

struct CellData {
    position:       vec3<f32>,
    speciesID:      i32,
    customParams:   vec3<f32>,
    state:          i32,
    aliveAge:       f32,
    deadAge:        f32,
    energy:         f32,
    prolifCapacity: i32,
    rngState:       u32,
    _reserved:      vec3<i32>,
}

struct ImpostorParams {
    scale: f32,
    necrotic_decay_time: f32,
}

const STATE_DELETED:       i32 = 0;
const STATE_PROLIFERATING: i32 = 1;
const STATE_QUIESCENT:     i32 = 2;
const STATE_NECROTIC:      i32 = 3;

@group(2) @binding(0) var<storage, read> cells: array<CellData>;
@group(2) @binding(1) var<uniform> params: ImpostorParams;

const QUAD = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0,  1.0), vec2<f32>(-1.0,  1.0)
);

struct VertexOutput {
    @builtin(position)  position_cs: vec4<f32>,
    @location(0)        ray_origin:  vec3<f32>,
    @location(1)        ray_dir:     vec3<f32>,
    @location(2)        sphere_pos:  vec3<f32>,
    @location(3)        radius:      f32,
    @location(4)        color:       vec4<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
    var out: VertexOutput;

    let cell_idx   = vertex_idx / 6u;
    let corner_idx = vertex_idx % 6u;
    let cell       = cells[cell_idx];

    if (cell.state == STATE_DELETED) {
        out.position_cs = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    var r = params.scale;
    if (cell.state == STATE_NECROTIC) {
        let t = clamp(cell.deadAge / max(0.001, params.necrotic_decay_time), 0.0, 1.0);
        r = mix(params.scale, params.scale * 0.7, t);
    }

    var color = vec4<f32>(0.2, 0.8, 0.2, 1.0);
    if (cell.state == STATE_NECROTIC) {
        color = vec4<f32>(0.18, 0.15, 0.15, 1.0);
    } else if (cell.state == STATE_QUIESCENT) {
        color = vec4<f32>(0.1, 0.4, 0.1, 1.0);
    }

    let view_matrix = mesh_bindings::view.view_from_world;
    let center_vs   = (view_matrix * vec4<f32>(cell.position, 1.0)).xyz;
    let offset      = QUAD[corner_idx] * r;
    let vertex_vs   = center_vs + vec3<f32>(offset, 0.0);

    let proj_matrix = mesh_bindings::view.clip_from_view;
    out.position_cs = proj_matrix * vec4<f32>(vertex_vs, 1.0);
    out.ray_origin  = vertex_vs;
    out.ray_dir     = vertex_vs;
    out.sphere_pos  = center_vs;
    out.radius      = r;
    out.color       = color;
    return out;
}

struct FragmentOutput {
    @location(0)        color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    let rd = normalize(in.ray_dir);
    let ro = vec3<f32>(0.0);
    let oc = ro - in.sphere_pos;
    let b  = dot(oc, rd);
    let c  = dot(oc, oc) - in.radius * in.radius;
    let h  = b * b - c;
    if (h < 0.0) { discard; }

    let t      = -b - sqrt(h);
    let hit_vs = ro + rd * t;
    let normal = normalize(hit_vs - in.sphere_pos);

    let proj_matrix = mesh_bindings::view.clip_from_view;
    let hit_cs      = proj_matrix * vec4<f32>(hit_vs, 1.0);
    out.depth       = hit_cs.z / hit_cs.w;

    let light_dir = normalize(vec3<f32>(0.5, 0.5, -1.0));
    let ndotl     = clamp(dot(normal, -light_dir), 0.0, 1.0);
    out.color     = vec4<f32>(in.color.rgb * (ndotl + 0.2), 1.0);
    return out;
}
