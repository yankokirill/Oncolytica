Shader "TumorABM/TumorAgent"
{
    Properties
    {
        _Scale             ("MockCell Scale",          Float)  = 0.45
        _NecroticDecayTime ("Necrotic Decay Time", Float)  = 50.0
        [Toggle] _EnableClipping ("Enable Slicing",Float)  = 0
        _ClipPlaneOrigin   ("Clip Plane Origin",   Vector) = (0,0,0,0)
        _ClipPlaneNormal   ("Clip Plane Normal",   Vector) = (0,1,0,0)
    }

    SubShader
    {
        Tags
        {
            "RenderType"      = "Opaque"
            "RenderPipeline"  = "UniversalPipeline"
            "Queue"           = "Geometry"
            "DisableBatching" = "True"
        }

        Pass
        {
            Name "TumorAgent_Impostor"
            Tags { "LightMode" = "UniversalForward" }
            Cull Off
            ZWrite On
            ZTest LEqual

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex   Vert
            #pragma fragment Frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            #define STATE_PROLIFERATING 0
            #define STATE_QUIESCENT     1
            #define STATE_NECROTIC      2
            #define STATE_DELETED       3
            #define MAX_SPECIES         8

            struct CellData
            {
                float3 position;
                int    state;
                float  aliveAge;
                float  deadAge;
                int    speciesID;
                float  energy;
                int    prolifCapacity;
                float3 customParams;
                float4 reserved;
            };

            StructuredBuffer<CellData> _Cells;

            CBUFFER_START(UnityPerMaterial)
                float  _Scale;
                float  _NecroticDecayTime;
                float4 _SpeciesColors[MAX_SPECIES];
                float  _EnableClipping;
                float4 _ClipPlaneOrigin;
                float4 _ClipPlaneNormal;
            CBUFFER_END

            // Квад: 2 треугольника, 6 вершин
            static const float2 QUAD[6] =
            {
                float2(-1,-1), float2(-1, 1), float2( 1, 1),
                float2(-1,-1), float2( 1, 1), float2( 1,-1)
            };

            struct Varyings
            {
                float4 posCS      : SV_POSITION;
                float3 rayOrigin  : TEXCOORD0;  // луч из камеры
                float3 rayDir     : TEXCOORD1;  // направление луча
                float3 spherePos  : TEXCOORD2;  // центр сферы в view space
                float  radius     : TEXCOORD3;
                float4 color      : TEXCOORD4;
            };

            Varyings Vert(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
            {
                Varyings o;

                CellData cell = _Cells[instanceID];

                // Выбрасываем удалённые и отсечённые клетки
                bool dead = (cell.state == STATE_DELETED);

                if (!dead && _EnableClipping > 0.5)
                {
                    float dist = dot(cell.position - _ClipPlaneOrigin.xyz,
                                     normalize(_ClipPlaneNormal.xyz));
                    if (dist < 0.0) dead = true;
                }

                if (dead)
                {
                    o.posCS     = float4(0,0,1,0); // w=0 убивает примитив
                    o.rayOrigin = 0; o.rayDir = 0;
                    o.spherePos = 0; o.radius = 0; o.color = 0;
                    return o;
                }

                // Радиус некроза
                float radius = _Scale;
                if (cell.state == STATE_NECROTIC)
                {
                    float t = saturate(cell.deadAge / max(0.001, _NecroticDecayTime));
                    radius  = lerp(_Scale, _Scale * 0.7, t);
                }

                // Цвет
                int sp = clamp(cell.speciesID, 0, MAX_SPECIES - 1);
                float4 color;
                if      (cell.state == STATE_NECROTIC)   color = float4(0.18, 0.15, 0.15, 1);
                else if (cell.state == STATE_QUIESCENT)  color = _SpeciesColors[sp] * 0.4;
                else                                     color = _SpeciesColors[sp];

                // -------------------------------------------------------
                // IMPOSTOR: строим квад в view space
                // -------------------------------------------------------
                // Центр сферы в view space
                float3 centerVS = mul(UNITY_MATRIX_V, float4(cell.position, 1)).xyz;

                // Угол квада в view space (смещение по X и Y камеры)
                float2 offset   = QUAD[vertexID % 6] * radius;
                float3 vertexVS = centerVS + float3(offset, 0);

                // Итоговая позиция
                o.posCS = mul(UNITY_MATRIX_P, float4(vertexVS, 1));

                // Передаём данные для ray-cast во фрагмент
                // rayOrigin: позиция вершины квада в view space
                // rayDir:    направление от камеры к вершине (в view space)
                o.rayOrigin = vertexVS;
                o.rayDir    = vertexVS; // для перспективы: луч из (0,0,0) к вершине
                o.spherePos = centerVS;
                o.radius    = radius;
                o.color     = color;

                return o;
            }

            struct FragOut
            {
                float4 color : SV_Target;
                float  depth : SV_Depth;
            };

            FragOut Frag(Varyings i)
            {
                FragOut o;

                // -------------------------------------------------------
                // RAY-SPHERE INTERSECTION в view space
                // -------------------------------------------------------
                // Луч: P(t) = rayOrigin + t * normalize(rayDir)
                // Но для perspective: луч из (0,0,0) в направлении rayDir
                float3 rd = normalize(i.rayDir);
                float3 ro = float3(0, 0, 0); // камера в origin view space

                float3 oc = ro - i.spherePos;
                float  b  = dot(oc, rd);
                float  c  = dot(oc, oc) - i.radius * i.radius;
                float  h  = b * b - c;

                // Нет пересечения — отсекаем
                if (h < 0.0) discard;

                // Ближайшая точка пересечения
                float  t       = -b - sqrt(h);
                float3 hitVS   = ro + rd * t;       // точка в view space
                float3 normalVS = normalize(hitVS - i.spherePos); // нормаль

                // -------------------------------------------------------
                // DEPTH: пишем корректную глубину точки на поверхности сферы
                // -------------------------------------------------------
                float4 hitCS   = mul(UNITY_MATRIX_P, float4(hitVS, 1));
                o.depth        = hitCS.z / hitCS.w;

                // -------------------------------------------------------
                // ОСВЕЩЕНИЕ: Lambert + ambient + небольшой Phong
                // -------------------------------------------------------
                // Нормаль из view space в world space
                float3 normalWS = normalize(mul((float3x3)UNITY_MATRIX_I_V, normalVS));

                Light  light  = GetMainLight();
                float  NdotL  = saturate(dot(normalWS, light.direction));
                float3 viewWS = normalize(_WorldSpaceCameraPos -
                                          mul(UNITY_MATRIX_I_V, float4(hitVS, 1)).xyz);
                float3 halfWS = normalize(light.direction + viewWS);
                float  spec   = pow(saturate(dot(normalWS, halfWS)), 32.0) * 0.3;

                float3 lit = i.color.rgb * (light.color * NdotL + 0.2) + spec;

                o.color = float4(lit, 1.0);
                return o;
            }
            ENDHLSL
        }
    }
}
