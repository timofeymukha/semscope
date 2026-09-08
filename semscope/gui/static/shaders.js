// GLSL ES 3.00 shaders.  The spectral basis is evaluated on the GPU: the
// vertex shader maps (r, s) of a tessellated reference element to physical
// space through the polynomial geometry, and the fragment shader evaluates the
// field's Lagrange expansion at every pixel.

export const MAXN = 32;

const COMMON = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform int uN;                 // GLL nodes per direction
uniform int uEPR;               // elements per texture row
uniform float uNodes[${MAXN}];  // GLL nodes
uniform float uBary[${MAXN}];   // barycentric weights

// Lagrange basis l_i(t), i < uN, by the barycentric formula (exact at nodes).
void basis(float t, out float L[${MAXN}]) {
  float d[${MAXN}];
  int hit = -1;
  for (int i = 0; i < uN; i++) {
    d[i] = t - uNodes[i];
    if (abs(d[i]) < 1e-7) hit = i;
  }
  if (hit >= 0) {
    for (int i = 0; i < uN; i++) L[i] = (i == hit) ? 1.0 : 0.0;
    return;
  }
  float S = 0.0;
  for (int i = 0; i < uN; i++) { float w = uBary[i] / d[i]; L[i] = w; S += w; }
  float inv = 1.0 / S;
  for (int i = 0; i < uN; i++) L[i] *= inv;
}

// texel of node (i, j) of element e in the flat (nelv, n, n) atlas
ivec2 texel(int e, int i, int j) {
  return ivec2((e % uEPR) * uN * uN + j * uN + i, e / uEPR);
}
`;

export const SURFACE_VS = COMMON + `
in vec2 aRS;
in uint aElem;
uniform sampler2D uX;
uniform sampler2D uY;
uniform vec2 uCenter;   // view centre (relative to the domain centre)
uniform vec2 uScale;    // 2 * pixels-per-unit / canvas size
out vec2 vRS;
flat out uint vElem;

void main() {
  float Lr[${MAXN}], Ls[${MAXN}];
  basis(aRS.x, Lr);
  basis(aRS.y, Ls);
  int e = int(aElem);
  vec2 p = vec2(0.0);
  for (int j = 0; j < uN; j++) {
    vec2 acc = vec2(0.0);
    for (int i = 0; i < uN; i++) {
      ivec2 t = texel(e, i, j);
      acc += vec2(texelFetch(uX, t, 0).r, texelFetch(uY, t, 0).r) * Lr[i];
    }
    p += acc * Ls[j];
  }
  vec2 q = (p - uCenter) * uScale;
  gl_Position = vec4(q, 0.0, 1.0);
  vRS = aRS;
  vElem = aElem;
}
`;

export const SURFACE_FS = COMMON + `
in vec2 vRS;
flat in uint vElem;
uniform sampler2D uField;
uniform sampler2D uCmap;
uniform vec2 uRange;
uniform int uMode;            // 0 = spectral, 1 = nodal/linear
uniform int uContours;        // number of iso-intervals (0 = off)
uniform float uContourWidth;
uniform vec4 uContourColor;
uniform int uEdges;
uniform float uEdgeWidth;
uniform vec4 uEdgeColor;
uniform vec4 uNanColor;
uniform int uInvert;
out vec4 fragColor;

float spectral(int e, vec2 rs) {
  float Lr[${MAXN}], Ls[${MAXN}];
  basis(rs.x, Lr);
  basis(rs.y, Ls);
  float v = 0.0;
  for (int j = 0; j < uN; j++) {
    float acc = 0.0;
    for (int i = 0; i < uN; i++) acc += texelFetch(uField, texel(e, i, j), 0).r * Lr[i];
    v += acc * Ls[j];
  }
  return v;
}

// bilinear interpolation inside the GLL sub-cell that contains (r, s)
float nodal(int e, vec2 rs) {
  int i0 = 0, j0 = 0;
  for (int i = 0; i < uN - 1; i++) if (rs.x >= uNodes[i]) i0 = i;
  for (int j = 0; j < uN - 1; j++) if (rs.y >= uNodes[j]) j0 = j;
  float a = clamp((rs.x - uNodes[i0]) / (uNodes[i0 + 1] - uNodes[i0]), 0.0, 1.0);
  float b = clamp((rs.y - uNodes[j0]) / (uNodes[j0 + 1] - uNodes[j0]), 0.0, 1.0);
  float f00 = texelFetch(uField, texel(e, i0, j0), 0).r;
  float f10 = texelFetch(uField, texel(e, i0 + 1, j0), 0).r;
  float f01 = texelFetch(uField, texel(e, i0, j0 + 1), 0).r;
  float f11 = texelFetch(uField, texel(e, i0 + 1, j0 + 1), 0).r;
  return mix(mix(f00, f10, a), mix(f01, f11, a), b);
}

// anti-aliased line coverage for a pixel distance d (px) and line width w (px)
float line(float d, float w) { return 1.0 - smoothstep(0.5 * w - 0.6, 0.5 * w + 0.6, d); }

void main() {
  int e = int(vElem);
  float v = (uMode == 0) ? spectral(e, vRS) : nodal(e, vRS);
  vec3 col;
  if (v != v) {
    col = uNanColor.rgb;
  } else {
    float t = (v - uRange.x) / (uRange.y - uRange.x);
    float tc = clamp(t, 0.0, 1.0);
    if (uInvert == 1) tc = 1.0 - tc;
    col = texture(uCmap, vec2(tc, 0.5)).rgb;
    if (uContours > 0) {
      float q = t * float(uContours);
      float w = fwidth(q);
      float d = (0.5 - abs(fract(q) - 0.5)) / max(w, 1e-6);
      float a = line(d, uContourWidth) * uContourColor.a;
      // fade contours where the field is nearly flat relative to the pixel size
      col = mix(col, uContourColor.rgb, a);
    }
  }
  if (uEdges == 1) {
    float er = (1.0 - abs(vRS.x)) / max(fwidth(vRS.x), 1e-9);
    float es = (1.0 - abs(vRS.y)) / max(fwidth(vRS.y), 1e-9);
    float a = line(min(er, es), uEdgeWidth) * uEdgeColor.a;
    col = mix(col, uEdgeColor.rgb, a);
  }
  fragColor = vec4(col, 1.0);
}
`;

export const PICK_FS = COMMON + `
in vec2 vRS;
flat in uint vElem;
out vec4 fragColor;
void main() {
  uint e = vElem + 1u;
  fragColor = vec4(float(e & 255u) / 255.0, float((e >> 8) & 255u) / 255.0, float((e >> 16) & 255u) / 255.0, 1.0);
}
`;

export const NODES_VS = COMMON + `
in uint aNode;    // per-vertex node index k = j*n + i
in uint aElem;    // per-instance element
uniform sampler2D uX;
uniform sampler2D uY;
uniform vec2 uCenter;
uniform vec2 uScale;
uniform float uSize;
void main() {
  int e = int(aElem);
  int k = int(aNode);
  ivec2 t = ivec2((e % uEPR) * uN * uN + k, e / uEPR);
  vec2 p = vec2(texelFetch(uX, t, 0).r, texelFetch(uY, t, 0).r);
  gl_Position = vec4((p - uCenter) * uScale, 0.0, 1.0);
  gl_PointSize = uSize;
}
`;

export const NODES_FS = `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 fragColor;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c) * 2.0;
  float a = 1.0 - smoothstep(0.8, 1.0, d);
  if (a <= 0.0) discard;
  fragColor = vec4(uColor.rgb, uColor.a * a);
}
`;
