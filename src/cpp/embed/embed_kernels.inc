// Floating-point kernels of the brisc PaCMAP / LocalMAP port, included twice by
// embed.cpp: once with AOF_KATTR empty (portable build, std::fma is a library
// call on CPUs without FMA) and, on x86 GCC/clang, once with
// AOF_KATTR = __attribute__((target("avx2,fma"))) so the fused multiply-adds
// compile to single instructions.  Both copies compute bit-identical results:
// every operation below is an IEEE-754 single-precision operation (or an
// explicitly fused multiply-add), in exactly the order brisc's compiled
// extension executes it.
//
// Ported from brisc/embed.pyx (Copyright (c) 2026 Michael Wainberg and Keon
// Arbabi, MIT License); see embed.cpp for the notice.  Part of
// anndataoom-fast, GPL-3.0-or-later.
//
// Expected macros: AOF_KATTR (function attribute), AOF_KNAME(name) (suffixes
// the kernel name).

// --- Adam (brisc `update_cell_embedding_adam`), fused with the gradient ---
// The gradient of cell i only reads the previous embedding, so the update is
// written to a second buffer (a.emb_out) and the buffers are swapped after
// each iteration.  Per coordinate:
//   momentum += (1 - beta1) * (g - momentum)      -> fma(g - m, 1 - beta1, m)
//   velocity += (1 - beta2) * (g * g - velocity)  -> fma(fma(g, g, -v), 1 - beta2, v)
//   embedding -= lr_t * momentum / (sqrt(velocity) + 1e-7)
AOF_KATTR static inline void AOF_KNAME(adam_cell)(const GradArgs& a, int64_t i, float g0,
                                                  float g1) {
    const float g[2] = {g0, g1};
    for (int c = 0; c < 2; c++) {
        const int64_t x = 2 * i + c;
        const float m = AOF_FMAF(g[c] - a.momentum[x], a.one_minus_beta1, a.momentum[x]);
        const float v = AOF_FMAF(AOF_FMAF(g[c], g[c], -a.velocity[x]), a.one_minus_beta2,
                                 a.velocity[x]);
        a.momentum[x] = m;
        a.velocity[x] = v;
        a.emb_out[x] = a.embedding[x] - (a.learning_rate * m) / (AOF_SQRTF(v) + 1e-7f);
    }
}

// --- gradients (brisc `get_gradient`) --------------------------------------
// brisc is compiled with -Ofast; GCC rewrites `10 + (1 + dx*dx + dy*dy)` as
// fma(dy, dy, fma(dx, dx, 11)) and accumulates `gradient += w * d` as a fused
// multiply-add.  The same holds for the mid-near (10001) and further (2) terms.
AOF_KATTR static void AOF_KNAME(gradients)(const GradArgs& a, int64_t begin, int64_t end) {
    const float* emb = a.embedding;
    for (int64_t i = begin; i < end; i++) {
        const float e0 = emb[2 * i], e1 = emb[2 * i + 1];
        float g0 = 0.0f, g1 = 0.0f;
        for (uint64_t p = a.nn_ptr[i], pe = a.nn_ptr[i + 1]; p < pe; p++) {
            if (p + 8 < pe) AOF_PREFETCH(emb + 2 * uint64_t(a.nn_idx[p + 8]));
            const uint64_t j = a.nn_idx[p];
            const float dx = e0 - emb[2 * j], dy = e1 - emb[2 * j + 1];
            const float s = AOF_FMAF(dy, dy, AOF_FMAF(dx, dx, 11.0f));
            const float w = (20.0f / (s * s)) * a.w_neighbors;
            g0 = AOF_FMAF(w, dx, g0);
            g1 = AOF_FMAF(w, dy, g1);
        }
        if (a.w_mid_near != 0.0f) {
            for (uint64_t p = a.mn_ptr[i], pe = a.mn_ptr[i + 1]; p < pe; p++) {
                if (p + 8 < pe) AOF_PREFETCH(emb + 2 * uint64_t(a.mn_idx[p + 8]));
                const uint64_t j = a.mn_idx[p];
                const float dx = e0 - emb[2 * j], dy = e1 - emb[2 * j + 1];
                const float s = AOF_FMAF(dy, dy, AOF_FMAF(dx, dx, 10001.0f));
                const float w = (20000.0f / (s * s)) * a.w_mid_near;
                g0 = AOF_FMAF(w, dx, g0);
                g1 = AOF_FMAF(w, dy, g1);
            }
        }
        for (uint64_t p = a.fp_ptr[i], pe = a.fp_ptr[i + 1]; p < pe; p++) {
            if (p + 8 < pe) AOF_PREFETCH(emb + 2 * uint64_t(a.fp_idx[p + 8]));
            const uint64_t j = a.fp_idx[p];
            const float dx = e0 - emb[2 * j], dy = e1 - emb[2 * j + 1];
            const float s = AOF_FMAF(dy, dy, AOF_FMAF(dx, dx, 2.0f));
            const float w = 2.0f / (s * s);
            g0 = AOF_FMAF(-w, dx, g0);
            g1 = AOF_FMAF(-w, dy, g1);
        }
        AOF_KNAME(adam_cell)(a, i, g0, g1);
    }
}

// --- LocalMAP phase-3 gradients (brisc `get_scaled_gradient`) --------------
// `half_max_distance / sqrt(distance)` is compiled to an RSQRTSS estimate
// refined by one Newton-Raphson step: y = (-0.5 r) * fma(d r, r, -3).
AOF_KATTR static void AOF_KNAME(scaled_gradients)(const GradArgs& a, int64_t begin, int64_t end) {
    const float* emb = a.embedding;
    const float hmd = a.half_max_distance;
    for (int64_t i = begin; i < end; i++) {
        const float e0 = emb[2 * i], e1 = emb[2 * i + 1];
        float g0 = 0.0f, g1 = 0.0f;
        for (uint64_t p = a.nn_ptr[i], pe = a.nn_ptr[i + 1]; p < pe; p++) {
            if (p + 8 < pe) AOF_PREFETCH(emb + 2 * uint64_t(a.nn_idx[p + 8]));
            const uint64_t j = a.nn_idx[p];
            const float dx = e0 - emb[2 * j], dy = e1 - emb[2 * j + 1];
            const float s = AOF_FMAF(dx, dx, dy * dy);
            const float dist = s + 1.0f, t = s + 11.0f;
            const float q = 20.0f / (t * t);
            const float r = rsqrt_estimate(dist);
            const float y = (r * -0.5f) * AOF_FMAF(dist * r, r, -3.0f);
            const float w = y * (q * hmd);
            g0 = AOF_FMAF(w, dx, g0);
            g1 = AOF_FMAF(w, dy, g1);
        }
        for (uint64_t p = a.fp_ptr[i], pe = a.fp_ptr[i + 1]; p < pe; p++) {
            if (p + 8 < pe) AOF_PREFETCH(emb + 2 * uint64_t(a.fp_idx[p + 8]));
            const uint64_t j = a.fp_idx[p];
            const float dx = e0 - emb[2 * j], dy = e1 - emb[2 * j + 1];
            const float s = AOF_FMAF(dy, dy, AOF_FMAF(dx, dx, 2.0f));
            const float w = 2.0f / (s * s);
            g0 = AOF_FMAF(-w, dx, g0);
            g1 = AOF_FMAF(-w, dy, g1);
        }
        AOF_KNAME(adam_cell)(a, i, g0, g1);
    }
}

// --- squared distance between two PC rows, as brisc's vectorised loop ------
// distance = (x0 - y0)^2, then dims 1..d-1: GCC vectorises with 16 float lanes
// (one fused accumulator per lane), folds 16 -> 8 lanes, optionally runs one
// 8-lane step, folds 8 -> 4 -> 2 -> 1, adds the dim-0 term, and finishes the
// remaining (< 8) dims with scalar fused multiply-adds.
AOF_KATTR static inline float AOF_KNAME(pc_sq_dist)(const float* x, const float* y, uint64_t d) {
    float d0 = x[0] - y[0];
    d0 = d0 * d0;
    if (d <= 1) return d0;
    const uint64_t m = d - 1;  // dims 1..d-1
    const float* a = x + 1;
    const float* b = y + 1;
    float acc8[8];
    uint64_t pos = 0;
    float dist;
    if (m >= 16) {
        float acc16[16];
        for (int l = 0; l < 16; l++) acc16[l] = 0.0f;
        const uint64_t m16 = m & ~uint64_t(15);
        for (; pos < m16; pos += 16)
            for (int l = 0; l < 16; l++) {
                const float t = a[pos + l] - b[pos + l];
                acc16[l] = AOF_FMAF(t, t, acc16[l]);
            }
        for (int l = 0; l < 8; l++) acc8[l] = acc16[l + 8] + acc16[l];
        dist = hsum8(acc8) + d0;
    } else {
        for (int l = 0; l < 8; l++) acc8[l] = 0.0f;
        dist = d0;
    }
    if (m - pos >= 8) {
        for (int l = 0; l < 8; l++) {
            const float t = a[pos + l] - b[pos + l];
            acc8[l] = AOF_FMAF(t, t, acc8[l]);
        }
        pos += 8;
        dist = hsum8(acc8) + d0;
    }
    for (; pos < m; pos++) {
        const float t = a[pos] - b[pos];
        dist = AOF_FMAF(t, t, dist);
    }
    return dist;
}

// --- mid-near pairs (brisc `sample_mid_near_pairs`, parallel branch) -------
AOF_KATTR static void AOF_KNAME(mid_near)(const MidNearArgs& a, int64_t begin, int64_t end) {
    const uint32_t n = uint32_t(a.n);
    const uint32_t nmn = a.num_mid_near_pairs;
    uint32_t sampled[6];
    uint32_t closest = 0, second_closest = 0;
    for (int64_t ii = begin; ii < end; ii++) {
        const uint32_t i = uint32_t(ii);
        uint32_t* out = a.mid_near_pairs + uint64_t(i) * nmn;
        uint64_t state = pcg_srand(a.seed + i);
        const float* xi = a.PCs + uint64_t(i) * a.d;
        for (uint32_t j = 0; j < nmn; j++) {
            float smallest = FLT_MAX, second_smallest = FLT_MAX;
            for (int k = 0; k < 6; k++) {
                for (;;) {
                    const uint32_t s = pcg_randint(n, state);
                    if (s == i) continue;
                    bool dup = false;
                    for (int l = 0; l < k && !dup; l++) dup = s == sampled[l];
                    for (uint32_t l = 0; l < j && !dup; l++) dup = s == out[l];
                    if (dup) continue;
                    sampled[k] = s;
                    break;
                }
            }
            for (int k = 0; k < 6; k++) {
                const uint32_t s = sampled[k];
                const float dist = AOF_KNAME(pc_sq_dist)(xi, a.PCs + uint64_t(s) * a.d, a.d);
                if (dist < smallest) {
                    second_smallest = smallest;
                    second_closest = closest;
                    smallest = dist;
                    closest = s;
                } else if (dist < second_smallest) {
                    second_smallest = dist;
                    second_closest = s;
                }
            }
            out[j] = second_closest;
        }
    }
}

// --- LocalMAP re-sampling (brisc `sample_further_pairs_nearby`) ------------
AOF_KATTR static void AOF_KNAME(further_nearby)(const NearbyArgs& a, int64_t begin, int64_t end) {
    const uint32_t n = uint32_t(a.n);
    const uint32_t nfp = a.num_further_pairs, nnb = a.num_neighbors;
    const float* emb = a.embedding;
    for (int64_t ii = begin; ii < end; ii++) {
        const uint32_t i = uint32_t(ii);
        uint64_t state = pcg_srand(a.seed + i);
        uint64_t lookahead = state;  // prefetch candidates 16 draws ahead
        for (int k = 0; k < 16; k++) AOF_PREFETCH(emb + 2 * uint64_t(pcg_randint(n, lookahead)));
        const uint32_t* nb = a.neighbor_pairs + uint64_t(i) * nnb;
        uint32_t* fp = a.further_pairs + uint64_t(i) * nfp;
        const float e0 = emb[2 * uint64_t(i)], e1 = emb[2 * uint64_t(i) + 1];
        for (uint32_t j = 0; j < nfp; j++) {
            // Up to 100 trials; on failure the old further pair is kept.
            for (int count = 0; count < 100; count++) {
                const uint32_t s = pcg_randint(n, state);
                AOF_PREFETCH(emb + 2 * uint64_t(pcg_randint(n, lookahead)));
                if (s == i) continue;
                const float dx = e0 - emb[2 * uint64_t(s)], dy = e1 - emb[2 * uint64_t(s) + 1];
                if (AOF_FMAF(dx, dx, dy * dy) > a.max_distance_squared) continue;
                bool dup = false;
                for (uint32_t k = 0; k < nnb && !dup; k++) dup = s == nb[k];
                for (uint32_t k = 0; k < j && !dup; k++) dup = s == fp[k];
                if (dup) continue;
                fp[j] = s;
                break;
            }
        }
    }
}
