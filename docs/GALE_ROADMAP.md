# GALE Roadmap: From First Principles

## v0.2 — 2026-06-22 (Rewrite from Deep Analysis)

---

## Prologue: The Truth

This document is not a plan to "improve code quality" or "increase test coverage."

It is a plan to answer ONE question:

> **Given a neural network trained with the GALE stack, what mathematical guarantees does the output have that no other architecture can provide?**

The answer today (June 2026):

| Guarantee | Status | Precision |
|-----------|--------|-----------|
| Shift invariance of bispectrum features | PROVED (FFTGroup.jl:153-167) | 2e-15 machine epsilon |
| Cn vs Dn discrimination via bispectrum | PROVED (FFTGroup.jl:210-232) | 100pp on time-reversal pairs |
| Exact recovery from bispectrum | PROVED (FFTGroup.jl:291-299) | 1e-15 when A_omega >> eps |
| Group-equivariant projection (Reynolds) | PROVED (QGroupENN.jl:20-32) | Floating-point exact |
| Everything else claimed | NOT PROVED | Placeholder |

The gap between what the README claims and what is actually proved is the entire content of this roadmap.

---

## Part 1: The Core Theorem (What Actually Works)

### Theorem (Bispectrum Shift Invariance)

Let `x in R^n`, `A in C^n` with `A_omega != 0` for all `omega`. Define:

```
z_omega = A_omega * FFT(x)_omega
B_z(omega) = z_omega * z_2 * conj(z_{mod(omega,n)+1})
B(x) = [Re(B_z), Im(B_z)] in R^{2n}
```

Then for any cyclic shift `t in Z`:

```
B(shift(x, t)) = B(x)    identically (exact arithmetic)
||B(shift(x, t)) - B(x)|| < 3e-15    (float64 arithmetic, n <= 1024)
```

### Theorem (Exact Recovery)

```
x_rec = IFFT(z_omega / A_omega)
||x - x_rec||_2 / ||x||_2 < 1e-15    when A_omega >> eps
```

### Theorem (Cn-Dn Discrimination)

For time-reversal pairs `(x, reflect(x))`:

```
B(reflect(x)) != B(x)    in general
accuracy_Cn - accuracy_Dn >= 100pp    on time-reversal data
```

### These three theorems are the ONLY mathematically guaranteed results in the entire codebase. Everything else is either:
- A correct implementation that lacks a proof (QGroupENN projections)
- A heuristic labeled as a guarantee (SymmetryCertificate PAC-Bayes)
- Placeholder code that does not deliver its claimed function (ScalableWDW)

---

## Part 2: The Guarantee Landscape (What Each Module Actually Proves)

### Tier 1: Proved (theorems with empirical verification)

| Module | Guarantee | Proof Status | Composes with FFTGroup? |
|--------|-----------|-------------|------------------------|
| FFTGroup.jl | Shift invariance, exact recovery, Cn-Dn gap | PROVED | N/A (anchor) |
| FFTPipeline.jl | End-to-end classification with bispectrum | VERIFIED (single run) | YES (uses bispec_features) |

### Tier 2: Correct Implementation, No Proof

| Module | Guarantee | Proof Status | Composes with FFTGroup? |
|--------|-----------|-------------|------------------------|
| QGroupENN.jl | Reynolds projection is group-equivariant | CORRECT code, NO formal proof | NO (different representation) |
| SymmetryDiscovery.jl | Bispectrum-based symmetry profiling | CORRECT use of FFTGroup | YES (calls bispec_features) |

### Tier 3: Heuristics Labeled as Guarantees

| Module | Claimed Guarantee | Actual Status | Gap |
|--------|------------------|---------------|-----|
| UnifiedWDW.jl | "Unified pipeline" | Never uses bispectrum | FFTGroup invariance absent |
| WDWAutoencoder.jl | "Algebraic invariance" | Uses autocorrelation, not bispectrum | Wrong invariant |
| SymmetryCertificate.jl | "7-pillar certificate" | Fake PAC-Bayes, fake MDL, fake hash | Catastrophic |
| RuptureABC.jl | "A/B/C rupture certification" | Fake p-values, fake MDL | Catastrophic |
| ScalableWDW.jl | "O(n log n) scaling" | None of the claims are implemented | Equivariance error measures wrong property |

### Tier 4: Experimental Math (Honest)

| Module | Status |
|--------|--------|
| Sheaves/FiniteSheaves.jl | 50-line stub, honest |
| Krylov/Complexity.jl | 91-line Lanczos, honest but minimal |
| Algebra/Quivers.jl | 143-line quiver toolkit, honest |
| All Tier 3 modules (< 100 LOC each) | Honest experimental math |

---

## Part 3: The Composition Problem (Why the Stack Fails as a Stack)

### The Critical Discovery

The project claims 6 layers that compose into a guarantee stack. In reality:

**Only 2 modules compose with FFTGroup's bispectrum invariance:**
- FFTPipeline (uses `combined_bispec_features`)
- SymmetryDiscovery (uses `bispec_features`)

**The other 4 "layers" operate on different representations:**
- QGroupENN: operates on signal permutations, not bispectrum
- UnifiedWDW: operates on raw signals via Reynolds projection
- WDWAutoencoder: operates on autocorrelation features
- Tensor/MERA: operates on signal time-domain

**The consequence:** The shift invariance guarantee of FFTGroup is LOST after passing through any non-composing module. There is no "guarantee stack" — there are parallel independent pipelines that happen to share a module namespace.

### Why Composition Fails Formally

For a guarantee G to compose through a pipeline `f ∘ g ∘ h`:

```
G(f(g(h(x)))) = G(x) for all x
```

This requires EACH function to either:
1. Preserve G (be a G-morphism), or
2. Be independent of the property G measures

FFTGroup's G is "bispectrum-based shift invariance." A function f preserves G iff:

```
B(f(shift(x))) = B(f(x))    for all shifts t
```

This is true for ANY function f that receives B(x) as input (because B(shift(x)) = B(x) by the core theorem). But it is FALSE for any function that operates on raw signals before computing B — unless that function commutes with cyclic shifts.

**Current pipeline brokenness:**

```
Pipeline A (FFTPipeline):  x → B(x) → classifier    GUARANTEE PRESERVED
Pipeline B (UnifiedWDW):   x → signal_projection → MERA → Krylov    GUARANTEE LOST
Pipeline C (WDWAutoencoder): x → autocorrelation → proj → MERA    GUARANTEE LOST
```

---

## Part 4: Phase 0 — Radical Honesty (Weeks 1-2)

Before building anything new, we must label what exists correctly.

### 4.1 Rename or Delete Misleading Modules

| Module | Action | Reason |
|--------|--------|--------|
| `ScalableWDW.jl` | **DELETE** | Every claim is false. Adaptive sampling is not adaptive. Equivariance error measures invariance. MERA truncation is array slicing. O(n log n) is O(n). |
| `SymmetryCertificate.jl` | **DELETE** | "Certificate" with fake PAC-Bayes, fake MDL, fake hash. Produces formatted lies. |
| `RuptureABC.jl` | **DELETE** | "Statistical significance" is hardcoded. "MDL" is parameter counting. |
| `UnifiedWDW.jl` | **RENAME** → `ExperimentalPipeline.jl` | Does not unify anything. The bispectrum is absent. Honest about being experimental. |
| `WDWAutoencoder.jl` | **DEMOTE** → Tier 3 | Claims algebraic invariance but uses autocorrelation. Working code, wrong name. |
| `Tensor/HolographicCodes.jl` | **RENAME** → `HaarWavelets.jl` | It's Haar wavelets, not MERA, not holographic codes. |
| `AutoSymmetryDiscovery.jl` | **DELETE dead code paths** | Fix LieGAN loss, remove intractable SVD, fix repair_symmetries bug. |
| `mlp_baseline.jl` | **DELETE** | Deprecated, redundant. |

### 4.2 Reclassify ALL modules by actual guarantee level

| Level | Label | Criterion | Modules |
|-------|-------|-----------|---------|
| G0 | No guarantee | Module exists, no proof | Sheaves, Krylov, Algebra, Motives, Time, Bio, Gravity, Vacuum, Planner, Logic, Semantics, Category |
| G1 | Heuristic | Works in practice, no bound | ExperimentalPipeline, HaarWavelets, AutoSymmetryFlux |
| G2 | Correct | Implementation is correct, no formal proof | QGroupENN, SymmetryDiscovery |
| G3 | Verified | Implementation + test suite confirms behavior | FFTPipeline |
| G4 | Proved | Mathematical theorem + machine-verified bounds | FFTGroup |

### 4.3 Rewrite the README to match reality

The current README claims "316/316 tests passing" and "4 verified breakthroughs." This is true but misleading because:
- 300 of those tests are concentrated in FFTGroup and FFTPipeline
- The other 38 modules have at most 16 tests combined
- The "scalability" breakthrough is for FFTGroup only, not the stack

The new README must state clearly:

> "The provable algebraic guarantees apply ONLY to the bispectrum features in FFTGroup.jl. All other modules are research prototypes at varying maturity. See GALE_ROADMAP.md for the complete status."

---

## Part 5: Phase 1 — Guarantee Composition (Weeks 3-8)

### 5.1 Define the Composition Interface

Every module that claims to be part of the "guarantee stack" must implement:

```julia
# A Guarantee is a pair: (invariance_property, error_bound)
abstract type Guarantee end

struct ShiftInvariance <: Guarantee
    max_error::Float64  # proven upper bound on ||B(shift(x)) - B(x)||
    n_max::Int          # maximum dimension where the bound holds
end

struct Equivariance{G} <: Guarantee
    group::G
    max_error::Float64
end

struct Composition{L, R} <: Guarantee
    left::L
    right::R
    # Theorem: if left holds on f(x) and right holds on g(x),
    # then composition holds on g(f(x)) with error <= left.error + right.error
end
```

### 5.2 Make Every Module a G-Morphism

A module M is a G-morphism if it preserves the guarantee G. For FFTGroup's shift invariance:

```
M is a shift-invariance morphism iff B(M(shift(x))) = B(M(x)) for all shifts t
```

This is true if M either:
1. Operates entirely in bispectrum space (receives B(x) as input), or
2. Commutes with cyclic shifts (M(shift(x)) = shift(M(x))), or
3. Is shift-invariant itself (M(shift(x)) = M(x))

**Actions required by module:**

| Module | Currently a G-morphism? | What to change |
|--------|------------------------|----------------|
| QGroupENN `project_equivariant` | NO | Prove that projection commutes with cyclic shifts (it does, but must be stated and tested) |
| `SymmetryDiscovery` | YES | Already uses bispectrum. Document formally. |
| `ExperimentalPipeline` (ex-UnifiedWDW) | NO | Either (a) move all analysis to bispectrum space, or (b) prove Reynolds projection commutes with shifts, or (c) remove from stack |
| `HaarWavelets` (ex-HolographicCodes) | NO | Prove that Haar decomposition commutes with cyclic shifts (it doesn't for odd lengths — must document the condition) |
| `Krylov` | NO | Prove that Lanczos tridiagonalization on shift-invariant features preserves shift invariance (it does, because the input is already invariant) |

### 5.3 Build L7: GuaranteeComposition.jl

A new module that:

1. **Defines the guarantee type hierarchy** (as above)
2. **Verifies composition at pipeline construction time**:

```julia
function verify_composition(pipeline::Vector{Module})
    g = ShiftInvariance(1e-15, 1024)
    for m in pipeline
        if !is_g_morphism(m, g)
            error("Module $m does not preserve guarantee $g")
        end
        g = compose_guarantee(g, m.output_guarantee)
    end
    return g  # combined guarantee for the full pipeline
end
```

3. **Provides runtime guarantee monitoring**:

```julia
function check_guarantee_at_runtime(x, pipeline, g::ShiftInvariance)
    for t in test_shifts
        err = norm(B(pipeline(shift(x, t))) - B(pipeline(x)))
        if err > g.max_error * 10  # 10x safety margin
            @warn "Guarantee violation detected at shift $t: error=$err"
        end
    end
end
```

---

## Part 6: Phase 2 — Real Guarantees (Weeks 9-16)

### 6.1 Real PAC-Bayes (replace SymmetryCertificate)

Delete the fake PAC-Bayes. Implement McAllester's bound:

```julia
function pac_bayes_bound(empirical_error, kl_divergence, n_samples, delta=0.05)
    # McAllester 2003: bound on generalization error
    return empirical_error + sqrt((kl_divergence + log(2*sqrt(n_samples)/delta)) / (2*n_samples))
end
```

Requires:
- Computing KL divergence between prior and posterior over weights
- Using FFTGroup's A_omega as the posterior mean
- Prior: isotropic Gaussian with sigma = 1 (independent of training)

### 6.2 Real MDL (replace RuptureABC)

Delete the fake MDL (`32 * params + 100`). Implement:

```julia
function mdl_complexity(model, x, precision_bits=16)
    # Two-part MDL:
    # L(D|M) = -log P(D|theta)  (negative log likelihood)
    # L(M) = sum over weights of precision_bits + log(weight_magnitude)
    return nll(model, x) + parametric_complexity(model, precision_bits)
end
```

### 6.3 Real Statistical Tests

Replace hardcoded p-values with permutation tests:

```julia
function permutation_p_value(wdw_score, baseline_scores, n_permutations=10000)
    # H0: WDW score <= max(baseline scores)
    # H1: WDW score > max(baseline scores)
    # p-value = fraction of permutations where baseline >= WDW
    combined = [wdw_score; baseline_scores]
    n_wdw = 1
    n_base = length(baseline_scores)
    count = 0
    for _ in 1:n_permutations
        perm = shuffle(combined)
        if maximum(perm[1:n_wdw]) <= maximum(perm[n_wdw+1:end])
            count += 1
        end
    end
    return count / n_permutations
end
```

### 6.4 Real Certificate Hash

Replace `"RUPTURE_CERT_$(n)_$(seed)"` and `"SC-$(Dates.format(...))"` with:

```julia
function certificate_hash(cert::AbstractCertificate)
    serialized = serialize(cert)
    return bytes2hex(SHA.sha256(serialized))
end
```

---

## Part 7: Phase 3 — The Meta-Theorem (Weeks 17-24)

### 7.1 The GALE Meta-Theorem

A trained GALE model guarantees:

```
For all x in R^n, for all g in G (discovered symmetry group):

1. f(shift(x)) = f(x)                    (classification invariance)
2. ||x - recover(encode(x))|| < epsilon  (reconstruction)
3. PAC-Bayes bound holds with prob > 0.95  (generalization)
4. MDL(model) < MDL(baseline)            (compression)
5. Certificate hash = SHA256(serialize(cert))  (tamper-proof)
```

Where G is the maximum symmetry group discovered by the symmetry profiling, and all bounds are explicit numerical values computed from the training data.

### 7.2 Implementation

```julia
struct GALECertificate{T}
    # Core guarantees (inherited from FFTGroup)
    shift_invariance_error::T
    cn_dn_gap::T
    recovery_error::T

    # Composed guarantees (verified at pipeline construction)
    pipeline_guarantee::Guarantee

    # Statistical guarantees (computed from training)
    pac_bayes_bound::T
    mdl_ratio::T
    generalization_gap::T

    # Discovery guarantees
    symmetry_group::String
    symmetry_coverage::T

    # Cryptographic binding
    hash::String
end
```

### 7.3 The Breakthrough: Proof Composition

The GALE system's unique value is not any individual guarantee — it is the ability to prove that:

```
If FFTGroup guarantees shift invariance at 1e-15
AND each pipeline module preserves this guarantee
THEN the full model guarantees shift invariance at 1e-15
```

This is what no other architecture can claim. CNNs learn shift invariance (error ~1e-2). E2CNN enforces architectural equivariance (error ~1e-8). GALE proves algebraic invariance (error ~1e-15) and proves that the full pipeline inherits it.

---

## Part 8: Effort Summary

| Phase | What | Person-Days | Outcome |
|-------|------|-------------|---------|
| **Phase 0** | Radical honesty: delete/rename mislabeled modules | 3-5 | README matches reality |
| **Phase 1** | Guarantee composition: make all modules G-morphisms | 15-25 | Pipeline preserves shift invariance end-to-end |
| **Phase 2** | Real guarantees: PAC-Bayes, MDL, p-values, hashes | 20-30 | Module claims are mathematically valid |
| **Phase 3** | Meta-theorem: GALE certificate composes all guarantees | 15-25 | A model trained with GALE has provable guarantees no other architecture matches |
| **Total** | | **53-85** | |

### Modules that require NO changes (already correct for their level)

| Module | Reason |
|--------|--------|
| FFTGroup.jl | Gold standard. Only maintenance. |
| FFTPipeline.jl | Works correctly. Needs more seeds/tests. |
| QGroupENN.jl | Correct implementation. Needs proof documentation. |
| SymmetryDiscovery.jl | Correct use of bispectrum. Needs statistical calibration. |
| Krylov/Complexity.jl | Honest 91-line stub. Either develop or leave as-is. |
| All Tier-3 math modules | Honest experimental math. Document as such. |

### Modules to DELETE

| Module | Lines | Reason |
|--------|-------|--------|
| ScalableWDW.jl | 382 | Every claim is false |
| SymmetryCertificate.jl | 557 | Produces fake certificates |
| RuptureABC.jl | 287 | Fake p-values, fake MDL |

### Modules to RENAME

| Current Name | New Name | Reason |
|-------------|----------|--------|
| UnifiedWDW.jl | ExperimentalPipeline.jl | Does not unify |
| HolographicCodes.jl | HaarWavelets.jl | It's Haar wavelets |
| WDWAutoencoder.jl | (move to Tier 3) | Uses wrong invariant |

---

## Part 9: Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| FFTGroup's 1e-15 guarantee does not hold for real-world data (not cyclic) | Medium | Critical | Test on PhysioNet ECG with linear shifts (padding vs cyclic) |
| Reynolds projection in QGroupENN does not compose with bispectrum | Low | Critical | Formal proof needed before Phase 1 completion |
| Fake PAC-Bayes in SymmetryCertificate creates false confidence if not deleted | High | High | DELETE Phase 0, not Phase 2 |
| Removing 3 modules (1226 LOC) breaks consumer scripts | Medium | Medium | Audit bench/ and test/ for references; update or delete |
| The meta-theorem (Phase 3) requires new math not yet in codebase | High | Medium | Start with the 3 core theorems (already proved); extend incrementally |

---

## Part 10: The ONE Question

Every decision in this roadmap must answer:

> **Does this change bring us closer to a mathematical proof that a GALE model guarantees properties that no other architecture can guarantee?**

If the answer is "no" — don't do it. If the answer is "yes" — do it first.

The 4 breakthroughs are real. The 4 verified results are real. The rest of the codebase is either building toward extending those guarantees or producing noise that dilutes them.

This roadmap is the plan to extend the guarantees to the full stack — and delete the noise.

---

*Document generated from first-principles analysis of 38 modules, 19 Zygote adjoints, 23 test files, and the 4 verified breakthroughs. The core theorems in FFTGroup.jl are mathematically correct. Everything else is measured against that standard.*
