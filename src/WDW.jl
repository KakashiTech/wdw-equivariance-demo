module WDW

# =============================================================================
# ▶ CORE — Verified algebraic guarantees (G3-G4)
#   FFTGroup has no WDW dependencies — can load first.
# =============================================================================
include("Core/FFTGroup.jl")

# =============================================================================
# ▶ EXPERIMENTAL — Mathematical foundations and research prototypes
#   Loaded before FFTPipeline and Research as they depend on these.
# =============================================================================
include("Experimental/Logic/DSL.jl")
include("Experimental/Semantics/Kripke.jl")
include("Experimental/Category/Sets.jl")
include("Experimental/Knowledge/TopologicalFunctors.jl")
include("Experimental/Sheaves/FiniteSheaves.jl")
include("Experimental/Algebra/Quivers.jl")
include("Experimental/Motives/ComputableMotives.jl")
include("Experimental/Motives/MotivicReduce.jl")
include("Experimental/Quantum/QGroupENN.jl")
include("Experimental/Tensor/HolographicCodes.jl")
include("Experimental/Krylov/Complexity.jl")
include("Experimental/Time/ITE.jl")
include("Experimental/Time/MultiTime.jl")
include("Experimental/Time/HyperTime.jl")
include("Experimental/Planner/ChronosKairos.jl")
include("Experimental/Bio/Microtubules.jl")
include("Experimental/Gravity/LQGDataSpace.jl")
include("Experimental/Vacuum/QET.jl")
include("Experimental/UnifiedWDW.jl")
include("Experimental/RuptureABC.jl")
include("Experimental/ScalableWDW.jl")

# =============================================================================
# ▶ CORE PIPELINE — Depends on Experimental modules (UnifiedWDW)
# =============================================================================
include("Core/FFTPipeline.jl")

# =============================================================================
# ▶ RESEARCH — Extensions, benchmarks, and experiments
# =============================================================================
include("Research/Symmetry/SymmetryDiscovery.jl")
include("Research/Symmetry/SymmetryCertificate.jl")
include("Research/Symmetry/AutoSymmetryDiscovery.jl")
include("Research/Symmetry/AutoSymmetryFlux.jl")
include("Research/Autoencoder/WDWAutoencoder.jl")
include("Research/Metrics/TheoreticalMetrics.jl")
include("Research/Metrics/RigorousMetrics.jl")
include("Research/Benchmarks/RealBaselines.jl")
include("Research/Benchmarks/RealWorldApplications.jl")
include("Research/Benchmarks/MultiDataset.jl")
include("Research/Benchmarks/LatticePhonons.jl")
include("Research/Benchmarks/PaperMetrics.jl")
include("Research/Experiments/BreakthroughExperiment.jl")
include("Research/Experiments/StructuralExperiments.jl")
include("Research/Experiments/StructuralEmbedding.jl")
include("Research/Integration/UnifiedIntegration.jl")

# =============================================================================
# ▶ DEPRECATED
# =============================================================================
include("Deprecated/mlp_baseline.jl")

# =============================================================================
# EXPORTS
# =============================================================================
export FFTGroup, FFTPipeline,
       Logic, Semantics, Category, Knowledge, Sheaves, Algebra,
       Motives, MotivicReduce, Quantum, Tensor, Krylov,
       TimeITE, TimeMulti, TimeHyper, Planner, Bio, Gravity, Vacuum,
       UnifiedWDW, RuptureABC, ScalableWDW,
       SymmetryDiscovery, SymmetryCertificate,
       AutoSymmetryDiscovery, AutoSymmetryFlux,
       WDWAutoencoder,
       TheoreticalMetrics, RigorousMetrics,
       RealBaselines, RealWorldApplications, MultiDataset,
       LatticePhonons, PaperMetrics,
       BreakthroughExperiment, StructuralExperiments, StructuralEmbedding,
       UnifiedIntegration

end
