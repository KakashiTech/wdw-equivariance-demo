# TIER 3 — EXPERIMENTAL: Categorical logic formula DSL
module Logic

abstract type Formula end

struct Top <: Formula end
struct Bot <: Formula end
struct Var <: Formula
    name::Symbol
end
struct And <: Formula
    left::Formula
    right::Formula
end
struct Or <: Formula
    left::Formula
    right::Formula
end
struct Imply <: Formula
    left::Formula
    right::Formula
end
struct Not <: Formula
    inner::Formula
end

function ∧(a::Formula, b::Formula)
    And(a,b)
end
function ∨(a::Formula, b::Formula)
    Or(a,b)
end
function ⇒(a::Formula, b::Formula)
    Imply(a,b)
end
function ¬(a::Formula)
    Not(a)
end

export Formula, Top, Bot, Var, And, Or, Imply, Not, ∧, ∨, ⇒, ¬

end
