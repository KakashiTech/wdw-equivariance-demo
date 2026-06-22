# TIER 3 — EXPERIMENTAL: Topological spaces and Heyting algebra
module Knowledge

struct TopSpace{T}
    X::Vector{T}
    opens::Vector{Vector{T}}
end

function is_open(space::TopSpace{T}, U::Vector{T}) where {T}
    any(Set(U) == Set(O) for O in space.opens)
end

function int(space::TopSpace{T}, S::Vector{T}) where {T}
    SS = Set(S)
    acc = Set{T}()
    for O in space.opens
        if issubset(Set(O), SS)
            union!(acc, Set(O))
        end
    end
    collect(acc)
end

function cl(space::TopSpace{T}, S::Vector{T}) where {T}
    Xs = Set(space.X)
    comp = setdiff(Xs, Set(S))
    collect(setdiff(Xs, Set(int(space, collect(comp)))))
end

struct HeytingOpen{T}
    space::TopSpace{T}
    U::Vector{T}
    function HeytingOpen(space::TopSpace{T}, U::Vector{T}) where {T}
        is_open(space, U) || (U = int(space, U))
        new{T}(space, U)
    end
end

function heyting_top(space::TopSpace{T}) where {T}
    HeytingOpen(space, space.X)
end

function heyting_bot(space::TopSpace{T}) where {T}
    HeytingOpen(space, Vector{T}())
end

function heyting_and(a::HeytingOpen{T}, b::HeytingOpen{T}) where {T}
    U = collect(intersect(Set(a.U), Set(b.U)))
    HeytingOpen(a.space, U)
end

function heyting_or(a::HeytingOpen{T}, b::HeytingOpen{T}) where {T}
    U = collect(union(Set(a.U), Set(b.U)))
    HeytingOpen(a.space, U)
end

function heyting_imply(a::HeytingOpen{T}, b::HeytingOpen{T}) where {T}
    Xs = Set(a.space.X)
    comp = collect(setdiff(Xs, Set(a.U)))
    U = collect(union(Set(comp), Set(b.U)))
    HeytingOpen(a.space, int(a.space, U))
end

function heyting_not(a::HeytingOpen{T}) where {T}
    heyting_imply(a, heyting_bot(a.space))
end

function heyting_leq(a::HeytingOpen{T}, b::HeytingOpen{T}) where {T}
    issubset(Set(a.U), Set(b.U))
end

struct Partial{U,V}
    domain::Vector{U}
    value::V
end

function restrict(p::Partial{U,V}, Vset::Vector{U}) where {U,V}
    issubset(Set(Vset), Set(p.domain)) || error("not a restriction")
    Partial{U,V}(Vset, p.value)
end

function compatible(p::Partial{U,V}, q::Partial{U,V}) where {U,V}
    inter = collect(intersect(Set(p.domain), Set(q.domain)))
    isempty(inter) || p.value == q.value
end

function glue_partial(ps::Vector{Partial{U,V}}) where {U,V}
    isempty(ps) && return false, nothing
    v = ps[1].value
    for p in ps
        if p.value != v
            return false, nothing
        end
    end
    dom = Vector{U}()
    for p in ps
        dom = collect(union(Set(dom), Set(p.domain)))
    end
    true, Partial{U,V}(dom, v)
end

export TopSpace, is_open, int, cl, HeytingOpen, heyting_top, heyting_bot, heyting_and, heyting_or, heyting_imply, heyting_not, heyting_leq, Partial, restrict, compatible, glue_partial

# --- Naming functor over opens (assigns stable names consistent under restriction) ---
struct Name
    id::Int
end

struct NamingFunctor{T}
    space::TopSpace{T}
    names::Dict{String,Name}  # key for open set -> Name
end

_key_of(U) = join(string.(sort(U)), ",")

function build_naming_functor(space::TopSpace{T}, cover::Vector{Vector{T}}, sections::Vector) where {T}
    @assert length(cover) == length(sections)
    # Assign equal names to opens carrying equal section values; include intersections
    names = Dict{String,Name}()
    nextid = 1
    # base opens
    for (U, v) in zip(cover, sections)
        key = _key_of(U)
        if !haskey(names, key)
            names[key] = Name(nextid); nextid += 1
        end
    end
    # intersections inherit the same name when compatible
    for i in 1:length(cover)
        for j in i:length(cover)
            Ui = cover[i]; Uj = cover[j]
            inter = collect(intersect(Set(Ui), Set(Uj)))
            if !isempty(inter)
                keyi = _key_of(Ui); keyj = _key_of(Uj); keyij = _key_of(inter)
                if sections[i] == sections[j]
                    # inherit consistent name
                    names[keyij] = get(names, keyij, names[keyi])
                else
                    # assign distinct name for conflicting values on overlap
                    if !haskey(names, keyij)
                        names[keyij] = Name(nextid); nextid += 1
                    end
                end
            end
        end
    end
    NamingFunctor(space, names)
end

function restrict_name(N::NamingFunctor{T}, U::Vector{T}, V::Vector{T}) where {T}
    issubset(Set(V), Set(U)) || error("V not subset of U")
    keyU = _key_of(U)
    keyV = _key_of(V)
    # If V has no explicit name, inherit name of U
    haskey(N.names, keyV) ? N.names[keyV] : N.names[keyU]
end

export Name, NamingFunctor, build_naming_functor, restrict_name

end
