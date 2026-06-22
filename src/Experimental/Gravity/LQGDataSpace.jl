# TIER 3 — EXPERIMENTAL: Loop quantum gravity spin networks
module Gravity

struct SpinNetwork
    nodes::Vector{Int}
    edges::Vector{Tuple{Int,Int,Int}}
end

function area_information(sn::SpinNetwork)
    s = 0.0
    for (_,_,j) in sn.edges
        s += j * (j + 1)
    end
    s
end

function relabel(sn::SpinNetwork, π::Vector{Int})
    nodes2 = [π[i] for i in sn.nodes]
    edges2 = [(π[u], π[v], j) for (u,v,j) in sn.edges]
    SpinNetwork(nodes2, edges2)
end

export SpinNetwork, area_information, relabel

end
