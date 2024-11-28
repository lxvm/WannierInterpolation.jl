"""
    Hamiltonian(basis::AbstractHamiltonianBasis)

Observable for the Hamiltonian operator measured in `basis`.
"""
struct Hamiltonian{B<:AbstractHamiltonianBasis,H} <: AbstractObservable
    basis::B
    herm::Val{H}
end
Hamiltonian(basis; herm=false) = Hamiltonian(basis, Val(herm))

struct HamiltonianProblem{H<:AbstractMatrix,O<:Hamiltonian}
    H::H
    obs::O
end

struct HamiltonianSolver{H,O,A,C,K}
    H::H
    obs::O
    alg::A
    cacheval::C
    kwargs::K
end

function init(prob::HamiltonianProblem, alg=prob.obs.herm isa Val{true} ? LAPACKEigenH() : LAPACKEigen(); kws...)
    (; H, obs) = prob
    cacheval = init_cacheval(prob, alg)
    return HamiltonianSolver(H, obs, alg, cacheval, NamedTuple(kws))
end

function  init_cacheval(prob::HamiltonianProblem, alg)
    (; H, obs) = prob
    (; basis) = obs
    if basis isa Wannier
        return nothing
    elseif basis isa Bloch
        return init(EigenProblem(H), alg)
    else
        error("Basis $basis not implemented")
    end
end

function solve!(solver::HamiltonianSolver)
    (; H, obs, cacheval) = solver
    (; basis) = obs
    if basis isa Wannier
        return H
    elseif basis isa Bloch
        cacheval.A = H
        return solve!(cacheval).value
    else
        error("Basis $basis not implemented")
    end
end

function wrap_hamiltonian(hk, obs)
    return HamiltonianProblem(hk, obs)
end

function interpolate_hamiltonian(obs::Hamiltonian, H, Rvectors, Rdegens)

    Rlims = ntuple(n -> extrema(r[n] for r in Rvectors), length(first(Rvectors)))
    Rmin, Rmax = getindex.(Rlims, 1), getindex.(Rlims, 2)
    Rsize = Tuple(Rmax .- Rmin .+ 1)
    n, m = size(first(H))

    H_R = OffsetArray(
        zeros(SMatrix{n,m,eltype(eltype(H)),n*m}, Rsize...),
        map(:, Rmin, Rmax)...
    )
    for (i, h, n) in zip(Rvectors, H, Rdegens)
        H_R[CartesianIndex(Tuple(i))] = h / n
    end

    h = FourierSeries(H_R, period=1.0)
    # TODO options for precision
    return WrapperFourierSeries(wrap_hamiltonian, obs.herm isa Val{true} ? HermitianFourierSeries(h) : h, obs)
end