"""
    AbstractHamiltonianBasis

Supertype for choices of basis for tight-binding models.
See [`Wannier`](@ref) and [`Bloch`](@ref)
"""
abstract type AbstractHamiltonianBasis end

"""
    Wannier <: AbstractHamiltonianBasis

Choice of basis in Wannier orbitals.
"""
struct Wannier <: AbstractHamiltonianBasis end

"""
    Bloch <: AbstractHamiltonianBasis

Choice of basis in Bloch functions, eigenstates of the Hamiltonian.
"""
struct Bloch <: AbstractHamiltonianBasis end


"""
    AbstractCoordinateBasis

Supertype of bases for coordinate/spatial indices
"""
abstract type AbstractCoordinateBasis end

"""
    Lattice <: AbstractCoordinateBasis

Singleton type representing lattice coordinates. The matrix ``B`` whose columns
are reciprocal lattice vectors converts this basis to the Cartesian basis.
"""
struct Lattice <: AbstractCoordinateBasis end

"""
    Cartesian <: AbstractCoordinateBasis

Singleton type representing Cartesian coordinates.
"""
struct Cartesian <: AbstractCoordinateBasis end

"""
    AbstractObservable

Supertype for observables to compute from tight-binding models.
"""
abstract type AbstractObservable end

struct WrapperFourierSeries{W,N,T,iip,S<:AbstractFourierSeries,P} <: AbstractFourierSeries{N,T,iip}
    w::W
    s::S
    p::P
    function WrapperFourierSeries(w, s::AbstractFourierSeries, p)
        return new{typeof(w),ndims(s),eltype(s),FourierSeriesEvaluators.isinplace(s),typeof(s), typeof(p)}(w, s, p)
    end
end

period(s::WrapperFourierSeries) = period(s.s)
frequency(s::WrapperFourierSeries) = frequency(s.s)
allocate(s::WrapperFourierSeries, x, dim) = allocate(s.s, x, dim)
function contract!(cache, s::WrapperFourierSeries, x, dim)
    return WrapperFourierSeries(s.w, contract!(cache, s.s, x, dim), s.p)
end
evaluate!(cache, s::WrapperFourierSeries, x) = s.w(evaluate!(cache, s.s, x), s.p)
nextderivative(s::WrapperFourierSeries, dim) = WrapperFourierSeries(s.w, nextderivative(s.s, dim), s.p)

show_dims(s::WrapperFourierSeries) = show_dims(s.s)
show_details(s::WrapperFourierSeries) = show_details(s.s)

struct Freq2RadSeries{N,T,iip,S<:AbstractFourierSeries{N,T,iip},Tt<:NTuple{N,T},Tf<:NTuple{N,Any}} <: AbstractFourierSeries{N,T,iip}
    s::S
    t::Tt
    f::Tf
end

function Freq2RadSeries(s::AbstractFourierSeries)
    f = map(freq2rad, frequency(s))
    t = map(inv, f)
    return Freq2RadSeries(s, t, f)
end

period(s::Freq2RadSeries) = s.t
frequency(s::Freq2RadSeries) = s.f
allocate(s::Freq2RadSeries, x, dim) = allocate(s.s, freq2rad(x), dim)
function contract!(cache, s::Freq2RadSeries, x, dim)
    t = FourierSeriesEvaluators.deleteat_(s.t, dim)
    f = FourierSeriesEvaluators.deleteat_(s.f, dim)
    return Freq2RadSeries(contract!(cache, s.s, freq2rad(x), dim), t, f)
end
function evaluate!(cache, s::Freq2RadSeries, x)
    return evaluate!(cache, s.s, freq2rad(x))
end
function nextderivative(s::Freq2RadSeries, dim)
    return Freq2RadSeries(nextderivative(s.s, dim), s.t, s.f)
end

show_dims(s::Freq2RadSeries) = show_dims(s.s)
show_details(s::Freq2RadSeries) = show_details(s.s)
