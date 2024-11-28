module WannierInterpolation

using FourierSeriesEvaluators
using StaticArrays
using OffsetArrays
using LinearAlgebra

using LinearAlgebra: checksquare
using FastLapackInterface: EigenWs, HermitianEigenWs
using FourierSeriesEvaluators: FourierWorkspace, freq2rad

import FourierSeriesEvaluators: period, frequency, allocate, contract!, evaluate!, nextderivative, show_dims, show_details
import CommonSolve: init, solve!, solve

export init, solve!, solve

export AbstractHamiltonianBasis, Wannier, Bloch
export AbstractCoordinateBasis, Cartesian, Lattice
export AbstractObservable
include("definitions.jl")

include("eigen.jl")

export Hamiltonian
include("hamiltonian.jl")
export BerryConnection
include("berryconnection.jl")
export BerryCurvature
include("berrycurvature.jl")
export Velocity
include("velocity.jl")

export interpolate
include("io.jl")

end # module WannierInterpolation
