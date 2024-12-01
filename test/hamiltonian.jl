using Test
using PyCall
wb = pyimport_conda("wannierberri", "wannierberri", "conda-forge")
using WannierIO, WannierInterpolation
using FourierSeriesEvaluators
using LinearAlgebra
# using Wannier
using LazyArtifacts

rtol = 1e-5

@testset "model $name" for (name, seed) in [
    ("Si2_valence", artifact"Si2_valence/reference/WS/Si2_valence"),
]
    system=wb.System_w90(seedname=seed)
    tabulators = Dict(
        "Energy" => wb.calculators.tabulate.Energy(),
    )
    grid = wb.Grid(system, NK=20)
    tab_all_path = wb.calculators.TabulatorAll(
                        tabulators,
                        ibands = 0:2,
                        mode = "grid",
                        save_mode="none",
                            )
    result=wb.run(system,
                    grid=grid,
                    calculators = Dict("tabulate" => tab_all_path),
                    print_Kpoints = false)

    grid_result = result.results["tabulate"]

    h = interpolate(WannierInterpolation.Hamiltonian(WannierInterpolation.Bloch()), seed)
    w = FourierSeriesEvaluators.workspace_allocate(h, FourierSeriesEvaluators.period(h))

    vals = getproperty.(solve.(w.(eachrow(grid.points_FFT))), :values)

    nband = system.num_wann.data[]
    for n in 1:nband
        @test norm(vec(permutedims(grid_result.get_data(iband=n-1, quantity="Energy"), (3,2,1))) - getindex.(vals, n))./norm(getindex.(vals, n)) < rtol
    end
    #=
    model = read_w90_with_chk(seed)

    hamiltonian = TBHamiltonian(model)
    interp = HamiltonianInterpolator(hamiltonian)
    wvals = getindex.(interp.(eachrow(grid.points_FFT)), 1)
    for n in 1:nband
        @test norm(vec(permutedims(grid_result.get_data(iband=n-1, quantity="Energy"), (3,2,1))) - getindex.(wvals, n))./norm(getindex.(wvals, n)) < rtol
    end
    =#
end