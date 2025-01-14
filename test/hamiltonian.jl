using Test
using PyCall
wb = pyimport_conda("wannierberri", "wannierberri", "conda-forge")
using WannierIO, WannierInterpolation
using FourierSeriesEvaluators
using LinearAlgebra
# using Wannier
using LazyArtifacts

rtol = 1e-5

mktempdir() do tmpdir

symlink(artifact"Si2_valence/Si2_valence.eig", joinpath(tmpdir, "Si2_valence.eig"))
symlink(artifact"Si2_valence/Si2_valence.mmn", joinpath(tmpdir, "Si2_valence.mmn"))
symlink(artifact"Si2_valence/reference/binary/Si2_valence.chk", joinpath(tmpdir, "Si2_valence.chk"))
symlink(artifact"Si2_valence/reference/WS/Si2_valence_hr.dat", joinpath(tmpdir, "Si2_valence_hr.dat"))

@testset "model $name" for name in [
    "Si2_valence",
]
    seed = joinpath(tmpdir, name)
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

    vals = getproperty.(solve.(w.(eachrow(grid_result.kpoints))), :values)

    nband = grid_result.nband
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
end