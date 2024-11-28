module WannierInterpolationWannierIOExt
using WannierIO
using WannierInterpolation
import WannierInterpolation: interpolate

function interpolate(obs::Hamiltonian, seed)
    hrdat = read_w90_hrdat(seed * "_hr.dat")
    return WannierInterpolation.interpolate_hamiltonian(obs, hrdat.H, hrdat.Rvectors, hrdat.Rdegens)
end

end