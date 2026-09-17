#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Versão MPI segura v8 + restrict_X/Y/Z + extend-columns periódico XY: cálculos GPAW coletivos em todos os ranks e I/O somente
# no rank 0. As escritas ASE feitas apenas pelo master usam parallel=False para
# não iniciar coletivas MPI internas e desalinhar os ranks antes do próximo SCF.
# Execute com: mpiexec -n N gpaw python este_script.py

import argparse, csv, math, random, os, traceback, builtins, pickle
from dataclasses import dataclass
from typing import Optional, Set
import numpy as np

from ase import Atom, Atoms
from ase.io import read, write, Trajectory
from ase.build import surface
from ase.constraints import FixAtoms
from ase.optimize import FIRE, BFGS
from ase.neighborlist import neighbor_list


# -----------------------------------------------------------------------------
# Suporte MPI seguro
# -----------------------------------------------------------------------------
# Com ``gpaw python`` o comunicador abaixo representa todos os processos MPI.
# Sem GPAW instalado, o script continua funcionando em modo serial para os
# demais backends.
try:
    from gpaw.mpi import world as MPI_WORLD
except ImportError:
    class _SerialWorld:
        rank = 0
        size = 1
        backend = "serial-sem-gpaw"

        @staticmethod
        def barrier():
            return None

    MPI_WORLD = _SerialWorld()

MPI_RANK = int(getattr(MPI_WORLD, "rank", 0))
MPI_SIZE = int(getattr(MPI_WORLD, "size", 1))
MPI_MASTER = MPI_RANK == 0
MPI_BACKEND = str(getattr(MPI_WORLD, "backend", "desconhecido"))


def master_print(*args, **kwargs):
    """Imprime somente no rank 0 para não multiplicar o log por MPI_SIZE."""
    if MPI_MASTER:
        builtins.print(*args, **kwargs)


# Todas as chamadas ``print(...)`` deste arquivo passam a ser master-only.
print = master_print


def mpi_barrier():
    """Barreira compatível com o comunicador serial e com o GPAW."""
    barrier = getattr(MPI_WORLD, "barrier", None)
    if callable(barrier):
        barrier()


def mpi_broadcast_object(obj, root=0):
    """Transmite um objeto Python usando somente buffers NumPy.

    Não usa ``gpaw.mpi.broadcast`` para manter o protocolo explícito. A tupla
    observada nas versões anteriores vinha de uma coletiva MPI do ASE iniciada
    por ``ase.io.write()``, não de uma falha de serialização desta função.
    Aqui fazemos explicitamente:

      1. ``pickle.dumps`` no rank raiz;
      2. broadcast do tamanho do buffer;
      3. broadcast dos bytes como ``int8``;
      4. ``pickle.loads`` nos demais ranks.

    O método ``MPI_WORLD.broadcast`` trabalha in-place com arrays NumPy e é a
    interface de baixo nível estável do comunicador GPAW.
    """
    if MPI_SIZE == 1:
        return obj

    if not (0 <= int(root) < MPI_SIZE):
        raise ValueError(f"Rank raiz inválido para broadcast: {root}")

    if MPI_RANK == root:
        raw = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        nbytes = np.array([len(raw)], dtype=np.int64)
    else:
        raw = None
        nbytes = np.zeros(1, dtype=np.int64)

    MPI_WORLD.broadcast(nbytes, int(root))
    size = int(nbytes[0])
    if size < 0:
        raise RuntimeError(f"Tamanho de broadcast inválido: {size}")

    if MPI_RANK == root:
        # Cópia gravável e C-contiguous; np.frombuffer(bytes) é somente leitura.
        buffer = np.frombuffer(raw, dtype=np.int8).copy()
    else:
        buffer = np.empty(size, dtype=np.int8)

    MPI_WORLD.broadcast(buffer, int(root))

    if MPI_RANK == root:
        return obj
    return pickle.loads(buffer.tobytes())


def mpi_broadcast_atoms(atoms, root=0):
    """Reconstrói em todos os ranks uma cópia idêntica do ``Atoms`` do root.

    Não transmitimos diretamente o objeto ``ase.Atoms``. Em algumas combinações
    de ASE/GPAW/Python, o pickle interno pode ser desserializado como a tupla de
    estado usada pelo ASE, em vez de retornar uma instância pronta de ``Atoms``.
    Aqui transmitimos somente dados simples e reconstruímos o objeto localmente.

    São preservados números atômicos, posições, célula, PBC, arrays atômicos
    válidos (tags, massas, momentos etc.) e ``info``. A calculadora e constraints
    não são transmitidas; nos pontos em que esta função é chamada, a calculadora
    é recriada coletivamente logo depois e não deve haver constraint ativa.
    """
    if MPI_SIZE == 1:
        if atoms is not None:
            atoms.calc = None
        return atoms

    payload = None
    if MPI_RANK == root:
        if atoms is None:
            payload = {"present": False}
        else:
            arrays = {}
            natoms = len(atoms)
            for name, array in atoms.arrays.items():
                if name in {"numbers", "positions"}:
                    continue
                arr = np.asarray(array)
                if arr.ndim > 0 and len(arr) == natoms:
                    arrays[name] = np.array(arr, copy=True)

            payload = {
                "present": True,
                "numbers": np.array(atoms.numbers, dtype=int, copy=True),
                "positions": np.array(atoms.positions, dtype=float, copy=True),
                "cell": np.array(atoms.cell.array, dtype=float, copy=True),
                "pbc": np.array(atoms.pbc, dtype=bool, copy=True),
                "arrays": arrays,
                "info": dict(atoms.info),
            }

    payload = mpi_broadcast_object(payload, root=root)
    if not payload.get("present", False):
        return None

    synced = Atoms(
        numbers=np.asarray(payload["numbers"], dtype=int),
        positions=np.asarray(payload["positions"], dtype=float),
        cell=np.asarray(payload["cell"], dtype=float),
        pbc=np.asarray(payload["pbc"], dtype=bool),
    )

    for name, array in payload.get("arrays", {}).items():
        arr = np.array(array, copy=True)
        if name in synced.arrays:
            synced.set_array(name, arr)
        else:
            synced.new_array(name, arr)

    synced.info.update(payload.get("info", {}))
    synced.calc = None
    return synced


_SYNC_PY_RNG = random.Random(0)
_SYNC_NP_RNG = np.random.default_rng(0)


def initialize_synced_rng(seed):
    """Inicializa geradores usados apenas pelo rank 0 e transmitidos por MPI."""
    global _SYNC_PY_RNG, _SYNC_NP_RNG
    _SYNC_PY_RNG = random.Random(int(seed))
    _SYNC_NP_RNG = np.random.default_rng(int(seed))


def mpi_random():
    value = _SYNC_PY_RNG.random() if MPI_MASTER else None
    return float(mpi_broadcast_object(value))


def mpi_uniform(a, b):
    value = _SYNC_PY_RNG.uniform(a, b) if MPI_MASTER else None
    return float(mpi_broadcast_object(value))


def mpi_choice(sequence):
    if len(sequence) == 0:
        raise IndexError("Não é possível escolher de uma sequência vazia.")
    index = _SYNC_PY_RNG.randrange(len(sequence)) if MPI_MASTER else None
    index = int(mpi_broadcast_object(index))
    return sequence[index]


def mpi_np_choice_index(n, probabilities):
    index = (
        int(_SYNC_NP_RNG.choice(int(n), p=np.asarray(probabilities, dtype=float)))
        if MPI_MASTER else None
    )
    return int(mpi_broadcast_object(index))


def is_atoms_mismatch_exception(exc):
    return "Mismatch of Atoms objects" in str(exc)


def reraise_mpi_atoms_mismatch(exc, context):
    if is_atoms_mismatch_exception(exc):
        raise RuntimeError(
            f"Divergência de estrutura entre ranks MPI durante {context}. "
            "O job foi interrompido para não continuar com um comunicador GPAW inconsistente."
        ) from exc


def detected_launcher_size():
    """Detecta quantos processos o launcher MPI aparenta ter iniciado.

    Isso permite detectar o erro perigoso ``mpiexec python script.py`` em
    versões recentes do GPAW, nas quais o Python normal não ativa o backend MPI.
    ``PBS_NP`` não é usado aqui porque indica recursos alocados, não processos
    efetivamente iniciados.
    """
    values = []
    for name in (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
        "MPI_LOCALNRANKS",
    ):
        value = os.environ.get(name)
        if value:
            try:
                values.append(int(value))
            except ValueError:
                pass
    return max(values, default=1)


def validate_mpi_runtime(args):
    """Valida o launcher e informa a configuração MPI usada pelo script."""
    launcher_size = detected_launcher_size()

    if args.backend == "gpaw" and launcher_size > 1 and MPI_SIZE == 1:
        raise RuntimeError(
            "O launcher iniciou vários processos, mas o GPAW está em modo serial. "
            "Execute com: mpiexec -n N gpaw python mc_flux_corrigido_mpi.py ..."
        )

    if MPI_SIZE > 1 and args.backend != "gpaw":
        print(
            f"[AVISO] MPI_SIZE={MPI_SIZE}, mas backend={args.backend}. "
            "O paralelismo interno implementado aqui é destinado ao GPAW; "
            "outros backends podem repetir o mesmo cálculo em cada rank."
        )

    print(
        f"MPI: rank={MPI_RANK}/{MPI_SIZE}; backend_comunicador={MPI_BACKEND}; "
        f"launcher_size_detectado={launcher_size}"
    )

KB_EV = 8.617333262145e-5
AXIS = {"x": 0, "y": 1, "z": 2}
AXIS_NAME = {0: "x", 1: "y", 2: "z"}



class FreshCalculatorFactory:
    """Fábrica para calculadoras que não devem ser reutilizadas entre Atoms.

    Calculadoras ML/empíricas costumam ser reutilizáveis. O GPAW, por outro lado,
    guarda estado interno do sistema calculado; para crescimento/deposição,
    quando o número de átomos muda a cada candidato, é mais seguro criar uma
    calculadora nova para cada estrutura.
    """

    _fresh_calculator_factory = True

    def __init__(self, maker):
        self.maker = maker

    def __call__(self):
        return self.maker()


def is_fresh_calculator_factory(calc):
    return bool(getattr(calc, "_fresh_calculator_factory", False))


def _make_gpaw_calculator(args):
    """Cria uma calculadora GPAW/DFT nova.

    Padrões escolhidos para crescimento:
      - symmetry='off' por padrão, porque a deposição de um átomo quebra a simetria;
      - uma instância nova por Atoms, via FreshCalculatorFactory.
    """
    try:
        from gpaw import GPAW, PW
    except ImportError as exc:
        raise ImportError(
            "Backend gpaw solicitado, mas o pacote gpaw não está instalado. "
            "Ative/instale um ambiente com GPAW e ASE."
        ) from exc

    txt = args.gpaw_txt
    if txt is not None and str(txt).lower() in {"none", "null", "off"}:
        txt = None

    kwargs = {
        "xc": args.gpaw_xc,
        "kpts": tuple(args.gpaw_kpts),
        "txt": txt,
        "maxiter": int(args.gpaw_maxiter),
        "spinpol": bool(args.gpaw_spinpol),
    }

    if args.gpaw_symmetry == "off":
        kwargs["symmetry"] = "off"

    mode = args.gpaw_mode.lower()
    if mode == "pw":
        kwargs["mode"] = PW(float(args.gpaw_ecut))
    elif mode == "fd":
        kwargs["mode"] = "fd"
        kwargs["h"] = float(args.gpaw_h)
    elif mode == "lcao":
        kwargs["mode"] = "lcao"
        if args.gpaw_basis:
            kwargs["basis"] = args.gpaw_basis
    else:
        raise ValueError("--gpaw-mode deve ser pw, fd ou lcao")

    return GPAW(**kwargs)



def make_calculator(args):
    """Cria uma calculadora ASE.

    Backends ML universais opcionais:
      chgnet, mace, mace-medium, sevennet, sevennet-d3, mattersim

    Observação: cada backend precisa estar instalado no seu ambiente Python.
    O script falha com uma mensagem clara se o pacote não estiver instalado.
    """
    backend = args.backend
    device = args.backend_device

    if backend == "gpaw":
        # GPAW/DFT é stateful; retornamos uma fábrica para criar uma calculadora
        # nova a cada attach_calculator(), evitando reutilizar estado entre slab,
        # candidatos de deposição e relaxações.
        return FreshCalculatorFactory(lambda: _make_gpaw_calculator(args))

    if backend == "chgnet":
        try:
            from chgnet.model.dynamics import CHGNetCalculator
        except ImportError as exc:
            raise ImportError("Backend chgnet solicitado, mas o pacote chgnet não está instalado. Use: pip install chgnet") from exc
        return CHGNetCalculator(use_device=device)

    if backend in ("mace", "mace-small", "mace-medium"):
        try:
            from mace.calculators import mace_mp
        except ImportError as exc:
            raise ImportError("Backend MACE solicitado, mas mace-torch não está instalado. Use: pip install mace-torch") from exc

        model = args.mace_model
        if backend in ("mace", "mace-small") and model == "auto":
            model = "small"
        elif backend == "mace-medium" and model == "auto":
            model = "medium"

        return mace_mp(
            model=model,
            dispersion=bool(args.mace_dispersion),
            default_dtype=args.backend_dtype,
            device=device,
        )

    if backend in ("sevennet", "sevennet-d3"):
        if backend == "sevennet-d3":
            if device == "cpu":
                print("[AVISO] SevenNet-D3 pode exigir CUDA/nvcc na versão atual do SevenNet; se falhar, use --backend sevennet ou --backend mace.")
            try:
                from sevenn.calculator import SevenNetD3Calculator
                return SevenNetD3Calculator(model=args.sevennet_model, device=device)
            except ImportError as exc:
                raise ImportError("Backend sevennet-d3 solicitado, mas SevenNetD3Calculator não está disponível. Tente: pip install sevenn") from exc
            except TypeError:
                from sevenn.calculator import SevenNetD3Calculator
                return SevenNetD3Calculator(args.sevennet_model, device=device)

        # sevennet puro. Tenta API nova e depois API antiga.
        try:
            from sevenn.calculator import SevenNetCalculator
            try:
                return SevenNetCalculator(model=args.sevennet_model, device=device)
            except TypeError:
                return SevenNetCalculator(args.sevennet_model, device=device)
        except ImportError:
            try:
                from sevenn.sevennet_calculator import SevenNetCalculator
                return SevenNetCalculator(args.sevennet_model, device=device)
            except ImportError as exc:
                raise ImportError("Backend sevennet solicitado, mas sevenn não está instalado. Use: pip install sevenn") from exc

    if backend == "mattersim":
        try:
            from mattersim.forcefield.potential import MatterSimCalculator
        except ImportError as exc:
            raise ImportError("Backend mattersim solicitado, mas mattersim não está instalado. Use: pip install mattersim") from exc

        # A API do MatterSim mudou entre versões; tentamos as formas mais comuns.
        for kwargs in (
            {"device": device, "load_path": args.mattersim_model} if args.mattersim_model else {"device": device},
            {"device": device},
            {},
        ):
            try:
                return MatterSimCalculator(**kwargs)
            except TypeError:
                continue
        return MatterSimCalculator()

    if backend == "emt":
        from ase.calculators.emt import EMT
        return EMT()

    if backend == "lj":
        from ase.calculators.lj import LennardJones
        return LennardJones()

    raise ValueError(f"Backend desconhecido: {backend}")


def attach_calculator(atoms, calc):
    if is_fresh_calculator_factory(calc):
        atoms.calc = calc()
    else:
        atoms.calc = calc
    return atoms


def orient_slab_growth_axis(atoms: Atoms, growth_axis: str) -> Atoms:
    """
    ASE surface() gera o slab com a normal/vácuo no eixo z do slab.

    Aqui fazemos uma transformação RIGOROSA da célula e das posições para que:
      growth-axis z -> normal continua em z, pbc=(True, True, False)
      growth-axis x -> normal antiga z vira novo eixo x, pbc=(False, True, True)
      growth-axis y -> normal antiga z vira novo eixo y, pbc=(True, False, True)

    Importante: reordenamos LINHAS da célula e também COMPONENTES cartesianas.
    Isso mantém o vetor não-periódico como cell[g], evitando corte/wrap errado.
    """
    growth_axis = growth_axis.lower()

    if growth_axis == "z":
        atoms.pbc = (True, True, False)
        return atoms

    if growth_axis == "x":
        # novo x = velho z; novo y = velho x; novo z = velho y
        order = [2, 0, 1]
        pbc = (False, True, True)
    elif growth_axis == "y":
        # novo x = velho x; novo y = velho z; novo z = velho y
        order = [0, 2, 1]
        pbc = (True, False, True)
    else:
        raise ValueError("--growth-axis deve ser x, y ou z")

    old_cell = atoms.cell.array.copy()
    old_pos = atoms.positions.copy()

    new_cell = old_cell[np.array(order), :][:, np.array(order)]
    new_pos = old_pos[:, np.array(order)]

    atoms.set_cell(new_cell, scale_atoms=False)
    atoms.set_positions(new_pos)
    atoms.pbc = pbc
    return atoms


def growth_axis_index(args):
    return AXIS[args.growth_axis.lower()]


def plane_axes_from_growth(g):
    return [i for i in [0, 1, 2] if i != g]


def top_coord(atoms, g):
    return float(np.max(atoms.positions[:, g]))


def bottom_coord(atoms, g):
    return float(np.min(atoms.positions[:, g]))


def coord_species_filtered(
    atoms, g, species_filter, reducer, restrictions=None, reference_cell=None
):
    """Extremo no eixo g usando somente espécies/região elegíveis ao crescimento."""
    idx = growth_eligible_indices(
        atoms, species_filter=species_filter, restrictions=restrictions,
        reference_cell=reference_cell,
    )
    if not idx:
        desc = []
        if species_filter is not None:
            desc.append("espécies=" + "+".join(sorted(species_filter)))
        if restrictions is not None and any(v is not None for v in restrictions):
            desc.append(f"restrições={restrictions}")
        raise RuntimeError(
            "Nenhum átomo inicial elegível para definir a frente de crescimento"
            + (" (" + ", ".join(desc) + ")" if desc else "")
        )
    vals = atoms.positions[np.asarray(idx, dtype=int), g]
    return float(reducer(vals))


def top_coord_species(atoms, g, species_filter=None, restrictions=None, reference_cell=None):
    return coord_species_filtered(
        atoms, g, species_filter, np.max, restrictions, reference_cell
    )


def bottom_coord_species(atoms, g, species_filter=None, restrictions=None, reference_cell=None):
    return coord_species_filtered(
        atoms, g, species_filter, np.min, restrictions, reference_cell
    )


def slab_thickness(atoms, g):
    return top_coord(atoms, g) - bottom_coord(atoms, g)


def surface_area(atoms, g):
    pa = plane_axes_from_growth(g)
    a = atoms.cell.array[pa[0]]
    b = atoms.cell.array[pa[1]]
    return float(np.linalg.norm(np.cross(a, b)))


def ev_a2_to_j_m2(x):
    return x * 16.021766208


def min_distance_to_atoms(atoms, position):
    return float(np.min(np.linalg.norm(atoms.positions - position, axis=1)))


def wrap_surface_plane_to_cell(atoms, position, g):
    pa = plane_axes_from_growth(g)
    cell = atoms.cell.array
    frac = np.linalg.solve(cell.T, position)
    frac[pa[0]] %= 1.0
    frac[pa[1]] %= 1.0
    wrapped = frac @ cell
    wrapped[g] = position[g]
    return wrapped


def periodic_plane_distances_to_point(atoms, position, g):
    pa = plane_axes_from_growth(g)
    cell = atoms.cell.array
    frac_atoms = atoms.get_scaled_positions(wrap=True)
    frac_point = np.linalg.solve(cell.T, position)

    dfrac = frac_atoms - frac_point
    dfrac[:, pa[0]] -= np.round(dfrac[:, pa[0]])
    dfrac[:, pa[1]] -= np.round(dfrac[:, pa[1]])
    dfrac[:, g] = 0.0

    dcart = dfrac @ cell
    return np.linalg.norm(dcart, axis=1)


def local_top_near_surface_point(atoms, position, g, radius):
    d = periodic_plane_distances_to_point(atoms, position, g)
    idx = np.where(d <= radius)[0]
    if len(idx) == 0:
        return top_coord(atoms, g)
    return float(np.max(atoms.positions[idx, g]))


def get_surface_indices(atoms, g, surface_window):
    t = top_coord(atoms, g)
    return [i for i, p in enumerate(atoms.positions) if p[g] >= t - surface_window]


def get_grid_top_surface_indices(
    atoms,
    g,
    n1,
    n2,
    surface_window,
    include_species: Optional[Set[str]] = None,
):
    """
    Divide o plano perpendicular ao eixo de crescimento em n1 x n2.

    growth-axis x -> grade em yz
    growth-axis y -> grade em xz
    growth-axis z -> grade em xy
    """
    pa = plane_axes_from_growth(g)
    eligible = []

    for i, atom in enumerate(atoms):
        if include_species is not None and atom.symbol not in include_species:
            continue
        eligible.append(i)

    if not eligible:
        return []

    top_allowed = max(atoms.positions[i, g] for i in eligible)
    frac = atoms.get_scaled_positions(wrap=True)
    best = {}

    for i in eligible:
        h = atoms.positions[i, g]
        if h < top_allowed - surface_window:
            continue

        i1 = min(max(int(frac[i, pa[0]] * n1), 0), n1 - 1)
        i2 = min(max(int(frac[i, pa[1]] * n2), 0), n2 - 1)
        key = (i1, i2)

        if key not in best or atoms.positions[i, g] > atoms.positions[best[key], g]:
            best[key] = i

    return list(best.values())


def generate_candidate_above_anchor(
    atoms,
    anchor_index,
    g,
    lateral_radius,
    height_min,
    height_max,
    local_top_radius,
):
    pa = plane_axes_from_growth(g)
    base = atoms.positions[anchor_index].copy()

    theta = mpi_uniform(0.0, 2.0 * math.pi)
    r = mpi_uniform(0.0, lateral_radius)

    pos = base.copy()
    pos[pa[0]] += r * math.cos(theta)
    pos[pa[1]] += r * math.sin(theta)
    pos = wrap_surface_plane_to_cell(atoms, pos, g)

    htop = local_top_near_surface_point(atoms, pos, g, local_top_radius)
    pos[g] = htop + mpi_uniform(height_min, height_max)
    return pos


def get_local_mobile_indices(atoms, center_indices, radius):
    """Seleciona átomos móveis para relaxação local usando PBC.

    Antes esta função usava distância cartesiana direta entre posições.
    Para sistemas periódicos em um eixo, por exemplo grafeno com --pbc 0 1 0,
    um átomo perto de y=0 não enxergava o vizinho periódico em y=L como próximo.
    A calculadora já usa PBC, mas a lista de átomos móveis podia ficar
    incompleta nas bordas periódicas. Agora usamos pbc_distances_to_point(),
    que respeita atoms.pbc.
    """
    mobile = set(int(i) for i in center_indices)
    for j in center_indices:
        d = pbc_distances_to_point(atoms, atoms.positions[int(j)])
        for i in np.where(d <= radius)[0]:
            mobile.add(int(i))
    return sorted(mobile)


def coordination_numbers(atoms, cutoff):
    """Números de coordenação respeitando todas as condições periódicas de contorno."""
    if len(atoms) == 0:
        return np.zeros(0, dtype=int)
    i = neighbor_list("i", atoms, float(cutoff), self_interaction=False)
    return np.bincount(i, minlength=len(atoms)).astype(int)


def surface_height_statistics(
    atoms, g, n1=8, n2=8, include_species=None, restrictions=None, reference_cell=None
):
    """Altura local da frente em uma malha 2D, com média, mediana, p90 e RMS.

    Cada célula da malha usa o átomo mais alto naquela região. Células vazias são
    ignoradas para não transformar ausência de átomos em altura zero artificial.
    """
    pa = plane_axes_from_growth(g)
    frac = atoms.get_scaled_positions(wrap=True)
    heights = {}
    for idx, atom in enumerate(atoms):
        if include_species is not None and atom.symbol not in include_species:
            continue
        if not position_passes_fractional_restrictions(
            atom.position, restrictions, reference_cell
        ):
            continue
        i1 = min(max(int(frac[idx, pa[0]] * n1), 0), n1 - 1)
        i2 = min(max(int(frac[idx, pa[1]] * n2), 0), n2 - 1)
        key = (i1, i2)
        z = float(atoms.positions[idx, g])
        if key not in heights or z > heights[key]:
            heights[key] = z
    vals = np.asarray(list(heights.values()), dtype=float)
    if len(vals) == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "rms": 0.0, "occupied_fraction": 0.0}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90.0)),
        "rms": float(np.std(vals)),
        "occupied_fraction": float(len(vals) / (n1 * n2)),
    }


def template_growth_metrics(
    atoms,
    template,
    g,
    occupancy_tol=0.6,
    layer_tol=0.25,
    origin_front=None,
    growth_direction="plus",
    front_min=0.0,
    allowed_site_species=None,
    growth_species=None,
    restrictions=None,
    reference_cell=None,
):
    """Métricas normalizadas dos sítios do template à frente inicial.

    Apenas sítios estritamente à frente da interface inicial são contados. Isso
    evita incluir os átomos da estrutura de partida no denominador. As métricas
    são independentes da densidade planar e do número de sítios de cada orientação.
    """
    empty = {
        "first_layer_occupancy": 0.0,
        "highest_complete_layer": -1,
        "n_layers": 0,
        "contiguous_complete_layers": 0,
        "contiguous_complete_layer_fraction": 0.0,
        "mean_layer_occupancy": 0.0,
        "n_growth_sites": 0,
        "n_occupied_growth_sites": 0,
        "growth_site_occupancy_fraction": 0.0,
        "growth_extent_A": 0.0,
    }
    if template is None or len(template) == 0:
        return empty

    allowed = None
    if allowed_site_species and not (len(allowed_site_species) == 1 and str(allowed_site_species[0]).lower() == "all"):
        allowed = set(allowed_site_species)
    eligible_atoms = growth_eligible_indices(
        atoms, species_filter=growth_species, restrictions=restrictions,
        reference_cell=reference_cell,
    ) if (growth_species is not None or (restrictions is not None and any(v is not None for v in restrictions))) else None

    rows = []
    for site in template:
        if allowed is not None and site.symbol not in allowed:
            continue
        if not position_passes_fractional_restrictions(
            site.position, restrictions, reference_cell
        ):
            continue
        zg = float(site.position[g])
        if origin_front is None:
            ahead = zg
        elif growth_direction == "plus":
            ahead = zg - float(origin_front)
        elif growth_direction == "minus":
            ahead = float(origin_front) - zg
        else:
            ahead = abs(zg - float(origin_front))

        # Conta somente a região destinada ao crescimento, não a célula inicial.
        if ahead < float(front_min) - 1e-12:
            continue

        pos = wrap_pbc_axes_to_cell(atoms, site.position.copy())
        occupied = min_distance_to_selected_atoms_pbc(atoms, pos, eligible_atoms) <= occupancy_tol
        rows.append((float(ahead), bool(occupied)))

    if not rows:
        return empty

    rows.sort(key=lambda x: x[0])
    layers = []
    for ahead, occ in rows:
        if not layers or abs(ahead - layers[-1][0]) > layer_tol:
            layers.append([ahead, [occ]])
        else:
            layers[-1][1].append(occ)

    occs = [float(np.mean(values)) for _, values in layers]
    complete = [i for i, value in enumerate(occs) if value >= 0.90]

    contiguous = 0
    for value in occs:
        if value >= 0.90:
            contiguous += 1
        else:
            break

    n_sites = len(rows)
    n_occupied = int(sum(int(occ) for _, occ in rows))
    n_layers = len(layers)

    return {
        "first_layer_occupancy": occs[0] if occs else 0.0,
        "highest_complete_layer": max(complete) if complete else -1,
        "n_layers": n_layers,
        "contiguous_complete_layers": contiguous,
        "contiguous_complete_layer_fraction": (contiguous / n_layers) if n_layers else 0.0,
        "mean_layer_occupancy": float(np.mean(occs)) if occs else 0.0,
        "n_growth_sites": n_sites,
        "n_occupied_growth_sites": n_occupied,
        "growth_site_occupancy_fraction": (n_occupied / n_sites) if n_sites else 0.0,
        "growth_extent_A": max(ahead for ahead, _ in rows),
    }


def template_layer_occupancy(atoms, template, g, occupancy_tol=0.6, layer_tol=0.25,
                             origin_front=None, growth_direction="plus", front_min=0.0):
    """Compatibilidade com versões anteriores; prefira template_growth_metrics()."""
    metrics = template_growth_metrics(
        atoms, template, g, occupancy_tol, layer_tol, origin_front,
        growth_direction, front_min,
    )
    return (
        metrics["first_layer_occupancy"],
        metrics["highest_complete_layer"],
        metrics["n_layers"],
    )


def kinetic_event_rate(base_rate, delta_e, temperature, barrier_eV=0.0, bep_alpha=0.5):
    """Taxa efetiva Arrhenius/BEP para incorporação.

    E_a = barrier + alpha*max(deltaE, 0). Eventos endotérmicos ficam mais lentos;
    eventos exotérmicos ainda precisam superar a barreira basal.
    """
    base_rate = max(float(base_rate), 0.0)
    if base_rate == 0.0:
        return 0.0
    if temperature <= 0:
        return base_rate if (barrier_eV + bep_alpha * max(float(delta_e), 0.0)) <= 0 else 0.0
    ea = max(0.0, float(barrier_eV) + float(bep_alpha) * max(float(delta_e), 0.0))
    exponent = min(700.0, ea / (KB_EV * float(temperature)))
    return base_rate * math.exp(-exponent)


def surface_roughness(atoms, g, surface_window):
    idx = get_surface_indices(atoms, g, surface_window)
    if len(idx) < 2:
        return 0.0
    return float(np.std(atoms.positions[idx, g]))


def coverage_estimate(atoms, g, n_added, site_area):
    area = surface_area(atoms, g)
    return 0.0 if area <= 0 else float(n_added * site_area / area)


def relax_local(atoms, mobile_indices, optimizer, fmax, max_steps):
    fixed = [i for i in range(len(atoms)) if i not in set(mobile_indices)]
    atoms.set_constraint(FixAtoms(indices=fixed))
    opt = FIRE(atoms, logfile=None) if optimizer == "fire" else BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    atoms.set_constraint()
    return atoms


def bulk_energy_per_atom(
    cif_path,
    calc,
    bulk_repeat,
    relax_bulk,
    optimizer,
    fmax,
    relax_steps,
    pbc_override=None,
):
    bulk = read(cif_path).repeat(bulk_repeat)
    if pbc_override is None:
        bulk.pbc = (True, True, True)
    else:
        bulk.pbc = normalize_pbc_override(pbc_override)
    attach_calculator(bulk, calc)

    if relax_bulk:
        relax_local(bulk, list(range(len(bulk))), optimizer, fmax, relax_steps)
        attach_calculator(bulk, calc)

    return bulk.get_potential_energy() / len(bulk)


def build_slab_from_cif(cif_path, miller, layers, vacuum, replicate, calc, growth_axis):
    """
    Nova lógica pedida:
      1) lê CIF
      2) replica a estrutura bulk ANTES de gerar o slab: --replicate a b c
      3) gera o slab com ASE surface(), ainda sem vácuo
      4) escolhe o eixo final de crescimento x/y/z
      5) coloca vácuo somente nesse eixo final

    Assim não existe repetição do vácuo nem slab-repeat depois do slab.
    """
    bulk = read(cif_path)
    bulk = bulk.repeat(replicate)
    bulk.pbc = (True, True, True)

    # Sem vácuo aqui. O vácuo é aplicado depois, no eixo final escolhido.
    slab = surface(bulk, indices=miller, layers=layers, vacuum=0.0)
    slab.pbc = (True, True, False)

    slab = orient_slab_growth_axis(slab, growth_axis)

    g = AXIS[growth_axis.lower()]
    slab.center(axis=g, vacuum=vacuum)

    attach_calculator(slab, calc)
    return slab




def pbc_from_growth_axis(g):
    pbc = [True, True, True]
    pbc[g] = False
    return tuple(pbc)


def normalize_pbc_override(pbc_values):
    if pbc_values is None:
        return None
    if len(pbc_values) != 3:
        raise ValueError("--pbc precisa ter três valores: 0/1 0/1 0/1")
    return tuple(bool(int(v)) for v in pbc_values)


def apply_pbc_override_or_growth(atoms, g, pbc_override=None):
    atoms.pbc = normalize_pbc_override(pbc_override) or pbc_from_growth_axis(g)
    return atoms


def build_direct_from_cif(cif_path, replicate, vacuum, calc, growth_axis, pbc_override=None, center_growth=False):
    atoms = read(cif_path)
    atoms = atoms.repeat(replicate)
    g = AXIS[growth_axis.lower()]
    apply_pbc_override_or_growth(atoms, g, pbc_override)

    # No modo direct, por padrão não colocamos vácuo no eixo de crescimento.
    # Para sistemas 2D, o vácuo geralmente já está em z no CIF.
    if center_growth and vacuum is not None and vacuum > 0:
        atoms.center(axis=g, vacuum=vacuum)

    attach_calculator(atoms, calc)
    return atoms


def build_initial_structure_from_args(args, calc):
    g = AXIS[args.growth_axis.lower()]

    if args.build_mode == "direct":
        return build_direct_from_cif(
            args.cif,
            tuple(args.replicate),
            args.vacuum,
            calc,
            args.growth_axis,
            args.pbc,
            args.direct_center_growth,
        )

    slab = build_slab_from_cif(
        args.cif,
        tuple(args.miller),
        args.layers,
        args.vacuum,
        tuple(args.replicate),
        calc,
        args.growth_axis,
    )
    if args.pbc is not None:
        apply_pbc_override_or_growth(slab, g, args.pbc)
    return slab


def pbc_distances_to_point(atoms, position):
    cell = atoms.cell.array
    frac_atoms = atoms.get_scaled_positions(wrap=False)
    frac_point = np.linalg.solve(cell.T, np.asarray(position, dtype=float))

    dfrac = frac_atoms - frac_point
    for ax in range(3):
        if bool(atoms.pbc[ax]):
            dfrac[:, ax] -= np.round(dfrac[:, ax])

    dcart = dfrac @ cell
    return np.linalg.norm(dcart, axis=1)


def min_distance_to_atoms_pbc(atoms, position):
    return float(np.min(pbc_distances_to_point(atoms, position)))


def pbc_distances_positions_to_point(atoms, positions, position):
    """Distâncias PBC entre uma lista de posições cartesianas e um ponto."""
    if positions is None or len(positions) == 0:
        return np.asarray([], dtype=float)

    cell = atoms.cell.array
    frac_pos = np.linalg.solve(cell.T, np.asarray(positions, dtype=float).T).T
    frac_point = np.linalg.solve(cell.T, np.asarray(position, dtype=float))

    dfrac = frac_pos - frac_point
    for ax in range(3):
        if bool(atoms.pbc[ax]):
            dfrac[:, ax] -= np.round(dfrac[:, ax])

    dcart = dfrac @ cell
    return np.linalg.norm(dcart, axis=1)


def min_distance_to_vacancies_pbc(atoms, vacancy_positions, position):
    d = pbc_distances_positions_to_point(atoms, vacancy_positions, position)
    if len(d) == 0:
        return float("inf")
    return float(np.min(d))


def wrap_pbc_axes_to_cell(atoms, position):
    cell = atoms.cell.array
    frac = np.linalg.solve(cell.T, np.asarray(position, dtype=float))
    for ax in range(3):
        if bool(atoms.pbc[ax]):
            frac[ax] %= 1.0
    return frac @ cell


def normalize_fractional_restrictions(restrict_x=None, restrict_y=None, restrict_z=None):
    """Retorna limites mínimos em coordenadas fracionárias/cristalográficas.

    Um eixo com valor None fica completamente livre. Quando mais de um eixo é
    definido, todos os limites são aplicados simultaneamente (lógica AND).
    Ex.: Z>=0.394; ou X>=0.2 AND Y>=0.14 AND Z>=0.394.
    """
    values = (restrict_x, restrict_y, restrict_z)
    out = []
    for value in values:
        if value is None:
            out.append(None)
            continue
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("--restrict-X/--restrict-Y/--restrict-Z precisam ser números finitos")
        out.append(value)
    return tuple(out)


def fractional_coords_from_reference_cell(position, reference_cell):
    """Converte uma posição cartesiana em coordenadas fracionárias da célula de referência."""
    cell = np.asarray(reference_cell, dtype=float)
    return np.linalg.solve(cell.T, np.asarray(position, dtype=float))


def position_passes_fractional_restrictions(position, restrictions=None, reference_cell=None, tol=1e-12):
    """True se a posição satisfaz todos os limites fracionários definidos.

    A célula de referência é congelada antes de --match-cell-to-template para
    que os valores fornecidos pelo usuário mantenham o significado da célula
    cristalográfica inicial, mesmo se a célula de simulação crescer depois.
    """
    if restrictions is None or all(v is None for v in restrictions):
        return True
    if reference_cell is None:
        raise ValueError("reference_cell é obrigatório quando há --restrict-X/Y/Z")
    frac = fractional_coords_from_reference_cell(position, reference_cell)
    return all(limit is None or float(frac[ax]) >= float(limit) - tol
               for ax, limit in enumerate(restrictions))


def growth_eligible_indices(atoms, species_filter=None, restrictions=None, reference_cell=None):
    """Índices que podem participar da frente/conectividade do crescimento."""
    allowed = None if species_filter is None else set(species_filter)
    out = []
    for i, atom in enumerate(atoms):
        if allowed is not None and atom.symbol not in allowed:
            continue
        if not position_passes_fractional_restrictions(
            atom.position, restrictions, reference_cell
        ):
            continue
        out.append(int(i))
    return out


def filtered_growth_region_atoms(atoms, species_filter=None, restrictions=None, reference_cell=None):
    """Cópia contendo somente espécies/sítios permitidos para construir o template."""
    idx = growth_eligible_indices(
        atoms, species_filter=species_filter, restrictions=restrictions,
        reference_cell=reference_cell,
    )
    if not idx:
        raise RuntimeError(
            "Nenhum átomo do CIF-template satisfaz --template-site-species e --restrict-X/Y/Z."
        )
    out = atoms[idx]
    out.set_cell(atoms.cell, scale_atoms=False)
    out.pbc = atoms.pbc
    out.calc = None
    return out


def min_distance_to_selected_atoms_pbc(atoms, position, indices=None):
    """Menor distância PBC a um subconjunto; vazio => infinito."""
    d = pbc_distances_to_point(atoms, position)
    if indices is None:
        return float(np.min(d))
    idx = np.asarray(indices, dtype=int)
    if len(idx) == 0:
        return float("inf")
    return float(np.min(d[idx]))


def count_neighbors_pbc(atoms, position, cutoff, indices=None):
    """Conta vizinhos PBC, opcionalmente só entre átomos elegíveis ao crescimento."""
    d = pbc_distances_to_point(atoms, position)
    mask = (d > 1e-8) & (d <= cutoff)
    if indices is None:
        return int(np.sum(mask))
    idx = np.asarray(indices, dtype=int)
    if len(idx) == 0:
        return 0
    return int(np.sum(mask[idx]))


def estimate_template_growth_period(atoms, g):
    coords = np.sort(np.asarray(atoms.positions[:, g], dtype=float))
    if len(coords) < 2:
        return None

    unique = []
    for x in coords:
        if not unique or abs(x - unique[-1]) > 1e-4:
            unique.append(float(x))

    if len(unique) < 2:
        return None

    diffs = np.diff(unique)
    diffs = diffs[diffs > 1e-4]
    if len(diffs) == 0:
        return None

    return float((unique[-1] - unique[0]) + np.median(diffs))


def repeat_template_custom_growth_period(base, repeat, g, growth_period):
    repeat = tuple(int(v) for v in repeat)
    non_growth_repeat = list(repeat)
    n_growth = non_growth_repeat[g]
    non_growth_repeat[g] = 1

    base_ng = base.repeat(tuple(non_growth_repeat))
    out = Atoms(cell=base_ng.cell.array.copy(), pbc=base_ng.pbc)

    for i in range(n_growth):
        cp = base_ng.copy()
        cp.positions[:, g] += i * growth_period
        out += cp

    cell = out.cell.array.copy()
    axis_vec = cell[g].copy()
    norm = np.linalg.norm(axis_vec)
    if norm > 1e-12:
        cell[g] = axis_vec / norm * max(norm, growth_period * n_growth)
    else:
        cell[g, g] = growth_period * n_growth
    out.set_cell(cell, scale_atoms=False)
    out.pbc = base_ng.pbc
    return out


def axis_columns(atoms, g, tol=0.2):
    order = np.argsort(atoms.positions[:, g])
    groups = []
    current = []
    current_ref = None
    for idx in order:
        x = float(atoms.positions[idx, g])
        if current_ref is None or abs(x - current_ref) <= tol:
            current.append(int(idx))
            current_ref = x if current_ref is None else current_ref
        else:
            groups.append(current)
            current = [int(idx)]
            current_ref = x
    if current:
        groups.append(current)
    return groups



def _column_signature(atoms, indices, g, tol=1e-3):
    """Assinatura geométrica/química de uma coluna, ignorando a coordenada em ``g``.

    Permite detectar automaticamente padrões como A-B-A-B (crescimento em x) ou
    A-A-B-B-A-A-B-B (crescimento em y), sem assumir previamente o período.
    """
    axes = [ax for ax in range(3) if ax != int(g)]
    qtol = max(float(tol), 1e-8)
    rows = []
    for idx in indices:
        atom = atoms[int(idx)]
        quantized = tuple(
            int(np.rint(float(atom.position[ax]) / qtol)) for ax in axes
        )
        rows.append((str(atom.symbol),) + quantized)
    rows.sort()
    return tuple(rows)


def _minimal_period(sequence, equal, max_period=16):
    """Retorna o menor período que reproduz toda a sequência observada.

    Para evitar detectar período espúrio a partir de uma única ocorrência, exige
    pelo menos duas repetições potenciais (period <= n//2). Se isso não puder ser
    demonstrado pelo CIF, retorna None e o chamador usa o fallback configurado.
    """
    n = len(sequence)
    if n == 0:
        return None
    if n == 1:
        return 1

    limit = min(int(max_period), n // 2)
    for period in range(1, limit + 1):
        ok = True
        for i in range(period, n):
            if not equal(sequence[i], sequence[i - period]):
                ok = False
                break
        if ok:
            return period
    return None


def _detect_column_and_gap_periods(base, groups, g, column_tol, fallback_column_period):
    """Detecta os padrões periódicos das formas e dos gaps entre colunas.

    São tratados separadamente:
      * column_period: periodicidade da geometria transversal das colunas;
      * gap_pattern: periodicidade dos espaçamentos ao longo do eixo de crescimento.

    Isso resolve o caso do grafeno retangular: em x o gap é praticamente constante,
    enquanto em y ele alterna aproximadamente 1.42/0.71 Å. A implementação antiga
    usava apenas a mediana (1.065 Å em y), quebrando a conectividade da borda.
    """
    col_coords = np.asarray(
        [float(np.mean(base.positions[idxs, g])) for idxs in groups],
        dtype=float,
    )
    diffs = np.diff(col_coords)
    if len(diffs) == 0 or np.any(diffs <= 1e-8):
        raise ValueError(
            "não consegui estimar espaçamento positivo entre colunas do template"
        )

    signature_tol = max(1e-4, min(0.01, float(column_tol) * 0.05))
    signatures = [
        _column_signature(base, idxs, g, tol=signature_tol) for idxs in groups
    ]
    detected_column_period = _minimal_period(
        signatures,
        equal=lambda a, b: a == b,
        max_period=16,
    )

    fallback_column_period = max(1, int(fallback_column_period))
    if detected_column_period is None:
        column_period = min(fallback_column_period, len(groups))
        column_period_source = "fallback"
    else:
        column_period = int(detected_column_period)
        column_period_source = "auto"

    gap_tol = max(1e-5, min(0.02, float(column_tol) * 0.05))
    detected_gap_period = _minimal_period(
        list(map(float, diffs)),
        equal=lambda a, b: abs(float(a) - float(b)) <= gap_tol,
        max_period=16,
    )

    if detected_gap_period is None:
        gap_pattern = np.asarray([float(np.median(diffs))], dtype=float)
        gap_period_source = "median-fallback"
    else:
        q = int(detected_gap_period)
        values = []
        for phase in range(q):
            phase_values = diffs[phase::q]
            values.append(float(np.mean(phase_values)))
        gap_pattern = np.asarray(values, dtype=float)
        gap_period_source = "auto"

    return {
        "col_coords": col_coords,
        "column_period": int(column_period),
        "column_period_source": column_period_source,
        "gap_pattern": gap_pattern,
        "gap_period": int(len(gap_pattern)),
        "gap_period_source": gap_period_source,
        "median_gap": float(np.median(diffs)),
    }


def build_template_extend_columns_from_cif(
    cif_path,
    growth_axis,
    pbc_override=None,
    growth_direction="plus",
    ncols=30,
    column_period=2,
    column_tol=0.2,
    allowed_site_species=None,
    restrictions=None,
):
    """Estende o template pela borda preservando padrões periódicos reais.

    IMPORTANTE: mantém integralmente o suporte a ``--restrict_X/Y/Z`` e a
    ``--template-site-species``. Primeiro o CIF-template é filtrado pela região
    cristalográfica permitida; somente depois são detectadas e extrapoladas as
    colunas elegíveis.

    A implementação anterior forçava ``dx = median(diffs)`` para todas as novas
    colunas. Isso funciona quando os gaps são uniformes (como x no grafeno
    retangular), mas falha quando eles alternam (como y: ~1.42/0.71 Å).
    """
    base = read(cif_path)
    base_reference_cell = np.array(base.cell.array, dtype=float, copy=True)

    allowed = None
    if allowed_site_species and not (
        len(allowed_site_species) == 1
        and str(allowed_site_species[0]).lower() == "all"
    ):
        allowed = set(allowed_site_species)

    restrictions_active = (
        restrictions is not None and any(v is not None for v in restrictions)
    )
    if allowed is not None or restrictions_active:
        base = filtered_growth_region_atoms(
            base,
            species_filter=allowed,
            restrictions=restrictions,
            reference_cell=base_reference_cell,
        )
        print(
            f"Template-base filtrado para extend-columns: N={len(base)}; "
            f"espécies={allowed_site_species}; restrict_fractional={restrictions}"
        )

    g = AXIS[growth_axis.lower()]
    groups = axis_columns(base, g, tol=column_tol)
    if len(groups) < 2:
        raise ValueError(
            "template-build-mode extend-columns precisa de pelo menos 2 colunas "
            "no eixo de crescimento depois dos filtros de espécie/restrição"
        )

    pattern = _detect_column_and_gap_periods(
        base=base,
        groups=groups,
        g=g,
        column_tol=column_tol,
        fallback_column_period=column_period,
    )
    col_coords = pattern["col_coords"]
    detected_column_period = int(pattern["column_period"])
    gap_pattern = np.asarray(pattern["gap_pattern"], dtype=float)
    gap_period = int(pattern["gap_period"])

    ncols = max(0, int(ncols))
    out = base.copy()

    if growth_direction not in {"plus", "minus"}:
        raise ValueError(
            "extend-columns suporta --template-growth-direction plus ou minus"
        )

    nbase = len(groups)

    if growth_direction == "plus":
        new_coord = float(col_coords[-1])
        source_start = nbase - detected_column_period

        for j in range(1, ncols + 1):
            # Gap entre a última coluna conhecida e a coluna abstrata seguinte,
            # preservando a fase do padrão detectado no CIF.
            transition_index = nbase + j - 2
            gap = float(gap_pattern[transition_index % gap_period])
            new_coord += gap

            src_group_index = source_start + ((j - 1) % detected_column_period)
            src_group = groups[src_group_index]
            for idx in src_group:
                atom = base[idx]
                pos = atom.position.copy()
                pos[g] = new_coord
                out.append(Atom(atom.symbol, position=pos))

    else:
        new_coord = float(col_coords[0])

        for j in range(1, ncols + 1):
            transition_index = -j
            gap = float(gap_pattern[transition_index % gap_period])
            new_coord -= gap

            src_group_index = (-j) % detected_column_period
            src_group = groups[src_group_index]
            for idx in src_group:
                atom = base[idx]
                pos = atom.position.copy()
                pos[g] = new_coord
                out.append(Atom(atom.symbol, position=pos))

    cell = out.cell.array.copy()
    axis_vec = cell[g].copy()
    norm = np.linalg.norm(axis_vec)
    min_g = float(np.min(out.positions[:, g]))
    max_g = float(np.max(out.positions[:, g]))
    needed = (max_g - min(0.0, min_g)) + 5.0
    if norm > 1e-12 and needed > norm:
        cell[g] = axis_vec / norm * needed
        out.set_cell(cell, scale_atoms=False)

    apply_pbc_override_or_growth(out, g, pbc_override)

    gaps_text = ",".join(f"{value:.6f}" for value in gap_pattern)
    print(
        f"Template extend-columns: eixo={AXIS_NAME[g]} "
        f"periodo_colunas={detected_column_period}({pattern['column_period_source']}) "
        f"periodo_gaps={gap_period}({pattern['gap_period_source']}) "
        f"gaps=[{gaps_text}] Å; dx_mediana={pattern['median_gap']:.6f} Å; "
        f"colunas_base={len(groups)}, colunas_novas={ncols}, N_template={len(out)}; "
        f"restrict_fractional={restrictions}"
    )

    if (
        pattern["column_period_source"] == "auto"
        and int(column_period) != detected_column_period
    ):
        print(
            f"[INFO] --template-column-period={int(column_period)} não foi usado: "
            f"o CIF filtrado demonstra período geométrico {detected_column_period} "
            f"no eixo {AXIS_NAME[g]}. O valor informado fica como fallback."
        )

    return out

def build_template_sites_from_cif(
    cif_path,
    repeat,
    growth_axis,
    vacuum,
    pbc_override=None,
    center_growth=False,
    growth_period=None,
    auto_period=False,
    template_build_mode="repeat",
    growth_direction="plus",
    extend_ncols=30,
    column_period=2,
    column_tol=0.2,
    allowed_site_species=None,
    restrictions=None,
):
    g = AXIS[growth_axis.lower()]

    if template_build_mode == "extend-columns":
        template = build_template_extend_columns_from_cif(
            cif_path=cif_path,
            growth_axis=growth_axis,
            pbc_override=pbc_override,
            growth_direction=growth_direction,
            ncols=extend_ncols,
            column_period=column_period,
            column_tol=column_tol,
            allowed_site_species=allowed_site_species,
            restrictions=restrictions,
        )
        if center_growth and vacuum is not None and vacuum > 0:
            template.center(axis=g, vacuum=vacuum)
        return template

    base = read(cif_path)
    base_reference_cell = np.array(base.cell.array, dtype=float, copy=True)
    allowed = None
    if allowed_site_species and not (len(allowed_site_species) == 1 and str(allowed_site_species[0]).lower() == "all"):
        allowed = set(allowed_site_species)
    if allowed is not None or (restrictions is not None and any(v is not None for v in restrictions)):
        base = filtered_growth_region_atoms(
            base, species_filter=allowed, restrictions=restrictions,
            reference_cell=base_reference_cell,
        )
        print(
            f"Template-base filtrado: N={len(base)}; espécies={allowed_site_species}; "
            f"restrict_fractional={restrictions}"
        )

    period = None
    if growth_period is not None and growth_period > 0:
        period = float(growth_period)
    elif auto_period:
        period = estimate_template_growth_period(base, g)
        if period is not None:
            print(f"Template: período automático no eixo {AXIS_NAME[g]} = {period:.6f} Å")

    if period is not None:
        template = repeat_template_custom_growth_period(base, repeat, g, period)
    else:
        template = base.repeat(repeat)

    apply_pbc_override_or_growth(template, g, pbc_override)

    if center_growth and vacuum is not None and vacuum > 0:
        template.center(axis=g, vacuum=vacuum)

    return template


def species_for_template_site(site_symbol, species_list, flux_map, template_species_mode):
    if template_species_mode == "template":
        return [site_symbol] if site_symbol in species_list and flux_map.get(site_symbol, 0.0) > 0 else []
    return [s for s in species_list if flux_map.get(s, 0.0) > 0]


def direction_ahead_value(pos_g, origin_front, origin_back, growth_direction):
    if growth_direction == "plus":
        return float(pos_g - origin_front)
    if growth_direction == "minus":
        return float(origin_back - pos_g)
    if pos_g >= origin_front:
        return float(pos_g - origin_front)
    if pos_g <= origin_back:
        return float(origin_back - pos_g)
    return -1.0


def passes_growth_direction(pos_g, origin_front, origin_back, growth_direction, front_min):
    ahead0 = direction_ahead_value(pos_g, origin_front, origin_back, growth_direction)
    return ahead0 >= front_min, ahead0


def generate_candidates_from_template_sites(
    slab,
    template,
    g,
    species_list,
    calc,
    mu_map,
    flux_map,
    min_dist_allowed,
    temperature,
    front_min,
    front_max,
    occupancy_tol,
    connect_cutoff,
    min_neighbors,
    template_species_mode,
    allowed_site_species,
    front_mode="shell",
    growth_direction="plus",
    layer_tol=0.25,
    origin_front=None,
    origin_back=None,
    template_debug=False,
    vacancy_positions=None,
    vacancy_mode="fillable",
    growth_species=None,
    restrictions=None,
    reference_cell=None,
):
    e_before = slab.get_potential_energy()
    restrictions_active = restrictions is not None and any(v is not None for v in restrictions)
    growth_atom_indices = growth_eligible_indices(
        slab, species_filter=growth_species, restrictions=restrictions,
        reference_cell=reference_cell,
    ) if (growth_species is not None or restrictions_active) else None
    if growth_atom_indices is not None and len(growth_atom_indices) == 0:
        raise RuntimeError(
            "Nenhum átomo do cristal em crescimento satisfaz espécie/restrições; "
            "verifique --species/--growth-origin-species e --restrict-X/Y/Z."
        )
    if growth_atom_indices is None:
        current_front = top_coord(slab, g)
        current_back = bottom_coord(slab, g)
    else:
        current_front = float(np.max(slab.positions[np.asarray(growth_atom_indices), g]))
        current_back = float(np.min(slab.positions[np.asarray(growth_atom_indices), g]))
    if origin_front is None:
        origin_front = current_front
    if origin_back is None:
        origin_back = current_back

    allowed = None
    if allowed_site_species and not (len(allowed_site_species) == 1 and allowed_site_species[0].lower() == "all"):
        allowed = set(allowed_site_species)

    debug = {"total": 0, "species_filter": 0, "restriction": 0, "direction": 0, "global_front": 0, "occupied": 0, "connection": 0, "min_dist": 0, "pre_shell": 0, "post_shell": 0, "tested_energy": 0}
    site_pool = []

    for site_index, site_atom in enumerate(template):
        debug["total"] += 1
        if allowed is not None and site_atom.symbol not in allowed:
            debug["species_filter"] += 1
            continue
        if not position_passes_fractional_restrictions(
            site_atom.position, restrictions, reference_cell
        ):
            debug["restriction"] += 1
            continue

        pos = wrap_pbc_axes_to_cell(slab, site_atom.position.copy())
        pos_g = float(pos[g])

        ok_dir, ahead_origin = passes_growth_direction(pos_g, origin_front, origin_back, growth_direction, front_min)
        if not ok_dir:
            debug["direction"] += 1
            continue

        if front_mode == "global":
            if growth_direction == "plus":
                ahead_current = pos_g - current_front
            elif growth_direction == "minus":
                ahead_current = current_back - pos_g
            else:
                ahead_current = min(abs(pos_g - current_front), abs(pos_g - current_back))
            if ahead_current < front_min or ahead_current > front_max:
                debug["global_front"] += 1
                continue
        else:
            ahead_current = pos_g - current_front if growth_direction == "plus" else current_back - pos_g

        occ_atom_dist = min_distance_to_selected_atoms_pbc(slab, pos, growth_atom_indices)
        occ_vac_dist = min_distance_to_vacancies_pbc(slab, vacancy_positions, pos)

        # fillable: a vacância continua sendo um sítio candidato e pode ser curada.
        # frozen: o sítio da vacância é removido permanentemente do conjunto de
        # candidatos. Usamos a mesma tolerância geométrica de ocupação do template
        # para reconhecer o sítio, respeitando as condições periódicas.
        if vacancy_mode == "frozen" and occ_vac_dist < occupancy_tol:
            debug["occupied"] += 1
            continue

        occ_dist = occ_atom_dist
        if occ_dist < occupancy_tol:
            debug["occupied"] += 1
            continue

        n_neigh = count_neighbors_pbc(slab, pos, connect_cutoff, growth_atom_indices)
        if n_neigh < min_neighbors:
            debug["connection"] += 1
            continue

        if occ_dist < min_dist_allowed:
            debug["min_dist"] += 1
            continue

        site_pool.append({"site_index": site_index, "site_symbol": site_atom.symbol, "pos": pos, "pos_g": pos_g, "ahead_origin": ahead_origin, "ahead_current": float(ahead_current), "occ_dist": float(occ_dist), "n_neigh": int(n_neigh)})

    debug["pre_shell"] = len(site_pool)

    if front_mode == "shell" and site_pool:
        if growth_direction == "plus":
            layer_coord = min(r["pos_g"] for r in site_pool)
            site_pool = [r for r in site_pool if r["pos_g"] <= layer_coord + layer_tol]
        elif growth_direction == "minus":
            layer_coord = max(r["pos_g"] for r in site_pool)
            site_pool = [r for r in site_pool if r["pos_g"] >= layer_coord - layer_tol]
        else:
            layer_ahead = min(r["ahead_origin"] for r in site_pool)
            site_pool = [r for r in site_pool if r["ahead_origin"] <= layer_ahead + layer_tol]

    # O rank 0 é a fonte única da lista geométrica de sítios. Isso evita que
    # pequenas diferenças numéricas ou de estado entre processos façam cada rank
    # avaliar uma posição diferente na mesma chamada coletiva ao GPAW.
    site_pool = mpi_broadcast_object(site_pool)
    debug["post_shell"] = len(site_pool)

    candidates = []
    n_tested = 0
    for r in site_pool:
        trial_species = species_for_template_site(r["site_symbol"], species_list, flux_map, template_species_mode)
        for symbol in trial_species:
            mu = mu_map.get(symbol, 0.0)
            species_flux = flux_map.get(symbol, 0.0)
            trial, _ = make_deposition_trial(slab, symbol, r["pos"], calc)
            try:
                e_unrelaxed = trial.get_potential_energy()
            except Exception as exc:
                reraise_mpi_atoms_mismatch(exc, "energia candidata template")
                print(f"[WARN] energia candidata template falhou para {symbol}: {exc}")
                continue

            n_tested += 1
            debug["tested_energy"] += 1
            e_ads = e_unrelaxed - e_before - mu
            if not adsorption_energy_allowed(e_ads):
                if template_debug:
                    limit = getattr(adsorption_energy_allowed, "max_energy_eV", None)
                    print(
                        f"[TEMPLATE DEBUG] candidato rejeitado por energia: "
                        f"site={r['site_index']} species={symbol} "
                        f"Eads={e_ads:+.6f} eV > limite={limit:+.6f} eV"
                    )
                continue
            mode = f"template_site:index={r['site_index']}:site={r['site_symbol']}:neigh={r['n_neigh']}:ahead0={r['ahead_origin']:.3f}:mode={front_mode}"
            # Para candidatos de template, anchor_index identifica persistentemente
            # o índice do sítio cristalográfico. Isso também permite rastrear
            # vacâncias frozen no CSV.
            candidates.append(Candidate(symbol, r["pos"], int(r["site_index"]), r["occ_dist"], e_unrelaxed, e_ads, species_flux, mode, e_ads))

    if template_debug:
        print(
            "[TEMPLATE DEBUG] "
            f"total={debug['total']} species_filter={debug['species_filter']} "
            f"restriction={debug['restriction']} "
            f"direction={debug['direction']} global_front={debug['global_front']} "
            f"occupied={debug['occupied']} connection={debug['connection']} "
            f"min_dist={debug['min_dist']} pre_shell={debug['pre_shell']} "
            f"post_shell={debug['post_shell']} tested_energy={debug['tested_energy']}"
        )
        if site_pool:
            coords = [r["pos_g"] for r in site_pool]
            print(f"[TEMPLATE DEBUG] camada candidata {AXIS_NAME[g]}: min={min(coords):.6f} max={max(coords):.6f} n={len(coords)}")

    return candidates, len(site_pool), n_tested


def choose_template_deposition_candidate(
    slab,
    template,
    g,
    species_list,
    calc,
    mu_map,
    flux_map,
    min_dist_allowed,
    temperature,
    front_min,
    front_max,
    occupancy_tol,
    connect_cutoff,
    min_neighbors,
    template_species_mode,
    allowed_site_species,
    front_mode,
    growth_direction,
    layer_tol,
    origin_front,
    origin_back,
    template_debug,
    vacancy_positions=None,
    vacancy_mode="fillable",
    growth_species=None,
    restrictions=None,
    reference_cell=None,
):
    candidates, n_possible_sites, n_tested = generate_candidates_from_template_sites(
        slab=slab,
        template=template,
        g=g,
        species_list=species_list,
        calc=calc,
        mu_map=mu_map,
        flux_map=flux_map,
        min_dist_allowed=min_dist_allowed,
        temperature=temperature,
        front_min=front_min,
        front_max=front_max,
        occupancy_tol=occupancy_tol,
        connect_cutoff=connect_cutoff,
        min_neighbors=min_neighbors,
        template_species_mode=template_species_mode,
        allowed_site_species=allowed_site_species,
        front_mode=front_mode,
        growth_direction=growth_direction,
        layer_tol=layer_tol,
        origin_front=origin_front,
        origin_back=origin_back,
        template_debug=template_debug,
        vacancy_positions=vacancy_positions,
        vacancy_mode=vacancy_mode,
        growth_species=growth_species,
        restrictions=restrictions,
        reference_cell=reference_cell,
    )

    if candidates:
        return boltzmann_choose_candidate(candidates, temperature), n_possible_sites, n_tested
    return None, n_possible_sites, n_tested


def compute_template_deposition_event(
    slab,
    template,
    g,
    species_list,
    calc,
    mu_map,
    flux_map,
    n0,
    deposition_rate,
    mobile_radius,
    fmax,
    relax_steps,
    optimizer,
    min_dist_allowed,
    temperature,
    front_min,
    front_max,
    occupancy_tol,
    connect_cutoff,
    min_neighbors,
    template_species_mode,
    allowed_site_species,
    front_mode,
    growth_direction,
    layer_tol,
    origin_front,
    origin_back,
    template_debug,
    no_relax_first_added=False,
    impurity_specs=None,
    vacancy_positions=None,
    vacancy_mode="fillable",
    growth_species=None,
    restrictions=None,
    reference_cell=None,
):
    chosen_candidate, n_anchors, n_tested = choose_template_deposition_candidate(
        slab=slab,
        template=template,
        g=g,
        species_list=species_list,
        calc=calc,
        mu_map=mu_map,
        flux_map=flux_map,
        min_dist_allowed=min_dist_allowed,
        temperature=temperature,
        front_min=front_min,
        front_max=front_max,
        occupancy_tol=occupancy_tol,
        connect_cutoff=connect_cutoff,
        min_neighbors=min_neighbors,
        template_species_mode=template_species_mode,
        allowed_site_species=allowed_site_species,
        front_mode=front_mode,
        growth_direction=growth_direction,
        layer_tol=layer_tol,
        origin_front=origin_front,
        origin_back=origin_back,
        template_debug=template_debug,
        vacancy_positions=vacancy_positions,
        vacancy_mode=vacancy_mode,
        growth_species=growth_species,
        restrictions=restrictions,
        reference_cell=reference_cell,
    )

    if chosen_candidate is None:
        return None

    e_before = slab.get_potential_energy()
    intended_species = chosen_candidate.species
    impurity_kind = choose_impurity(impurity_specs)
    actual_species = intended_species if impurity_kind is None else impurity_kind

    if actual_species == "vac":
        trial = slab.copy()
        attach_calculator(trial, calc)
        added = -1
        mobile = []
        e_after = e_before
        delta_e = 0.0
        e_ads_unrelaxed = 0.0
        event_type = "template_vacancy"
        print(f"[INFO] vacância sorteada por --impurity em posição template; espécie que seria depositada: {intended_species}.")
    else:
        trial, added = make_deposition_trial(slab, actual_species, chosen_candidate.position, calc)
        is_first_added_atom = (len(slab) == n0)

        try:
            e_unrelaxed_actual = chosen_candidate.e_unrelaxed if actual_species == intended_species else trial.get_potential_energy()
        except Exception as exc:
            reraise_mpi_atoms_mismatch(exc, "energia não relaxada da deposição template")
            print(f"[WARN] energia não relaxada da deposição template falhou para {actual_species}: {exc}")
            return None

        if no_relax_first_added and is_first_added_atom:
            mobile = []
            attach_calculator(trial, calc)
            e_after = trial.get_potential_energy()
            print("[INFO] primeira deposição: relaxação local pulada por --no-relax-first-added.")
        else:
            mobile = get_local_mobile_indices(trial, [added], mobile_radius)
            try:
                relax_local(trial, mobile, optimizer, fmax, relax_steps)
                attach_calculator(trial, calc)
                e_after = trial.get_potential_energy()
            except Exception as exc:
                reraise_mpi_atoms_mismatch(exc, "relaxação template")
                print(f"[WARN] relaxação template falhou: {exc}")
                return None

        mu = mu_map.get(actual_species, 0.0)
        delta_e = e_after - e_before - mu
        e_ads_unrelaxed = e_unrelaxed_actual - e_before - mu
        event_type = "template_deposition" if impurity_kind is None else "template_impurity"
        if impurity_kind is not None:
            print(f"[INFO] impureza sorteada por --impurity: {actual_species} no lugar de {intended_species}.")

    effective_rate = kinetic_event_rate(
        deposition_rate, delta_e, temperature,
        barrier_eV=getattr(compute_template_deposition_event, "kinetic_barrier_eV", 0.0),
        bep_alpha=getattr(compute_template_deposition_event, "kinetic_bep_alpha", 0.5),
    )

    return Event(
        event_type=event_type,
        deposited_species=actual_species,
        rate=effective_rate,
        trial=trial,
        delta_e=delta_e,
        e_before=e_before,
        e_after=e_after,
        e_ads_unrelaxed=e_ads_unrelaxed,
        position=chosen_candidate.position,
        added_index=added,
        anchor_index=chosen_candidate.anchor_index,
        mobile_count=len(mobile),
        min_dist=chosen_candidate.min_dist,
        n_anchors=n_anchors,
        n_tested=n_tested,
        chosen_probability=chosen_candidate.probability,
        anchor_mode=chosen_candidate.anchor_mode,
        species_flux=chosen_candidate.species_flux,
        intended_species=intended_species,
        impurity_kind=actual_species if impurity_kind is not None else "none",
        is_vacancy=(actual_species == "vac"),
    )


def enlarge_cell_to_include_template(atoms, template, g, margin):
    if template is None or margin is None or margin < 0:
        return atoms

    cell = atoms.cell.array.copy()
    axis_vec = cell[g].copy()
    norm = np.linalg.norm(axis_vec)
    if norm <= 1e-12:
        return atoms

    max_needed = max(float(np.max(atoms.positions[:, g])), float(np.max(template.positions[:, g]))) + float(margin)
    min_needed = min(float(np.min(atoms.positions[:, g])), float(np.min(template.positions[:, g]))) - float(margin)
    required_length = max(max_needed, norm, max_needed - min(0.0, min_needed))
    if required_length > norm:
        cell[g] = axis_vec / norm * required_length
        atoms.set_cell(cell, scale_atoms=False)
    return atoms



def gamma_initial_symmetric_slab(e_slab, n_atoms, e_bulk_atom, area):
    return (e_slab - n_atoms * e_bulk_atom) / (2.0 * area)


def gamma_effective_growth_top(e_now, n0, n_added, e_bulk_atom, area):
    return (e_now - n0 * e_bulk_atom) / area


@dataclass
class Candidate:
    species: str
    position: np.ndarray
    anchor_index: int
    min_dist: float
    e_unrelaxed: float
    e_ads_unrelaxed: float
    species_flux: float
    anchor_mode: str
    selection_energy: float = 0.0
    boltzmann_weight: float = 0.0
    probability: float = 0.0


@dataclass
class Event:
    event_type: str
    deposited_species: str
    rate: float
    trial: Atoms
    delta_e: float
    e_before: float
    e_after: float
    e_ads_unrelaxed: float
    position: np.ndarray
    added_index: int
    anchor_index: int
    mobile_count: int
    min_dist: float
    n_anchors: int
    n_tested: int
    chosen_probability: float
    anchor_mode: str
    species_flux: float
    intended_species: str = ""
    impurity_kind: str = "none"
    is_vacancy: bool = False


def make_deposition_trial(slab, symbol, position, calc):
    trial = slab.copy()
    trial.append(Atom(symbol, position=position))
    # Fonte única: o rank 0 transmite números, posições, célula, PBC e arrays.
    # Assim o GPAW recebe exatamente o mesmo Atoms em todos os processos.
    trial = mpi_broadcast_atoms(trial)
    attach_calculator(trial, calc)
    return trial, len(trial) - 1


def parse_flux_species(flux_species, species_list):
    if not flux_species:
        return {s: 1.0 for s in species_list}

    if len(flux_species) % 2 != 0:
        raise ValueError("--flux-species precisa vir em pares: elemento valor")

    out = {s: 0.0 for s in species_list}
    for i in range(0, len(flux_species), 2):
        out[flux_species[i]] = float(flux_species[i + 1])
    return out


def parse_mu_species(mu_species):
    if not mu_species:
        return {}
    if len(mu_species) % 2 != 0:
        raise ValueError("--mu-species precisa vir em pares: elemento valor")
    out = {}
    for i in range(0, len(mu_species), 2):
        out[mu_species[i]] = float(mu_species[i + 1])
    return out


def normalize_impurity_symbol(symbol):
    s = str(symbol).strip()
    if not s:
        raise ValueError("Símbolo vazio em --impurity")
    if s.lower() in {"vac", "vacancy", "vacancia", "vacância"}:
        return "vac"
    return s


def parse_impurities(impurity_values):
    """
    Lê --impurity como pares símbolo probabilidade.

    Exemplos:
      --impurity vac 0.1
      --impurity N 0.1
      --impurity vac 0.1 N 0.1 B 0.1

    As probabilidades são frações mutuamente exclusivas por evento de deposição.
    Se a soma for 0.3, então 30% dos eventos viram impureza/vacância e 70% seguem normais.
    """
    if not impurity_values:
        return []
    if len(impurity_values) % 2 != 0:
        raise ValueError("--impurity precisa vir em pares: espécie probabilidade. Ex.: --impurity vac 0.1 N 0.1")

    acc = {}
    order = []
    for i in range(0, len(impurity_values), 2):
        symbol = normalize_impurity_symbol(impurity_values[i])
        try:
            prob = float(impurity_values[i + 1])
        except ValueError as exc:
            raise ValueError(f"Probabilidade inválida em --impurity para {symbol}: {impurity_values[i + 1]}") from exc

        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"Probabilidade de --impurity para {symbol} precisa estar entre 0 e 1")
        if symbol not in acc:
            acc[symbol] = 0.0
            order.append(symbol)
        acc[symbol] += prob

    total = sum(acc.values())
    if total > 1.0 + 1e-12:
        raise ValueError(f"A soma das probabilidades em --impurity não pode passar de 1. Soma atual = {total:.6f}")

    return [(symbol, acc[symbol]) for symbol in order if acc[symbol] > 0.0]


def choose_impurity(impurity_specs):
    """Retorna None para deposição normal, 'vac' para vacância ou o símbolo da impureza."""
    if not impurity_specs:
        return None

    r = mpi_random()
    acc = 0.0
    for symbol, prob in impurity_specs:
        acc += prob
        if r < acc:
            return symbol
    return None


def impurity_summary(impurity_specs):
    if not impurity_specs:
        return "desligado"
    total = sum(prob for _, prob in impurity_specs)
    parts = [f"{symbol}:{prob:.4f}" for symbol, prob in impurity_specs]
    parts.append(f"normal:{max(0.0, 1.0 - total):.4f}")
    return ", ".join(parts)


def write_vacancies_csv(path, vacancy_records):
    if not MPI_MASTER:
        return
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vacancy_index", "step", "intended_species", "x_A", "y_A", "z_A", "anchor_mode", "anchor_index"])
        for i, rec in enumerate(vacancy_records, start=1):
            pos = rec["position"]
            w.writerow([
                i,
                rec.get("step", ""),
                rec.get("intended_species", ""),
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                rec.get("anchor_mode", ""),
                rec.get("anchor_index", ""),
            ])


def copy_atoms_for_cif_output(atoms, wrap_mode="all"):
    """
    Retorna uma cópia apropriada para escrita em CIF.

    O objetivo é evitar CIFs com coordenadas fracionárias fora de [0, 1),
    que alguns visualizadores como VESTA/OVITO podem abrir incorretamente.

    wrap_mode:
      all  -> normaliza x, y e z fracionários para [0, 1). Recomendado para visualização.
      pbc  -> normaliza apenas os eixos marcados como periódicos em atoms.pbc.
      none -> não altera as coordenadas antes de escrever o CIF.

    Importante: a estrutura usada na simulação não é modificada; somente a
    cópia escrita no arquivo CIF é transformada por translações de rede.
    """
    out = atoms.copy()

    if wrap_mode == "none":
        return out

    if wrap_mode not in {"all", "pbc"}:
        raise ValueError("wrap_mode precisa ser 'all', 'pbc' ou 'none'")

    frac = out.get_scaled_positions(wrap=False)

    if wrap_mode == "all":
        axes = [0, 1, 2]
    else:
        axes = [ax for ax in range(3) if bool(out.pbc[ax])]

    for ax in axes:
        frac[:, ax] = frac[:, ax] - np.floor(frac[:, ax])
        frac[np.isclose(frac[:, ax], 1.0, atol=1e-12), ax] = 0.0
        frac[np.isclose(frac[:, ax], 0.0, atol=1e-12), ax] = 0.0

    out.set_scaled_positions(frac)
    return out




def atoms_for_safe_output(atoms):
    """Retorna cópia sem resultados obsoletos do calculador.

    Calculadores reutilizáveis como CHGNet podem manter em ``calc.results``
    vetores de forças do último candidato testado. Quando esse candidato possui
    N+1 átomos, mas o slab aceito ainda possui N, o escritor EXTXYZ tenta salvar
    um array com dimensão incompatível. A cópia abaixo preserva estrutura,
    célula, PBC, tags e arrays atômicos válidos, mas remove o calculador e
    descarta arrays cujo primeiro eixo não coincide com o número de átomos.
    """
    out = atoms.copy()
    out.calc = None
    n = len(out)
    for name in list(out.arrays):
        arr = out.arrays[name]
        if getattr(arr, "ndim", 0) > 0 and len(arr) != n:
            del out.arrays[name]
    return out


def write_xyz_safe(path, atoms):
    """Escreve XYZ/EXTXYZ no rank 0, sem resultados antigos do calculador."""
    if not MPI_MASTER:
        return
    clean = atoms_for_safe_output(atoms)
    # write() é decorada com ase.parallel.parallel_function. Como esta função
    # já roda somente no rank 0, precisamos desativar o MPI interno do ASE;
    # caso contrário o root transmite (exceção, resultado) enquanto os demais
    # ranks já estão no próximo broadcast da geometria.
    write(path, clean, write_results=False, parallel=False)

def write_cif_compatible(path, atoms, wrap_mode="all"):
    """Escreve CIF no rank 0, preservando o objeto atoms original."""
    if not MPI_MASTER:
        return
    cif_atoms = copy_atoms_for_cif_output(atoms, wrap_mode=wrap_mode)
    # Mesma razão de write_xyz_safe(): a escrita é deliberadamente serial.
    write(path, cif_atoms, format="cif", parallel=False)


def adsorption_energy_allowed(e_ads):
    """Retorna True quando a energia de adsorção passa pelo corte absoluto.

    O limite é configurado em tempo de execução por
    adsorption_energy_allowed.max_energy_eV. Quando None, nenhum corte é
    aplicado. O teste usa a energia de adsorção não relaxada, antes da seleção
    de Boltzmann, evitando aceitar um sítio extremamente desfavorável apenas
    porque ele é o único candidato restante.
    """
    limit = getattr(adsorption_energy_allowed, "max_energy_eV", None)
    return limit is None or float(e_ads) <= float(limit)


def boltzmann_choose_candidate(candidates, temperature):
    if len(candidates) == 1:
        candidates[0].boltzmann_weight = candidates[0].species_flux
        candidates[0].probability = 1.0
        return candidates[0]

    if temperature <= 0:
        best = min(candidates, key=lambda c: c.selection_energy)
        best.probability = 1.0
        best.boltzmann_weight = 1.0
        return best

    energies = np.array([c.selection_energy for c in candidates], dtype=float)
    emin = float(np.min(energies))
    beta = 1.0 / (KB_EV * temperature)

    weights = np.array([
        c.species_flux * math.exp(-(c.e_ads_unrelaxed - emin) * beta)
        for c in candidates
    ], dtype=float)

    total = float(np.sum(weights))
    if total <= 0 or not np.isfinite(total):
        best = min(candidates, key=lambda c: c.selection_energy)
        best.probability = 1.0
        best.boltzmann_weight = 1.0
        return best

    probs = weights / total
    for c, w, p in zip(candidates, weights, probs):
        c.boltzmann_weight = float(w)
        c.probability = float(p)

    idx = int(mpi_np_choice_index(len(candidates), probs))
    return candidates[idx]



def grid_cell_indices_for_point(frac_value, n):
    return min(max(int(frac_value * n), 0), n - 1)


def atoms_in_surface_grid_cell(
    atoms,
    g,
    i1,
    i2,
    n1,
    n2,
    surface_window,
    include_species: Optional[Set[str]] = None,
):
    """
    Retorna átomos pertencentes a uma célula da malha no plano da superfície.

    A malha é definida nas coordenadas fracionárias dos dois eixos periódicos
    perpendiculares ao eixo de crescimento:
      growth-axis x -> plano yz
      growth-axis y -> plano xz
      growth-axis z -> plano xy

    Primeiro filtra por espécie, depois pega apenas átomos próximos ao topo
    local/global da superfície usando surface_window.
    """
    pa = plane_axes_from_growth(g)
    frac = atoms.get_scaled_positions(wrap=True)

    idx = []
    for k, atom in enumerate(atoms):
        if include_species is not None and atom.symbol not in include_species:
            continue

        c1 = grid_cell_indices_for_point(frac[k, pa[0]], n1)
        c2 = grid_cell_indices_for_point(frac[k, pa[1]], n2)
        if c1 == i1 and c2 == i2:
            idx.append(k)

    if not idx:
        return []

    # Mantém só a parte superior da célula da malha.
    cell_top = max(atoms.positions[k, g] for k in idx)
    idx = [k for k in idx if atoms.positions[k, g] >= cell_top - surface_window]
    return idx


def random_point_in_surface_grid_cell(atoms, g, i1, i2, n1, n2):
    """
    Sorteia um ponto aleatório dentro de uma célula da malha da superfície.
    O retorno ainda não tem a coordenada de altura definida; ela será ajustada
    depois com base no átomo/topo escolhido.
    """
    pa = plane_axes_from_growth(g)
    frac = np.zeros(3, dtype=float)

    frac[pa[0]] = (i1 + mpi_random()) / n1
    frac[pa[1]] = (i2 + mpi_random()) / n2
    frac[g] = 0.0

    return frac @ atoms.cell.array


def choose_anchor_for_grid_point(atoms, point, g, cell_indices):
    """
    Escolhe, dentro da célula da malha, o átomo mais próximo do ponto sorteado
    no plano da superfície. Se a célula tiver átomos, esse é o átomo usado
    como referência de altura.
    """
    if not cell_indices:
        return None

    pa = plane_axes_from_growth(g)
    best = None
    best_d2 = None

    for k in cell_indices:
        d = atoms.positions[k] - point
        d[g] = 0.0
        d2 = float(d[pa[0]] ** 2 + d[pa[1]] ** 2)
        if best is None or d2 < best_d2:
            best = k
            best_d2 = d2

    return best


def generate_candidate_in_grid_cell(
    atoms,
    g,
    i1,
    i2,
    n1,
    n2,
    height_min,
    height_max,
    surface_window,
    local_top_radius,
    anchor_species,
):
    """
    Regra de deposição por região da malha:
      1) para cada região da malha, sorteia um ponto aleatório no plano;
      2) se existir átomo-âncora elegível naquela região, escolhe o mais
         próximo do ponto sorteado no plano;
      3) com âncora: deposita no ponto sorteado, com altura = altura_da_âncora + delta;
      4) sem âncora: NÃO troca para o átomo mais alto da célula. Mantém o ponto
         sorteado e coloca delta acima do topo local ao redor desse ponto;
      5) se não houver átomo no raio local_top_radius, usa o topo global.

    Assim a deposição pode ocorrer mesmo em uma região sem átomo/âncora, sem
    deslocar lateralmente a posição para outro átomo.
    """
    point = random_point_in_surface_grid_cell(atoms, g, i1, i2, n1, n2)

    eligible = atoms_in_surface_grid_cell(
        atoms=atoms,
        g=g,
        i1=i1,
        i2=i2,
        n1=n1,
        n2=n2,
        surface_window=surface_window,
        include_species=anchor_species,
    )

    anchor = choose_anchor_for_grid_point(atoms, point, g, eligible)

    if anchor is not None:
        htop = float(atoms.positions[anchor, g])
        anchor_index = int(anchor)
        mode = "grid_anchor"
    else:
        # Sem âncora: mantém o ponto aleatório da célula e usa a altura local.
        # local_top_near_surface_point() retorna o topo global se não achar átomos
        # dentro de local_top_radius, então ainda funciona em região vazia.
        htop = local_top_near_surface_point(atoms, point, g, local_top_radius)
        anchor_index = -1
        mode = "grid_no_anchor_point"

    pos = point.copy()
    pos[g] = htop + mpi_uniform(height_min, height_max)
    pos = wrap_surface_plane_to_cell(atoms, pos, g)
    return pos, anchor_index, mode


def generate_candidates_with_anchor_mode(
    slab,
    g,
    species_list,
    calc,
    mu_map,
    flux_map,
    n_candidates_per_species,
    surface_grid,
    surface_window,
    lateral_radius,
    height_min,
    height_max,
    min_dist_allowed,
    local_top_radius,
    anchor_species,
    anchor_mode_name,
    vacancy_positions=None,
):
    """
    Gera candidatos cobrindo TODA a malha da superfície.

    Diferente da versão antiga, não sorteia uma âncora global e depois joga o
    átomo perto dela. Agora cada célula da surface-grid recebe tentativas de
    deposição. Isso evita buracos sistemáticos onde nenhuma âncora foi sorteada.

    n_candidates_per_species é mantido por compatibilidade e agora significa:
      número de pontos aleatórios por célula da malha, por espécie.
    """
    n1, n2 = surface_grid
    if n1 <= 0 or n2 <= 0:
        raise ValueError("--surface-grid precisa ser maior que zero")

    e_before = slab.get_potential_energy()
    candidates = []
    n_tested = 0
    n_regions = n1 * n2
    trials_per_cell = max(1, int(n_candidates_per_species))

    for symbol in species_list:
        species_flux = flux_map.get(symbol, 0.0)
        if species_flux <= 0:
            continue

        mu = mu_map.get(symbol, 0.0)

        for i1 in range(n1):
            for i2 in range(n2):
                for _ in range(trials_per_cell):
                    pos, anchor, grid_mode = generate_candidate_in_grid_cell(
                        atoms=slab,
                        g=g,
                        i1=i1,
                        i2=i2,
                        n1=n1,
                        n2=n2,
                        height_min=height_min,
                        height_max=height_max,
                        surface_window=surface_window,
                        local_top_radius=local_top_radius,
                        anchor_species=anchor_species,
                    )

                    dmin_atom = min_distance_to_atoms(slab, pos)
                    # Sítios de vacância podem ser preenchidos posteriormente.
                    dmin = dmin_atom
                    if dmin < min_dist_allowed:
                        continue

                    trial, _ = make_deposition_trial(slab, symbol, pos, calc)

                    try:
                        e_unrelaxed = trial.get_potential_energy()
                    except Exception as exc:
                        reraise_mpi_atoms_mismatch(exc, "energia candidata de malha")
                        print(f"[WARN] energia candidata falhou para {symbol}: {exc}")
                        continue

                    n_tested += 1
                    e_ads = e_unrelaxed - e_before - mu
                    if not adsorption_energy_allowed(e_ads):
                        continue
                    mode = f"{anchor_mode_name}:{grid_mode}:{i1},{i2}"
                    candidates.append(
                        Candidate(
                            symbol,
                            pos,
                            anchor,
                            dmin,
                            e_unrelaxed,
                            e_ads,
                            species_flux,
                            mode,
                            e_ads,
                        )
                    )

    return candidates, n_regions, n_tested



def local_coordination_for_position(atoms, position, cutoff):
    d = np.linalg.norm(atoms.positions - position, axis=1)
    return int(np.sum((d > 1e-8) & (d <= cutoff)))


def generate_candidate_side_attach(
    atoms,
    anchor_index,
    g,
    side_radius_min,
    side_radius_max,
    side_height_jitter,
):
    """
    Gera uma tentativa de anexação lateral para representar crescimento tipo gota.

    A deposição física continua sendo aleatória/uniforme na superfície quando o modo
    de malha é usado. Este modo representa uma etapa efetiva de difusão superficial:
    um átomo depositado encontra uma ilha/gota existente e gruda preferencialmente
    na lateral, não diretamente acima dela.
    """
    pa = plane_axes_from_growth(g)
    base = atoms.positions[anchor_index].copy()

    theta = mpi_uniform(0.0, 2.0 * math.pi)
    r = mpi_uniform(side_radius_min, side_radius_max)

    pos = base.copy()
    pos[pa[0]] += r * math.cos(theta)
    pos[pa[1]] += r * math.sin(theta)
    pos[g] += mpi_uniform(-side_height_jitter, side_height_jitter)
    pos = wrap_surface_plane_to_cell(atoms, pos, g)
    return pos


def generate_droplet_side_candidates(
    slab,
    g,
    species_list,
    calc,
    mu_map,
    flux_map,
    n0,
    n_candidates_per_species,
    min_dist_allowed,
    side_radius_min,
    side_radius_max,
    side_height_jitter,
    droplet_anchor_mode,
    cn_cutoff,
    coord_bonus,
    vertical_penalty,
    vacancy_positions=None,
):
    """
    Candidatos de gota: escolhe átomos-âncora e tenta posições laterais.

    selection_energy = energia_adsorção - coord_bonus * coordenação + penalização vertical.
    Assim, a seleção favorece posições com mais vizinhos laterais e evita empilhamento
    vertical artificial. e_ads_unrelaxed continua sendo registrada sem esses termos.
    """
    e_before = slab.get_potential_energy()
    candidates = []
    n_tested = 0

    added_indices = list(range(n0, len(slab)))
    if droplet_anchor_mode == "added" and added_indices:
        anchors = added_indices
        label = "droplet_added"
    else:
        anchors = get_surface_indices(slab, g, surface_window=3.0)
        label = "droplet_surface"

    if not anchors:
        return [], 0, 0

    trials_per_species = max(1, int(n_candidates_per_species))
    top_now = top_coord(slab, g)

    for symbol in species_list:
        species_flux = flux_map.get(symbol, 0.0)
        if species_flux <= 0:
            continue

        mu = mu_map.get(symbol, 0.0)

        for _ in range(trials_per_species):
            anchor = int(mpi_choice(anchors))
            pos = generate_candidate_side_attach(
                atoms=slab,
                anchor_index=anchor,
                g=g,
                side_radius_min=side_radius_min,
                side_radius_max=side_radius_max,
                side_height_jitter=side_height_jitter,
            )

            dmin_atom = min_distance_to_atoms(slab, pos)
            # Sítios de vacância podem ser preenchidos posteriormente.
            dmin = dmin_atom
            if dmin < min_dist_allowed:
                continue

            trial, _ = make_deposition_trial(slab, symbol, pos, calc)
            try:
                e_unrelaxed = trial.get_potential_energy()
            except Exception as exc:
                reraise_mpi_atoms_mismatch(exc, "energia candidata tipo gota")
                print(f"[WARN] energia candidata gota falhou para {symbol}: {exc}")
                continue

            n_tested += 1
            e_ads = e_unrelaxed - e_before - mu
            if not adsorption_energy_allowed(e_ads):
                continue
            cn_local = local_coordination_for_position(slab, pos, cn_cutoff)
            vertical_excess = max(0.0, float(pos[g] - top_now))
            selection_energy = e_ads - coord_bonus * cn_local + vertical_penalty * vertical_excess
            mode = f"{label}:side_attach:anchor={anchor}:cn={cn_local}"

            candidates.append(
                Candidate(
                    symbol,
                    pos,
                    anchor,
                    dmin,
                    e_unrelaxed,
                    e_ads,
                    species_flux,
                    mode,
                    selection_energy,
                )
            )

    return candidates, len(anchors), n_tested


def choose_boltzmann_deposition_candidate(
    slab,
    g,
    species_list,
    calc,
    mu_map,
    flux_map,
    n0,
    n_candidates_per_species,
    surface_grid,
    surface_window,
    lateral_radius,
    height_min,
    height_max,
    min_dist_allowed,
    local_top_radius,
    anchor_species,
    anchor_species_label,
    temperature,
    fallback_to_all,
    droplet_prob,
    droplet_candidates,
    droplet_anchor_mode,
    side_radius_min,
    side_radius_max,
    side_height_jitter,
    cn_cutoff,
    coord_bonus,
    vertical_penalty,
    vacancy_positions=None,
):
    use_droplet = (len(slab) > n0) and (mpi_random() < droplet_prob)

    if use_droplet:
        candidates, n_anchors, n_tested = generate_droplet_side_candidates(
            slab=slab,
            g=g,
            species_list=species_list,
            calc=calc,
            mu_map=mu_map,
            flux_map=flux_map,
            n0=n0,
            n_candidates_per_species=droplet_candidates,
            min_dist_allowed=min_dist_allowed,
            side_radius_min=side_radius_min,
            side_radius_max=side_radius_max,
            side_height_jitter=side_height_jitter,
            droplet_anchor_mode=droplet_anchor_mode,
            cn_cutoff=cn_cutoff,
            coord_bonus=coord_bonus,
            vertical_penalty=vertical_penalty,
            vacancy_positions=vacancy_positions,
        )
        if candidates:
            return boltzmann_choose_candidate(candidates, temperature), n_anchors, n_tested
        print("[INFO] modo gota não gerou candidato válido; usando deposição uniforme por malha.")

    candidates, n_anchors, n_tested = generate_candidates_with_anchor_mode(
        slab, g, species_list, calc, mu_map, flux_map, n_candidates_per_species,
        surface_grid, surface_window, lateral_radius, height_min, height_max,
        min_dist_allowed, local_top_radius, anchor_species, anchor_species_label,
        vacancy_positions=vacancy_positions
    )

    if candidates:
        return boltzmann_choose_candidate(candidates, temperature), n_anchors, n_tested

    if fallback_to_all and anchor_species is not None:
        print("[INFO] nenhum candidato válido com anchor-species; fallback para toda a malha sem filtro de espécie.")
        candidates, n_anchors, n_tested = generate_candidates_with_anchor_mode(
            slab, g, species_list, calc, mu_map, flux_map, n_candidates_per_species,
            surface_grid, surface_window, lateral_radius, height_min, height_max,
            min_dist_allowed, local_top_radius, None, "all_fallback",
            vacancy_positions=vacancy_positions
        )
        if candidates:
            return boltzmann_choose_candidate(candidates, temperature), n_anchors, n_tested

    return None, n_anchors, n_tested

def compute_boltzmann_deposition_event(
    slab,
    g,
    species_list,
    calc,
    mu_map,
    flux_map,
    n0,
    deposition_rate,
    mobile_radius,
    fmax,
    relax_steps,
    optimizer,
    min_dist_allowed,
    height_min,
    height_max,
    n_candidates_per_species,
    surface_grid,
    surface_window,
    lateral_radius,
    local_top_radius,
    anchor_species,
    anchor_species_label,
    temperature,
    fallback_to_all,
    droplet_prob,
    droplet_candidates,
    droplet_anchor_mode,
    side_radius_min,
    side_radius_max,
    side_height_jitter,
    cn_cutoff,
    coord_bonus,
    vertical_penalty,
    no_relax_first_added=False,
    impurity_specs=None,
    vacancy_positions=None,
):
    chosen_candidate, n_anchors, n_tested = choose_boltzmann_deposition_candidate(
        slab, g, species_list, calc, mu_map, flux_map, n0, n_candidates_per_species,
        surface_grid, surface_window, lateral_radius, height_min, height_max,
        min_dist_allowed, local_top_radius, anchor_species, anchor_species_label,
        temperature, fallback_to_all,
        droplet_prob, droplet_candidates, droplet_anchor_mode,
        side_radius_min, side_radius_max, side_height_jitter,
        cn_cutoff=cn_cutoff, coord_bonus=coord_bonus, vertical_penalty=vertical_penalty,
        vacancy_positions=vacancy_positions
    )

    if chosen_candidate is None:
        return None

    e_before = slab.get_potential_energy()
    intended_species = chosen_candidate.species
    impurity_kind = choose_impurity(impurity_specs)
    actual_species = intended_species if impurity_kind is None else impurity_kind

    if actual_species == "vac":
        trial = slab.copy()
        attach_calculator(trial, calc)
        added = -1
        mobile = []
        e_after = e_before
        delta_e = 0.0
        e_ads_unrelaxed = 0.0
        event_type = "vacancy"
        print(f"[INFO] vacância sorteada por --impurity; espécie que seria depositada: {intended_species}.")
    else:
        trial, added = make_deposition_trial(slab, actual_species, chosen_candidate.position, calc)
        is_first_added_atom = (len(slab) == n0)

        try:
            e_unrelaxed_actual = chosen_candidate.e_unrelaxed if actual_species == intended_species else trial.get_potential_energy()
        except Exception as exc:
            reraise_mpi_atoms_mismatch(exc, "energia não relaxada da deposição")
            print(f"[WARN] energia não relaxada da deposição falhou para {actual_species}: {exc}")
            return None

        if no_relax_first_added and is_first_added_atom:
            mobile = []
            attach_calculator(trial, calc)
            e_after = trial.get_potential_energy()
            print("[INFO] primeira deposição: relaxação local pulada por --no-relax-first-added.")
        else:
            mobile = get_local_mobile_indices(trial, [added], mobile_radius)
            try:
                relax_local(trial, mobile, optimizer, fmax, relax_steps)
                attach_calculator(trial, calc)
                e_after = trial.get_potential_energy()
            except Exception as exc:
                reraise_mpi_atoms_mismatch(exc, "relaxação")
                print(f"[WARN] relaxação falhou: {exc}")
                return None

        mu = mu_map.get(actual_species, 0.0)
        delta_e = e_after - e_before - mu
        e_ads_unrelaxed = e_unrelaxed_actual - e_before - mu
        event_type = "deposition" if impurity_kind is None else "impurity"
        if impurity_kind is not None:
            print(f"[INFO] impureza sorteada por --impurity: {actual_species} no lugar de {intended_species}.")

    effective_rate = kinetic_event_rate(
        deposition_rate, delta_e, temperature,
        barrier_eV=getattr(compute_boltzmann_deposition_event, "kinetic_barrier_eV", 0.0),
        bep_alpha=getattr(compute_boltzmann_deposition_event, "kinetic_bep_alpha", 0.5),
    )

    return Event(
        event_type=event_type,
        deposited_species=actual_species,
        rate=effective_rate,
        trial=trial,
        delta_e=delta_e,
        e_before=e_before,
        e_after=e_after,
        e_ads_unrelaxed=e_ads_unrelaxed,
        position=chosen_candidate.position,
        added_index=added,
        anchor_index=chosen_candidate.anchor_index,
        mobile_count=len(mobile),
        min_dist=chosen_candidate.min_dist,
        n_anchors=n_anchors,
        n_tested=n_tested,
        chosen_probability=chosen_candidate.probability,
        anchor_mode=chosen_candidate.anchor_mode,
        species_flux=chosen_candidate.species_flux,
        intended_species=intended_species,
        impurity_kind=actual_species if impurity_kind is not None else "none",
        is_vacancy=(actual_species == "vac"),
    )


def choose_event(events):
    total_rate = sum(max(0.0, ev.rate) for ev in events)
    if total_rate <= 0.0:
        raise RuntimeError("A taxa total KMC é zero. Reduza --kinetic-barrier-eV, aumente a temperatura ou verifique deltaE.")
    r = mpi_random() * total_rate
    acc = 0.0
    for idx, ev in enumerate(events):
        acc += ev.rate
        if acc >= r:
            return idx, ev, -math.log(mpi_random()) / total_rate, total_rate
    return len(events) - 1, events[-1], -math.log(mpi_random()) / total_rate, total_rate


def init_log(path):
    if not MPI_MASTER:
        return None, None
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "step", "event_type", "deposited_species", "intended_species", "impurity_kind", "is_vacancy",
        "species_flux", "anchor_mode",
        "growth_axis", "surface_axes", "n_atoms", "n_initial", "n_added", "n_vacancies", "n_growth_events",
        "kmc_time", "dt", "temperature_K", "flux_ML_s",
        "n_sites", "deposition_total_rate_s_inv",
        "e_total_eV", "e_before_eV", "e_after_eV",
        "delta_e_relaxed_eV", "e_ads_unrelaxed_eV",
        "chosen_boltzmann_probability",
        "rate_s_inv", "total_rate_s_inv",
        "mobile_count", "added_index", "anchor_index",
        "n_surface_anchors", "n_tested_candidates",
        "min_distance_A", "x_event_A", "y_event_A", "z_event_A",
        "surface_area_A2",
        "gamma_initial_eV_A2", "gamma_effective_eV_A2", "gamma_effective_J_m2",
        "top_coord_A", "bottom_coord_A", "slab_thickness_A", "growth_height_A",
        "roughness_A", "coverage_estimate_ML",
        "front_mean_A", "front_median_A", "front_p90_A", "front_grid_rms_A", "front_grid_occupied_fraction",
        "growth_front_mean_A", "growth_front_median_A", "growth_front_p90_A",
        "first_template_layer_occupancy", "highest_90pct_complete_layer", "n_template_layers",
        "contiguous_90pct_complete_layers", "contiguous_complete_layer_fraction",
        "mean_template_layer_occupancy",
        "n_template_growth_sites", "n_occupied_template_growth_sites",
        "template_growth_occupancy_fraction", "frozen_vacancy_fraction_of_template",
        "incorporation_efficiency_per_event", "added_areal_density_atoms_A2",
        "growth_front_mean_per_event_A", "growth_front_mean_per_added_atom_A",
        "growth_front_mean_per_template_site_A",
        "template_growth_extent_A", "growth_front_mean_fraction_of_template",
        "coordination_mean", "coordination_surface_mean", "coordination_added_mean", "fraction_added_CN4",
    ])
    return f, w


def write_log(writer, step, slab, event, dt, kmc_time, args, total_rate, n0, h0_top, e_bulk_atom, gamma0, area, g, n_vacancies=0, template_atoms=None, h0_front_stats=None):
    e_total = slab.get_potential_energy()
    n_added = len(slab) - n0
    n_growth_events = n_added + int(n_vacancies)
    gamma_eff = gamma_effective_growth_top(e_total, n0, n_added, e_bulk_atom, area)

    cn = coordination_numbers(slab, args.cn_cutoff)
    surface_idx = get_surface_indices(slab, g, args.surface_window)
    cn_surface = float(np.mean(cn[surface_idx])) if surface_idx else 0.0

    added_idx = list(range(n0, len(slab)))
    cn_added = float(np.mean(cn[added_idx])) if added_idx else 0.0

    rough = surface_roughness(slab, g, args.surface_window)
    coverage = coverage_estimate(slab, g, n_added, args.site_area)
    pa = plane_axes_from_growth(g)
    surface_axes = AXIS_NAME[pa[0]] + AXIS_NAME[pa[1]]
    restrictions = normalize_fractional_restrictions(args.restrict_X, args.restrict_Y, args.restrict_Z)
    growth_species_for_metrics = set(args.species) if any(v is not None for v in restrictions) else None
    fs = surface_height_statistics(
        slab, g, args.front_grid[0], args.front_grid[1],
        include_species=growth_species_for_metrics, restrictions=restrictions,
        reference_cell=getattr(args, "restriction_reference_cell", None),
    )
    if h0_front_stats is None:
        h0_front_stats = {"mean": h0_top, "median": h0_top, "p90": h0_top}
    tm = template_growth_metrics(
        slab, template_atoms, g,
        occupancy_tol=args.template_occupancy_tol,
        layer_tol=args.template_layer_tol,
        origin_front=h0_top,
        growth_direction=args.template_growth_direction,
        front_min=args.template_front_min,
        allowed_site_species=args.template_site_species,
        growth_species=growth_species_for_metrics,
        restrictions=restrictions,
        reference_cell=getattr(args, "restriction_reference_cell", None),
    )
    first_occ = tm["first_layer_occupancy"]
    complete_layer = tm["highest_complete_layer"]
    n_layers = tm["n_layers"]

    growth_front_mean = fs["mean"] - h0_front_stats["mean"]
    n_template_sites = tm["n_growth_sites"]
    incorporation_efficiency = (n_added / n_growth_events) if n_growth_events > 0 else 0.0
    added_areal_density = (n_added / area) if area > 0 else 0.0
    growth_per_event = (growth_front_mean / n_growth_events) if n_growth_events > 0 else 0.0
    growth_per_added = (growth_front_mean / n_added) if n_added > 0 else 0.0
    growth_per_template_site = (growth_front_mean / n_template_sites) if n_template_sites > 0 else 0.0
    vacancy_fraction_template = (n_vacancies / n_template_sites) if n_template_sites > 0 else 0.0
    growth_fraction_template = (growth_front_mean / tm["growth_extent_A"]) if tm["growth_extent_A"] > 0 else 0.0

    fraction_added_cn4 = float(np.mean(cn[added_idx] == 4)) if added_idx else 0.0

    row = [
        step, event.event_type, event.deposited_species, event.intended_species, event.impurity_kind, int(event.is_vacancy),
        event.species_flux, event.anchor_mode,
        AXIS_NAME[g], surface_axes, len(slab), n0, n_added, n_vacancies, n_growth_events,
        kmc_time, dt, args.temperature, args.flux_ML_s,
        area / args.site_area if args.site_area > 0 else 0.0,
        args.flux_ML_s * (area / args.site_area) if args.site_area > 0 else 0.0,
        e_total, event.e_before, event.e_after,
        event.delta_e, event.e_ads_unrelaxed,
        event.chosen_probability,
        event.rate, total_rate,
        event.mobile_count, event.added_index, event.anchor_index,
        event.n_anchors, event.n_tested,
        event.min_dist,
        event.position[0], event.position[1], event.position[2],
        area,
        gamma0, gamma_eff, ev_a2_to_j_m2(gamma_eff),
        top_coord(slab, g), bottom_coord(slab, g), slab_thickness(slab, g),
        top_coord(slab, g) - h0_top,
        rough, coverage,
        fs["mean"], fs["median"], fs["p90"], fs["rms"], fs["occupied_fraction"],
        fs["mean"] - h0_front_stats["mean"], fs["median"] - h0_front_stats["median"], fs["p90"] - h0_front_stats["p90"],
        first_occ, complete_layer, n_layers,
        tm["contiguous_complete_layers"], tm["contiguous_complete_layer_fraction"],
        tm["mean_layer_occupancy"],
        n_template_sites, tm["n_occupied_growth_sites"],
        tm["growth_site_occupancy_fraction"], vacancy_fraction_template,
        incorporation_efficiency, added_areal_density,
        growth_per_event, growth_per_added, growth_per_template_site,
        tm["growth_extent_A"], growth_fraction_template,
        float(np.mean(cn)), cn_surface, cn_added, fraction_added_cn4,
    ]
    if writer is not None:
        writer.writerow(row)
    return {
        "e_total": e_total,
        "roughness": rough,
        "gamma_eff": gamma_eff,
        "front_mean": fs["mean"],
        "front_p90": fs["p90"],
        "layer_occupancy": first_occ,
        "template_occupancy_fraction": tm["growth_site_occupancy_fraction"],
        "growth_per_event": growth_per_event,
        "growth_fraction_template": growth_fraction_template,
    }


def run(args):
    # É obrigatório que todos os ranks usem exatamente a mesma sequência
    # pseudoaleatória, pois cada cálculo GPAW é coletivo sobre a mesma estrutura.
    validate_mpi_runtime(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    initialize_synced_rng(args.seed)

    slab = None
    traj = None
    log_file = None
    log_writer = None
    failure = None

    try:
        if MPI_MASTER:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        mpi_barrier()

        calc = make_calculator(args)
        species_list = args.species
        flux_map = parse_flux_species(args.flux_species, species_list)
        mu_map = parse_mu_species(args.mu_species)
        impurity_specs = parse_impurities(args.impurity)
        g = growth_axis_index(args)
        pa = plane_axes_from_growth(g)
        surface_axes = AXIS_NAME[pa[0]] + AXIS_NAME[pa[1]]

        print(f"Backend = {args.backend}; device = {args.backend_device}; dtype = {args.backend_dtype}")
        print(f"Semente sincronizada em todos os ranks = {args.seed}")
        print(f"Espécies competindo: {species_list}")
        print(f"Fluxo relativo por espécie: {flux_map}")
        print(f"Mu por espécie: {mu_map if mu_map else 'todos 0.0 eV'}")
        print(f"Impurezas/vacâncias: {impurity_summary(impurity_specs)}")
        print(f"Modo de vacância: {args.vacancy_mode}")
        print(f"growth_axis = {args.growth_axis}; surface_axes = {surface_axes}")
        print(f"replicate antes do slab = {tuple(args.replicate)}")
        print(f"miller = {tuple(args.miller)}")
        print(f"build_mode = {args.build_mode}; site_mode = {args.site_mode}")
        print(f"pbc = {tuple(args.pbc) if args.pbc is not None else 'auto'}")
        print(f"bulk_energy_repeat = {tuple(args.bulk_energy_repeat)}; skip_bulk_reference = {args.skip_bulk_reference}")
        print(f"cif_wrap_mode = {args.cif_wrap_mode} (aplicado somente aos arquivos .cif salvos)")

        slab = build_initial_structure_from_args(args, calc)

        # --restrict-X/Y/Z usam coordenadas fracionárias da célula inicial, antes
        # de qualquer aumento feito por --match-cell-to-template.
        restrictions = normalize_fractional_restrictions(
            args.restrict_X, args.restrict_Y, args.restrict_Z
        )
        restriction_reference_cell = np.array(slab.cell.array, dtype=float, copy=True)
        args.restriction_reference_cell = restriction_reference_cell
        restrictions_active = any(v is not None for v in restrictions)
        if restrictions_active:
            labels = [
                f"{AXIS_NAME[i].upper()}>={v:.6f}"
                for i, v in enumerate(restrictions) if v is not None
            ]
            print("Restrição cristalográfica de crescimento: " + " AND ".join(labels))
            print("  Aplicada a origem, template, ocupação e conectividade; substrato continua no cálculo de energia/forças.")
        else:
            print("Restrição cristalográfica de crescimento: desligada")

        template_atoms = None
        if args.site_mode == "template":
            template_cif = args.template_cif or args.cif
            template_atoms = build_template_sites_from_cif(
                template_cif,
                tuple(args.template_repeat),
                args.growth_axis,
                args.vacuum,
                args.pbc,
                args.template_center_growth,
                args.template_growth_period,
                args.template_auto_period,
                args.template_build_mode,
                args.template_growth_direction,
                args.template_extend_ncols,
                args.template_column_period,
                args.template_column_tol,
                args.template_site_species,
                restrictions,
            )
            if args.match_cell_to_template:
                enlarge_cell_to_include_template(slab, template_atoms, g, args.cell_growth_margin)
                attach_calculator(slab, calc)

        if args.initial_relax:
            relax_local(slab, list(range(len(slab))), args.optimizer, args.fmax, args.relax_steps)
            attach_calculator(slab, calc)

        # Reconstrói as estruturas a partir do rank 0 antes do primeiro SCF.
        # A calculadora anterior é descartada e recriada coletivamente.
        slab = mpi_broadcast_atoms(slab)
        attach_calculator(slab, calc)
        if template_atoms is not None:
            template_atoms = mpi_broadcast_atoms(template_atoms)

        n0 = len(slab)
        origin_lower = [x.lower() for x in args.growth_origin_species]
        if "all" in origin_lower:
            if restrictions_active:
                growth_origin_species = set(species_list)
                growth_origin_label = "+".join(species_list) + " (auto por --restrict)"
            else:
                growth_origin_species = None
                growth_origin_label = "all"
        else:
            growth_origin_species = set(args.growth_origin_species)
            growth_origin_label = "+".join(args.growth_origin_species)

        h0_top = top_coord_species(
            slab, g, growth_origin_species, restrictions, restriction_reference_cell
        )
        h0_bottom = bottom_coord_species(
            slab, g, growth_origin_species, restrictions, restriction_reference_cell
        )
        h0_front_stats = surface_height_statistics(
            slab, g, args.front_grid[0], args.front_grid[1], growth_origin_species,
            restrictions, restriction_reference_cell
        )
        area = surface_area(slab, g)
        e0 = slab.get_potential_energy()

        if args.e_bulk_atom is not None:
            e_bulk_atom = float(args.e_bulk_atom)
            print(f"Referência bulk: usando --e-bulk-atom = {e_bulk_atom:.8f} eV/átomo")
        elif args.skip_bulk_reference:
            e_bulk_atom = e0 / n0
            print("Referência bulk: pulada (--skip-bulk-reference).")
            print("  Usando E0/N0 como referência; gamma vira energia relativa ao estado inicial.")
        else:
            print("Calculando referência bulk...")
            e_bulk_atom = bulk_energy_per_atom(
                args.cif, calc, tuple(args.bulk_energy_repeat),
                args.relax_bulk_reference, args.optimizer, args.fmax, args.relax_steps,
                pbc_override=args.bulk_reference_pbc,
            )

        gamma0 = gamma_initial_symmetric_slab(e0, n0, e_bulk_atom, area)

        anchor_lower = [x.lower() for x in args.anchor_species]
        if "all" in anchor_lower:
            anchor_species = None
            anchor_species_label = "all"
        else:
            anchor_species = set(args.anchor_species)
            anchor_species_label = "+".join(args.anchor_species)

        print("Slab inicial:")
        print(f"  N0 = {n0}")
        print(f"  area({surface_axes}) = {area:.6f} Å²")
        print(f"  E0 = {e0:.6f} eV")
        print(f"  gamma0 = {gamma0:.6f} eV/Å² = {ev_a2_to_j_m2(gamma0):.6f} J/m²")
        print(f"  growth_origin_species = {growth_origin_label}; origin_front={h0_top:.6f} Å; origin_back={h0_bottom:.6f} Å")
        if args.site_mode == "template":
            print("  regra_deposicao = template: somente sítios do CIF-template")
            print(f"  template_cif = {args.template_cif or args.cif}")
            print(f"  template_repeat = {tuple(args.template_repeat)}; N_template = {len(template_atoms) if template_atoms is not None else 0}")
            print(f"  template_front = [{args.template_front_min:.3f}, {args.template_front_max:.3f}] Å")
            print(f"  template_connect_cutoff = {args.template_connect_cutoff:.3f} Å; min_neighbors = {args.template_min_neighbors}")
            print(f"  template_species_mode = {args.template_species_mode}; template_build_mode = {args.template_build_mode}")
            if restrictions_active:
                eligible_initial = growth_eligible_indices(
                    slab, species_filter=set(species_list), restrictions=restrictions,
                    reference_cell=restriction_reference_cell,
                )
                print(f"  restrict_fractional = {restrictions}; átomos_de_crescimento_iniciais={len(eligible_initial)}")
        else:
            print(f"  surface_grid = {args.surface_grid[0]} x {args.surface_grid[1]}")
            print(f"  local_top_radius = {args.local_top_radius:.3f} Å")
            print(f"  regra_deposicao = grid: {args.deposition_candidates} tentativa(s) por célula por espécie")
            print(f"  modo_gota: prob={args.droplet_prob:.3f}, candidatos={args.droplet_candidates}, anchor={args.droplet_anchor_mode}")
            print(f"  side_attach: raio=[{args.side_radius_min:.3f}, {args.side_radius_max:.3f}] Å, jitter_altura={args.side_height_jitter:.3f} Å")

        if MPI_MASTER:
            traj = Trajectory(args.out, "w")
            traj.write(atoms_for_safe_output(slab))
        log_file, log_writer = init_log(args.log)
        if log_file is not None:
            log_file.flush()
        write_xyz_safe(args.out.replace(".traj", "_initial.xyz"), slab)
        write_cif_compatible(args.out.replace(".traj", "_initial.cif"), slab, args.cif_wrap_mode)
        # O rank 0 pode levar mais tempo escrevendo. A barreira garante que todos
        # iniciem o primeiro passo no mesmo ponto de comunicação coletiva.
        mpi_barrier()

        vacancy_positions = []
        vacancy_records = []
        vacancy_csv = args.out.replace(".traj", "_vacancies.csv")
        write_vacancies_csv(vacancy_csv, vacancy_records)

        # Parâmetros da taxa Arrhenius/BEP usados pelas rotinas de evento.
        compute_template_deposition_event.kinetic_barrier_eV = args.kinetic_barrier_eV
        compute_template_deposition_event.kinetic_bep_alpha = args.kinetic_bep_alpha
        adsorption_energy_allowed.max_energy_eV = args.max_adsorption_energy_eV
        if args.max_adsorption_energy_eV is None:
            print("Corte absoluto de Eads: desligado")
        else:
            print(f"Corte absoluto de Eads: {args.max_adsorption_energy_eV:+.6f} eV")
        compute_boltzmann_deposition_event.kinetic_barrier_eV = args.kinetic_barrier_eV
        compute_boltzmann_deposition_event.kinetic_bep_alpha = args.kinetic_bep_alpha

        kmc_time = 0.0
        for step in range(1, args.steps + 1):
            # Evita acumular qualquer divergência após relaxações, escrita ou
            # decisões do passo anterior. O rank 0 é sempre a fonte da geometria.
            slab = mpi_broadcast_atoms(slab)
            attach_calculator(slab, calc)
            vacancy_positions = mpi_broadcast_object(vacancy_positions)
            vacancy_records = mpi_broadcast_object(vacancy_records)

            n_sites = area / args.site_area if args.site_area > 0 else 0.0
            deposition_total_rate = args.flux_ML_s * n_sites

            if args.site_mode == "template":
                ev = compute_template_deposition_event(
                    slab=slab,
                    template=template_atoms,
                    g=g,
                    species_list=species_list,
                    calc=calc,
                    mu_map=mu_map,
                    flux_map=flux_map,
                    n0=n0,
                    deposition_rate=deposition_total_rate,
                    mobile_radius=args.mobile_radius,
                    fmax=args.fmax,
                    relax_steps=args.relax_steps,
                    optimizer=args.optimizer,
                    min_dist_allowed=args.min_dist,
                    temperature=args.temperature,
                    front_min=args.template_front_min,
                    front_max=args.template_front_max,
                    occupancy_tol=args.template_occupancy_tol,
                    connect_cutoff=args.template_connect_cutoff,
                    min_neighbors=args.template_min_neighbors,
                    template_species_mode=args.template_species_mode,
                    allowed_site_species=args.template_site_species,
                    front_mode=args.template_front_mode,
                    growth_direction=args.template_growth_direction,
                    layer_tol=args.template_layer_tol,
                    origin_front=h0_top,
                    origin_back=h0_bottom,
                    template_debug=args.template_debug,
                    no_relax_first_added=args.no_relax_first_added,
                    impurity_specs=impurity_specs,
                    vacancy_positions=vacancy_positions,
                    vacancy_mode=args.vacancy_mode,
                    growth_species=set(species_list) if restrictions_active else None,
                    restrictions=restrictions,
                    reference_cell=restriction_reference_cell,
                )
            else:
                ev = compute_boltzmann_deposition_event(
                    slab=slab,
                    g=g,
                    species_list=species_list,
                    calc=calc,
                    mu_map=mu_map,
                    flux_map=flux_map,
                    n0=n0,
                    deposition_rate=deposition_total_rate,
                    mobile_radius=args.mobile_radius,
                    fmax=args.fmax,
                    relax_steps=args.relax_steps,
                    optimizer=args.optimizer,
                    min_dist_allowed=args.min_dist,
                    height_min=args.height_min,
                    height_max=args.height_max,
                    n_candidates_per_species=args.deposition_candidates,
                    surface_grid=tuple(args.surface_grid),
                    surface_window=args.surface_window,
                    lateral_radius=args.lateral_radius,
                    local_top_radius=args.local_top_radius,
                    anchor_species=anchor_species,
                    anchor_species_label=anchor_species_label,
                    temperature=args.temperature,
                    fallback_to_all=(args.anchor_fallback == "all"),
                    droplet_prob=args.droplet_prob,
                    droplet_candidates=args.droplet_candidates,
                    droplet_anchor_mode=args.droplet_anchor_mode,
                    side_radius_min=args.side_radius_min,
                    side_radius_max=args.side_radius_max,
                    side_height_jitter=args.side_height_jitter,
                    cn_cutoff=args.cn_cutoff,
                    coord_bonus=args.coord_bonus,
                    vertical_penalty=args.vertical_penalty,
                    no_relax_first_added=args.no_relax_first_added,
                    impurity_specs=impurity_specs,
                    vacancy_positions=vacancy_positions,
                )

            if ev is None:
                print("[STOP] nenhum candidato de deposição válido.")
                break

            _, chosen, dt, total_rate = choose_event([ev])
            slab = chosen.trial
            attach_calculator(slab, calc)
            kmc_time += dt

            if chosen.is_vacancy:
                vacancy_positions.append(np.asarray(chosen.position, dtype=float).copy())
                vacancy_records.append({
                    "step": step,
                    "intended_species": chosen.intended_species,
                    "position": np.asarray(chosen.position, dtype=float).copy(),
                    "anchor_mode": chosen.anchor_mode,
                    "anchor_index": chosen.anchor_index,
                })
                write_vacancies_csv(vacancy_csv, vacancy_records)
                if args.vacancy_mode == "frozen":
                    print(
                        f"[INFO] vacância frozen registrada no sítio template "
                        f"{chosen.anchor_index}; esse sítio não poderá ser depositado novamente."
                    )
            elif vacancy_positions and args.vacancy_mode == "fillable":
                # No modo fillable, uma deposição normal próxima ao sítio vazio
                # pode curar a vacância. No modo frozen, este bloco não é executado:
                # a posição permanece na lista vacancy_positions e continua
                # excluída da geração de candidatos até o fim da simulação.
                d = pbc_distances_positions_to_point(slab, vacancy_positions, chosen.position)
                if len(d) and float(np.min(d)) <= args.vacancy_fill_tol:
                    k = int(np.argmin(d))
                    vacancy_positions.pop(k)
                    vacancy_records.pop(k)
                    write_vacancies_csv(vacancy_csv, vacancy_records)
                    print(f"[INFO] vacância preenchida no step {step}.")

            if traj is not None:
                traj.write(atoms_for_safe_output(slab))

            # Todos os ranks executam write_log(), pois a função solicita uma
            # energia GPAW coletiva. Somente o rank 0 possui log_writer.
            metrics = write_log(
                log_writer, step, slab, chosen, dt, kmc_time, args,
                total_rate, n0, h0_top, e_bulk_atom, gamma0, area, g,
                n_vacancies=len(vacancy_positions), template_atoms=template_atoms,
                h0_front_stats=h0_front_stats
            )
            if log_file is not None:
                log_file.flush()
            write_xyz_safe(args.out.replace(".traj", "_checkpoint.xyz"), slab)
            write_cif_compatible(args.out.replace(".traj", "_checkpoint.cif"), slab, args.cif_wrap_mode)
            # Evita que ranks sem I/O entrem no broadcast do passo seguinte antes
            # de o master terminar os checkpoints seriais.
            mpi_barrier()

            impurity_note = "" if chosen.impurity_kind == "none" else f" impurity={chosen.impurity_kind} intended={chosen.intended_species}"
            print(
                f"step={step:05d} species={chosen.deposited_species:>3s}{impurity_note} "
                f"N={len(slab):05d} vac={len(vacancy_positions):04d} axis={AXIS_NAME[g]} surf={surface_axes} "
                f"Eads={chosen.e_ads_unrelaxed:+.4f} eV P={chosen.chosen_probability:.3e} "
                f"E={metrics['e_total']:.4f} eV hmax={top_coord(slab, g)-h0_top:.3f} A "
                f"hmean={metrics['front_mean']-h0_front_stats['mean']:.3f} A "
                f"rough={metrics['roughness']:.3f} A occ1={metrics['layer_occupancy']:.3f} tested={chosen.n_tested}"
            )

    except Exception as exc:
        failure = exc
        print("\n[ERRO] A simulação falhou, tentando salvar o estado atual.")
        if MPI_MASTER:
            traceback.print_exc()

    finally:
        if traj is not None:
            try:
                traj.close()
            except Exception:
                pass
        if log_file is not None:
            try:
                log_file.flush()
                log_file.close()
            except Exception:
                pass
        if slab is not None:
            final_xyz = args.out.replace(".traj", "_final.xyz")
            final_cif = args.out.replace(".traj", "_final.cif")
            try:
                write_xyz_safe(final_xyz, slab)
                write_cif_compatible(final_cif, slab, args.cif_wrap_mode)
                print("\nArquivos salvos:")
                print(f"  {args.out}")
                print(f"  {args.log}")
                print(f"  {args.out.replace('.traj', '_initial.xyz')}")
                print(f"  {args.out.replace('.traj', '_initial.cif')}")
                print(f"  {args.out.replace('.traj', '_checkpoint.xyz')}")
                print(f"  {args.out.replace('.traj', '_checkpoint.cif')}")
                print(f"  {args.out.replace('.traj', '_vacancies.csv')}")
                print(f"  {final_xyz}")
                print(f"  {final_cif}")
            except Exception as exc:
                if failure is None:
                    failure = exc
                print("[ERRO] Não consegui salvar os arquivos finais.")
                if MPI_MASTER:
                    traceback.print_exc()

    # Faz o PBS/MPI marcar o job como falho em vez de esconder a exceção.
    if failure is not None:
        raise failure


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--cif", required=True)
    p.add_argument("--species", nargs="+", required=True)
    p.add_argument("--flux-species", nargs="+", default=None)
    p.add_argument("--mu-species", nargs="+", default=None)
    p.add_argument(
        "--impurity",
        nargs="+",
        default=None,
        help="Pares espécie probabilidade para impurezas/vacâncias por evento. Ex.: --impurity vac 0.1 N 0.1 B 0.1. O comportamento de 'vac' é controlado por --vacancy-mode.",
    )
    p.add_argument(
        "--backend",
        choices=[
            "chgnet",
            "mace",
            "mace-small",
            "mace-medium",
            "sevennet",
            "sevennet-d3",
            "mattersim",
            "gpaw",
            "emt",
            "lj",
        ],
        default="chgnet",
    )
    p.add_argument("--backend-device", choices=["cpu", "cuda", "mps"], default="cpu", help="Dispositivo usado por backends ML. Use cpu se o CUDA/driver estiver problemático.")
    p.add_argument("--backend-dtype", choices=["float32", "float64"], default="float32", help="Precisão numérica para backends que aceitam dtype, como MACE.")
    p.add_argument("--mace-model", default="auto", help="Modelo MACE-MP: auto, small, medium, large etc. auto usa small para mace/mace-small e medium para mace-medium.")
    p.add_argument("--no-mace-dispersion", dest="mace_dispersion", action="store_false", help="Desliga correção de dispersão no MACE-MP.")
    p.set_defaults(mace_dispersion=True)
    p.add_argument("--sevennet-model", default="7net-0", help="Nome/checkpoint do modelo SevenNet. Ex.: 7net-0, 7net-omni.")
    p.add_argument("--mattersim-model", default=None, help="Caminho opcional para modelo MatterSim local, quando a versão instalada aceitar load_path.")

    p.add_argument("--gpaw-mode", choices=["pw", "fd", "lcao"], default="pw", help="Modo GPAW/DFT: pw, fd ou lcao.")
    p.add_argument("--gpaw-xc", default="PBE", help="Funcional de troca-correlação do GPAW, ex.: PBE.")
    p.add_argument("--gpaw-kpts", nargs=3, type=int, default=[1, 1, 1], help="Malha k-points do GPAW. Ex.: --gpaw-kpts 1 1 1.")
    p.add_argument("--gpaw-h", type=float, default=0.20, help="Espaçamento de grade em Å para --gpaw-mode fd.")
    p.add_argument("--gpaw-ecut", type=float, default=340.0, help="Energia de corte em eV para --gpaw-mode pw.")
    p.add_argument("--gpaw-basis", default=None, help="Base para --gpaw-mode lcao. Se omitida, usa o padrão do GPAW.")
    p.add_argument("--gpaw-txt", default="gpaw.txt", help="Arquivo de saída textual do GPAW. Use none/off/null para desligar.")
    p.add_argument("--gpaw-maxiter", type=int, default=333, help="Número máximo de iterações SCF do GPAW.")
    p.add_argument("--gpaw-spinpol", action="store_true", help="Liga cálculo spin-polarized no GPAW.")
    p.add_argument("--gpaw-symmetry", choices=["off", "on"], default="off", help="Simetria no GPAW. off é recomendado para deposição/crescimento.")

    p.add_argument("--miller", nargs=3, type=int, default=[0, 0, 1])
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--vacuum", type=float, default=18.0)
    p.add_argument(
        "--build-mode",
        choices=["surface", "direct"],
        default="surface",
        help="surface: gera slab com ASE surface(); direct: lê o CIF diretamente, sem recortar superfície.",
    )
    p.add_argument(
        "--pbc",
        nargs=3,
        type=int,
        default=None,
        help="Sobrescreve PBC como 0/1 0/1 0/1. Ex.: grafeno crescendo em x com vácuo em z: --pbc 0 1 0.",
    )
    p.add_argument("--direct-center-growth", action="store_true", help="No build-mode direct, centraliza/adiciona vácuo no eixo de crescimento. Normalmente deixe desligado para 2D/ribbons.")
    p.add_argument("--match-cell-to-template", action="store_true", help="Aumenta a célula no eixo de crescimento para conter o template repetido.")
    p.add_argument("--cell-growth-margin", type=float, default=5.0, help="Margem extra em Å ao aumentar a célula com --match-cell-to-template.")

    p.add_argument(
        "--growth-axis",
        choices=["x", "y", "z"],
        default="z",
        help="Direção de crescimento/vácuo. x usa grade no plano yz; y usa xz; z usa xy.",
    )

    p.add_argument(
        "--restrict-X", "--restrict-x", "--restrict_X", dest="restrict_X", type=float, default=None,
        help="Limite mínimo da coordenada fracionária/cristalográfica X para participar do crescimento/template. Se omitido, X fica sem restrição.",
    )
    p.add_argument(
        "--restrict-Y", "--restrict-y", "--restrict_Y", dest="restrict_Y", type=float, default=None,
        help="Limite mínimo da coordenada fracionária/cristalográfica Y para participar do crescimento/template. Se omitido, Y fica sem restrição.",
    )
    p.add_argument(
        "--restrict-Z", "--restrict-z", "--restrict_Z", dest="restrict_Z", type=float, default=None,
        help="Limite mínimo da coordenada fracionária/cristalográfica Z para participar do crescimento/template. Ex.: --restrict-Z 0.394.",
    )

    p.add_argument(
        "--replicate",
        nargs=3,
        type=int,
        default=[1, 1, 1],
        help="Replica o bulk lido do CIF ANTES de gerar o slab. Ex.: --replicate 4 4 1",
    )

    # Mantidos só para compatibilidade com comandos antigos.
    # --bulk-repeat não é mais usado para gerar o slab; use --replicate.
    # --slab-repeat não deve ser usado nesta versão.
    p.add_argument("--bulk-repeat", nargs=3, type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--slab-repeat", nargs=3, type=int, default=None, help=argparse.SUPPRESS)

    p.add_argument("--bulk-energy-repeat", nargs=3, type=int, default=[1, 1, 1])
    p.add_argument("--skip-bulk-reference", action="store_true", help="Pula o cálculo da referência bulk. Útil para seeds 2D grandes ou crescimento template; usa E0/N0 como referência relativa.")
    p.add_argument("--e-bulk-atom", type=float, default=None, help="Energia bulk por átomo em eV. Se fornecida, evita recalcular a referência bulk.")
    p.add_argument("--bulk-reference-pbc", nargs=3, type=int, default=None, help="PBC usado só no cálculo da referência bulk. Ex.: grafeno/ribbon: --bulk-reference-pbc 0 1 0.")

    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--flux-ML-s", dest="flux_ML_s", type=float, default=1.0e-3)
    p.add_argument("--kinetic-barrier-eV", type=float, default=0.0, help="Barreira basal de incorporação usada na taxa Arrhenius/BEP.")
    p.add_argument("--kinetic-bep-alpha", type=float, default=0.5, help="Fator BEP aplicado à parte endotérmica de deltaE na taxa cinética.")
    p.add_argument(
        "--max-adsorption-energy-eV",
        type=float,
        default=None,
        help=(
            "Energia máxima de adsorção não relaxada permitida, em eV. "
            "Candidatos com Eads acima desse valor são rejeitados antes da "
            "seleção. Se omitido, nenhum corte absoluto é aplicado."
        ),
    )
    p.add_argument(
        "--vacancy-mode",
        choices=["fillable", "frozen"],
        default="fillable",
        help=(
            "Tratamento dos sítios vazios criados por --impurity vac: "
            "fillable permite preenchimento posterior; frozen remove permanentemente "
            "o sítio do conjunto de candidatos e mantém a vacância residual até o fim."
        ),
    )
    p.add_argument(
        "--vacancy-fill-tol",
        type=float,
        default=0.75,
        help="Distância em Å para preencher vacância no modo fillable; ignorada no modo frozen.",
    )
    p.add_argument("--front-grid", nargs=2, type=int, default=[8, 8], help="Malha usada para medir altura média/mediana/p90 da frente.")
    p.add_argument("--deposition-candidates", type=int, default=1, help="Nesta versão: tentativas por célula da surface-grid, por espécie. Ex.: 1 gera uma tentativa por região da malha.")

    p.add_argument("--site-mode", choices=["random", "template"], default="random", help="random: modo antigo por malha/gota; template: só deposita em sítios cristalográficos lidos de um CIF-template.")
    p.add_argument("--template-cif", default=None, help="CIF com os sítios cristalográficos permitidos. Se omitido, usa --cif.")
    p.add_argument("--template-repeat", nargs=3, type=int, default=[1, 1, 1], help="Repetição do CIF-template para criar sítios futuros.")
    p.add_argument("--template-front-min", type=float, default=0.05, help="Distância mínima à frente da frente de crescimento para aceitar um sítio template, em Å.")
    p.add_argument("--template-front-max", type=float, default=3.0, help="Distância máxima à frente da frente de crescimento para aceitar um sítio template, em Å.")
    p.add_argument("--template-occupancy-tol", type=float, default=0.6, help="Se houver átomo a menos que isso do sítio, ele é considerado ocupado, em Å.")
    p.add_argument("--template-connect-cutoff", type=float, default=2.0, help="Cutoff para exigir conexão do sítio template com o cristal atual, em Å.")
    p.add_argument("--template-min-neighbors", type=int, default=1, help="Número mínimo de vizinhos no cristal atual para permitir deposição no sítio template.")
    p.add_argument("--template-species-mode", choices=["command", "template"], default="command", help="command: espécies vêm de --species; template: deposita a espécie do sítio do template, se ela estiver em --species.")
    p.add_argument("--template-site-species", nargs="+", default=["all"], help="Filtra quais espécies do template podem virar sítios. Use all para todas.")
    p.add_argument("--growth-origin-species", nargs="+", default=["all"], help="Espécies usadas para definir a frente inicial de crescimento no modo template. Para MoS2 sobre grafeno crescendo lateralmente, use: --growth-origin-species Mo S")
    p.add_argument("--template-front-mode", choices=["global", "shell", "all_connected"], default="shell", help="global: usa a frente global atual; shell: preenche a camada template mais próxima antes de avançar; all_connected: usa todo sítio vazio conectado.")
    p.add_argument("--template-growth-direction", choices=["plus", "minus", "both"], default="plus", help="Direção do crescimento no eixo escolhido. Para crescer em +x use plus.")
    p.add_argument("--template-layer-tol", type=float, default=0.25, help="Tolerância em Å para agrupar uma camada/shell de sítios template.")
    p.add_argument("--template-debug", action="store_true", help="Imprime contadores de rejeição dos sítios template a cada step.")
    p.add_argument("--template-center-growth", action="store_true", help="Centraliza/adiciona vácuo ao template no eixo de crescimento. Normalmente deixe desligado.")
    p.add_argument("--template-growth-period", type=float, default=None, help="Período cartesiano em Å para repetir o template no eixo de crescimento, em vez da célula do CIF.")
    p.add_argument("--template-auto-period", action="store_true", help="Estima automaticamente o período no eixo de crescimento como largura + espaçamento mediano dos sítios.")
    p.add_argument("--template-build-mode", choices=["repeat", "extend-columns"], default="repeat", help="repeat: repete o CIF-template; extend-columns: extrapola as colunas da borda, melhor para grafeno/ribbons.")
    p.add_argument("--template-extend-ncols", type=int, default=30, help="Número de colunas novas criadas no modo --template-build-mode extend-columns.")
    p.add_argument("--template-column-period", type=int, default=2, help="Período de alternância das colunas no modo extend-columns. Para grafeno retangular geralmente 2.")
    p.add_argument("--template-column-tol", type=float, default=0.2, help="Tolerância em Å para agrupar átomos em uma coluna no modo extend-columns.")

    # Modo opcional para crescimento tipo gota/ilha.
    # A deposição uniforme da malha continua disponível; este modo representa
    # uma etapa efetiva de difusão superficial e anexação lateral.
    p.add_argument("--droplet-prob", type=float, default=0.0, help="Probabilidade de usar anexação lateral tipo gota em vez da deposição uniforme por malha. 0 desliga; 0.2-0.5 costuma gerar ilhas/gotas.")
    p.add_argument("--droplet-candidates", type=int, default=24, help="Número de candidatos laterais testados por espécie quando o modo gota é ativado.")
    p.add_argument("--droplet-anchor-mode", choices=["added", "surface"], default="added", help="added: ancora preferencialmente em átomos depositados; surface: ancora em qualquer átomo de superfície.")
    p.add_argument("--side-radius-min", type=float, default=1.7, help="Raio lateral mínimo para anexação tipo gota, em Å.")
    p.add_argument("--side-radius-max", type=float, default=2.8, help="Raio lateral máximo para anexação tipo gota, em Å.")
    p.add_argument("--side-height-jitter", type=float, default=0.45, help="Variação vertical permitida em torno da altura da âncora lateral, em Å.")
    p.add_argument("--coord-bonus", type=float, default=0.08, help="Bônus energético efetivo por vizinho local na seleção de candidatos tipo gota, em eV/vizinho.")
    p.add_argument("--vertical-penalty", type=float, default=0.20, help="Penalização efetiva para candidato acima do topo atual, em eV/Å.")

    p.add_argument(
        "--surface-grid",
        nargs=2,
        type=int,
        default=[8, 8],
        help="Grade no plano perpendicular ao growth-axis. Se growth-axis=x, é grade yz.",
    )
    p.add_argument("--xy-grid", nargs=2, type=int, default=None, help=argparse.SUPPRESS)

    p.add_argument("--lateral-radius", type=float, default=1.6)
    p.add_argument("--local-top-radius", type=float, default=2.0)
    p.add_argument("--temperature", type=float, default=500.0)
    p.add_argument("--height-min", type=float, default=1.2)
    p.add_argument("--height-max", type=float, default=1.6)
    p.add_argument("--min-dist", type=float, default=1.4)
    p.add_argument("--mobile-radius", type=float, default=2.5)
    p.add_argument("--surface-window", type=float, default=3.0)
    p.add_argument("--cn-cutoff", type=float, default=3.0)
    p.add_argument("--site-area", type=float, default=8.0)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--relax-steps", type=int, default=50)
    p.add_argument("--optimizer", choices=["fire", "bfgs"], default="fire")
    p.add_argument("--initial-relax", action="store_true")
    p.add_argument(
        "--no-relax-first-added",
        action="store_true",
        help="Não faz a relaxação local imediatamente após adicionar o primeiro átomo. A partir do segundo átomo, a relaxação volta ao comportamento normal.",
    )
    p.add_argument("--relax-bulk-reference", action="store_true")
    p.add_argument("--anchor-species", nargs="+", default=["all"])
    p.add_argument("--anchor-fallback", choices=["all", "stop"], default="all")
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help=(
            "Semente pseudoaleatória. Em MPI, o mesmo valor é usado em todos "
            "os ranks para que todos construam exatamente os mesmos candidatos."
        ),
    )
    p.add_argument("--out", default="growth.traj")
    p.add_argument("--log", default="growth_log.csv")
    p.add_argument(
        "--cif-wrap-mode",
        choices=["all", "pbc", "none"],
        default="all",
        help="Como normalizar coordenadas fracionárias ao salvar CIF. all: força x/y/z para [0,1), melhor para VESTA/OVITO; pbc: só eixos periódicos; none: não normaliza.",
    )

    args = p.parse_args()
    if args.xy_grid is not None:
        args.surface_grid = args.xy_grid

    if args.bulk_repeat is not None:
        print("[AVISO] --bulk-repeat foi ignorado para gerar slab. Use --replicate.")

    if args.slab_repeat is not None:
        print("[AVISO] --slab-repeat foi ignorado nesta versão. A repetição agora é antes do slab com --replicate.")

    if any(v <= 0 for v in args.replicate):
        raise ValueError("--replicate precisa ter três inteiros positivos")

    if any(v <= 0 for v in args.bulk_energy_repeat):
        raise ValueError("--bulk-energy-repeat precisa ter três inteiros positivos")

    if any(v <= 0 for v in args.template_repeat):
        raise ValueError("--template-repeat precisa ter três inteiros positivos")

    if args.pbc is not None:
        normalize_pbc_override(args.pbc)

    if args.bulk_reference_pbc is not None:
        normalize_pbc_override(args.bulk_reference_pbc)

    if any(v <= 0 for v in args.gpaw_kpts):
        raise ValueError("--gpaw-kpts precisa ter três inteiros positivos")

    if args.gpaw_h <= 0:
        raise ValueError("--gpaw-h precisa ser positivo")

    if args.gpaw_ecut <= 0:
        raise ValueError("--gpaw-ecut precisa ser positivo")

    if args.gpaw_maxiter <= 0:
        raise ValueError("--gpaw-maxiter precisa ser positivo")

    # Valida já no parse_args para erro rápido antes de iniciar cálculos caros.
    parse_impurities(args.impurity)

    if args.template_front_min < 0 or args.template_front_max <= 0:
        raise ValueError("--template-front-min precisa ser >= 0 e --template-front-max > 0")

    if args.template_front_min > args.template_front_max:
        raise ValueError("--template-front-min não pode ser maior que --template-front-max")

    if args.template_occupancy_tol <= 0 or args.template_connect_cutoff <= 0:
        raise ValueError("--template-occupancy-tol e --template-connect-cutoff precisam ser positivos")

    if args.template_min_neighbors < 0:
        raise ValueError("--template-min-neighbors precisa ser >= 0")

    if args.template_extend_ncols < 0:
        raise ValueError("--template-extend-ncols precisa ser >= 0")

    if args.template_column_period <= 0 or args.template_column_tol <= 0:
        raise ValueError("--template-column-period e --template-column-tol precisam ser positivos")

    if not (0.0 <= args.droplet_prob <= 1.0):
        raise ValueError("--droplet-prob precisa estar entre 0 e 1")

    if args.side_radius_min <= 0 or args.side_radius_max <= 0:
        raise ValueError("--side-radius-min e --side-radius-max precisam ser positivos")

    if args.side_radius_min > args.side_radius_max:
        raise ValueError("--side-radius-min não pode ser maior que --side-radius-max")

    normalize_fractional_restrictions(args.restrict_X, args.restrict_Y, args.restrict_Z)

    return args


if __name__ == "__main__":
    run(parse_args())
