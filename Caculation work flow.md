# Harmonic Approximation
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import warnings
import numpy as np

from ase.io import read (ase에서 구조 파일을 읽을 때 사용 > 초기 구조)
from ase import Atoms as ASEAtoms 
from ase.optimize import BFGS ( BFGS 구조 최적화 알고리즘, force를 줄여 구조 안정화 진행)

try:
    from sevenn.calculator import SevenNetCalculator ( Sevennet 공식 사이트에 있는 SevenNet 계산기 불러옴)
except Exception:
    from sevenn.sevennet_calculator import SevenNetCalculator ( SevenNet 버전에 맞게 한번 더 )

from phonopy import Phonopy ( Phonopy 계산을 위해서는 꼭 불러와야함)
from phonopy.structure.atoms import PhonopyAtoms ( ASE에서 읽은 원자 구조 파일을 phonopy 방식에 맞는 구조형식 객체를 불러오는 방법)
from phonopy.interface.calculator import get_default_displacement_distance(구조 변위를 움직 일 때 얼만큼 움직일 것인가를 추천값으로 가져오는 함수)

from phonopy.harmonic.force_constants import set_translational_invariance, set_permutation_symmetry ( force constants 후처리하는 함수 불러오기)

try:
    from phonopy.harmonic.force_constants import set_rotational_invariance ( force constants가 구조 변위를 진행할 때 이론 상가능하게 진행, 버전 X False)
    HAS_ROT = True
except Exception:
    HAS_ROT = False

try:
    from phonopy.file_IO import write_FORCE_CONSTANTS ( 계산된 force constant > FORCE_CONSTANTS 파일로 저장)
    HAS_WRITE_FC = True
except Exception:
    HAS_WRITE_FC = False
--------------------------------------------------라이브러리 불러오는 구간---------------------------------------------------------------------------------

MODEL  = "7net-omni"
MODAL  = "mp_r2scan"
TAG    = "re_imaginary_r2scan"
DEVICE = "cuda:0"

DISP_DISTANCE = 0.01 (구조 변위는 기본 값으로 진행)
MESH = (24, 24, 24)

SIGMA_THz = None ( total DOS 계산 시 Gaussian smearing 폭을 줄일지 말지)
FREQ_PITCH_THz = 0.001 ( DOS를 그릴 시 주파수 축 간격 설정 > 촘촘하게 진행)
USE_TETRA = True ( DOS 계산 시 사용하는 방법, 기본값)

DO_RELAX_UNITCELL = True ( Phonon 계산 전 ASE의 BFGS 공식 기능 > 원자 위치 relax) -- AI
RELAX_FMAX_UNITCELL = 1e-3 ( relaxation 수렴 기준 force 크기)
RELAX_MAXSTEP_UNITCELL = 0.04 ( 최적화 중 원자가 움직일 수 있는 최대 거리 설정, 기본 값 : 0.2Å)

DO_RELAX_PERFECT_SUPERCELL = True ( 변위를 주기 전 완전한 supercell도 force 0을 맞추기 위해서) -- AI
RELAX_FMAX_SUPERCELL = 1e-3 (ASE BFGS 공식 > fmax는 수렴 기준)
RELAX_MAXSTEP_SUPERCELL = 0.04 (maxstep 기본값 0.2 > 0.04로 더 강하게)

USE_FZ_RESIDUAL_FORCE_SUBTRACTION = True (구조 변위 후 구한 supercell force에서 기준 supercell 구조의 남아있는 force 값을 빼, 변위 구조의 force 만 사용) -- AI 

SAVE_WITH_IMAG_IF_NEGATIVE = True ( DOS 파일 생성 시 imaginary도 보이게 할 것인지)
IMAG_FREQ_MIN_THz = 5.0 ( imaginary DOS 그릴 시 최소 주파수 범위 설정)

OUTDIR = f"fcm_out_{TAG}"
os.makedirs(OUTDIR, exist_ok=True) ( 폴더 생성, 있어도 오류x)

SYMPREC_CANDIDATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2] ( symmetry 설정, 여러 값을 시도하여 최적의 값 선택) -- AI
DEFAULT_IS_SYMMETRY = True ( symmery 사용 > 구조 대칭을 사용)
DEFAULT_IS_MESH_SYMMETRY = True ( mesh 계산할 때도 symmetry reduction)
PG_WARNING_KEY = "Point group symmetries of supercell and primitivecell are different"

-------------------------------------------------실제 계산 조건을 정하는 단계------------------------------------------------------------------------------
(초기 cell opimization이 끝난 구조)

IN_VASP_MAP = {
    "alpha": "./r2scan/structure_file/mp_r2scan_alpha_cellopt_7net.vasp",
    "beta":  "./r2scan/structure_file/mp_r2scan_beta_cellopt_7net.vasp",
    "gamma": "./r2scan/structure_file/mp_r2scan_gamma_cellopt_7net.vasp",
    "delta": "./r2scan/structure_file/mp_r2scan_delta_cellopt_7net.vasp",
}

(supercell 설정)

SUPERCELL_MAP = {
    "alpha": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    "beta":  [[4, 0, 0], [0, 4, 0], [0, 0, 3]],
    "gamma": [[4, 0, 0], [0, 4, 0], [0, 0, 3]],
    "delta": [[6, 0, 0], [0, 4, 0], [0, 0, 2]],
}


PHASES = ["alpha", "beta", "gamma", "delta"]
-----------------------------------------------------직접 설정------------------------------------------------------------------------------------------

def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ase_to_phonopy_atoms(atoms): ( 구조 파일은 ASE가 읽고 > phonon 계산은 phonopy가 진행, ASE 구조를 phonopy 구조로 바꿈)
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(wrap=True),
    )


def phonopy_atoms_to_ase(ph_atoms): ( phonopy가 구조 변위 진행 후 > 실제 force 계산은 ASE calculator인 SevenNet이 진행, 반대)
    return ASEAtoms(
        symbols=ph_atoms.symbols,
        scaled_positions=ph_atoms.scaled_positions,
        cell=ph_atoms.cell,
        pbc=True,
    )

-------------------------------------------------계산을 위한 보조 설정------------------------------------------------------------------------------------
def extract_total_dos(phonon): ( 버전이 달라 get_total_dos, get_total_dos_dict(), phonon.total_dos 중 total DOS를 꺼내오는 역할) -- AI
    if hasattr(phonon, "get_total_dos"):
        f, d = phonon.get_total_dos()
        return np.array(f, float), np.array(d, float)
    if hasattr(phonon, "get_total_dos_dict"):
        dd = phonon.get_total_dos_dict()
        if "frequency_points" in dd and "total_dos" in dd:
            return np.array(dd["frequency_points"], float), np.array(dd["total_dos"], float)
        if "frequency" in dd and "dos" in dd:
            return np.array(dd["frequency"], float), np.array(dd["dos"], float)
    td = getattr(phonon, "total_dos", None)
    if td is not None:
        if hasattr(td, "frequency_points") and hasattr(td, "dos"):
            return np.array(td.frequency_points, float), np.array(td.dos, float)
        if hasattr(td, "get_dos"):
            f, d = td.get_dos()
            return np.array(f, float), np.array(d, float)
    raise RuntimeError("Cannot extract total DOS from this phonopy version.")

--------------------------------------------------DOS 결과를 꺼내는 함수----------------------------------------------------------------------------------
def relax_atoms_positions(atoms, calc, fmax, maxstep): (phonon 계산 전 한번 더 relax를 하는 조건) --AI
    atoms = atoms.copy() (구조 복사)
    atoms.calc = calc (calculator로 계산)
    f0 = float(np.abs(atoms.get_forces()).max()) ( 현재 force 최댓값 f0 계산)
    if f0 <= fmax: ( 이미 충분하면 relax 안하고 반환)
        return atoms, f0, f0
    opt = BFGS(atoms, logfile=None, maxstep=maxstep) ( 충분하지 않으면 BFGS 실행)
    opt.run(fmax=fmax)
    f1 = float(np.abs(atoms.get_forces()).max()) ( 최종 force 최댓값 f1 계산 후 반환)
    return atoms, f0, f1
--------------------------------------------------구조  relaxation 진행----------------------------------------------------------------------------------

def _make_phonopy_with_warning_capture(ph_unit, supercell_matrix, symprec, is_symmetry): ( phonopy 구조 생성 시 대칭을 잡을 때 다르다면 > warning)
    with warnings.catch_warnings(record=True) as wlist: (phonopy 생성에 대한 부분은 공식 사이트를 기반하지만 대칭을 자동으로 안정적인 부분을 선택하는 것) -- AI
        warnings.simplefilter("always")
        phonon = Phonopy(
            ph_unit,
            supercell_matrix=supercell_matrix,
            symprec=symprec,
            is_symmetry=is_symmetry,
        )
    got_pg_warning = any((str(w.message).find(PG_WARNING_KEY) >= 0) for w in wlist)
    return phonon, got_pg_warning

--------------------------------------------------phonopy 생성 시 symmetry 검사--------------------------------------------------------------------------

def build_phonopy_robust(ph_unit, supercell_matrix): ( warning이 안날 시 phonopy 설정 자동으로 찾는 함수 ) -- AI
    for sp in SYMPREC_CANDIDATES:
        phonon, warned = _make_phonopy_with_warning_capture(
            ph_unit, supercell_matrix, symprec=sp, is_symmetry=DEFAULT_IS_SYMMETRY
        )
        if not warned:
            return phonon, sp, DEFAULT_IS_SYMMETRY, False
    phonon, _ = _make_phonopy_with_warning_capture(
        ph_unit, supercell_matrix, symprec=SYMPREC_CANDIDATES[-1], is_symmetry=False
    )
    return phonon, SYMPREC_CANDIDATES[-1], False, True

--------------------------------------------------phonopy를 최대한 안정적으로 만들기 위한 자동 대칭 설정---------------------------------------------------

def get_mesh_minfreq_and_negcount(phonon): ( mesh 계산 후 최소 주파수, 음수 주파수 개수 판별, mesh 계산 자체는 공식을 기반) -- AI
    freqs = None
    mesh = getattr(phonon, "mesh", None)
    if mesh is not None:
        if hasattr(mesh, "frequencies"):
            freqs = np.array(mesh.frequencies, float)
        elif hasattr(mesh, "get_frequencies"):
            freqs = np.array(mesh.get_frequencies(), float)
    if freqs is None and hasattr(phonon, "get_mesh_dict"):
        try:
            md = phonon.get_mesh_dict()
            if isinstance(md, dict) and "frequencies" in md:
                freqs = np.array(md["frequencies"], float)
        except Exception:
            pass
    if freqs is None:
        return None, None
    freqs = freqs.ravel()
    return float(np.min(freqs)), int(np.sum(freqs < 0.0))

----------------------------------------------------mesh에서 음수 모드 상태를 읽는 함수--------------------------------------------------------------------

def run_total_dos_safe(phonon, freq_min): (run_total_dos 명렁어는 공식 기반, dos 계산 명령어)
    try:
        phonon.run_total_dos(
            sigma=SIGMA_THz,
            freq_min=freq_min,
            freq_max=None,
            freq_pitch=FREQ_PITCH_THz,
            tetrahedron_method=USE_TETRA, ( 기본 값이였던 tetrahedron이 있다면 진행)
        )
    except TypeError:
        phonon.run_total_dos(
            sigma=SIGMA_THz,
            freq_min=freq_min,
            freq_max=None,
            freq_pitch=FREQ_PITCH_THz, ( tetrahedron 버전이 아니라면 PITCH로 진행)
        )

-----------------------------------------------------DOS 계산 진행----------------------------------------------------------------------------------------
def run_fcm_one(phase, in_vasp, supercell_matrix, calc):
    if not os.path.exists(in_vasp):
        raise FileNotFoundError(in_vasp)

    prefix = os.path.join(OUTDIR, f"{TAG}_{phase}")

    out_fc_npy   = f"{prefix}_force_constants.npy"
    out_fc_txt   = f"{prefix}_FORCE_CONSTANTS"
    out_dos      = f"{prefix}_phonon_dos.dat"
    out_dos_imag = f"{prefix}_phonon_dos_with_imag.dat"
    out_tp       = f"{prefix}_thermal_properties.yaml"
    out_params   = f"{prefix}_phonopy_params.yaml"
    out_time     = f"{prefix}_timing_fcm.log"

    t_all0 = time.perf_counter()

------------------------------------------------------입력 구조 파일 확인 > 출력 파일 이름 설정------------------------------------------------------------

    atoms = read(in_vasp, format="vasp") (구조 자체를 읽는 ASE 공식 기반)
    if DO_RELAX_UNITCELL: -- (구조 준비가 완벽하게 된 상태에서 계산을 진행하라고 나와있기 때문에 한번 더 relax) -- AI
        atoms, _, _ = relax_atoms_positions(atoms, calc, RELAX_FMAX_UNITCELL, RELAX_MAXSTEP_UNITCELL) 

-----------------------------------------------------구조 > relaxation 계산 진행-------------------------------------------------------------------------

    ph_unit = ase_to_phonopy_atoms(atoms)

    phonon, used_symprec, used_is_sym, used_fallback = build_phonopy_robust(ph_unit, supercell_matrix)

------------------------------------------------------ASE 구조 > phonopy 방식으로 변환-------------------------------------------------------------------
    try:
        n_prim = int(phonon.primitive.get_number_of_atoms()) (phonopy가 내부적으로 쓰는 primitive cell을 불러옴)
    except Exception:
        n_prim = len(phonon.primitive) (phonon 계산 시 개수는 primitive cell 원자수를 기준, 3N 정규화)
    threeN = 3 * n_prim ( 원자 1개당 자유도(x,y,z)이기 때문에 3N 정규화 시도)

    disp = DISP_DISTANCE if DISP_DISTANCE is not None else get_default_displacement_distance(phonon.primitive)

-------------------------------------------------------primitive 원자수 확인, 3N 정규화------------------------------------------------------------------

    try:
        phonon.generate_displacements(distance=disp, is_plusminus="auto") (FCM 방법, 원자를 조금씩 움직여 force를 계산 > displacement)
    except TypeError:
        phonon.generate_displacements(distance=disp) (phonopy 공식 기반 명령어, 기본 값 0.01)

    try:
        supercells = phonon.get_supercells_with_displacements()(기본 값 0.01를 기준으로 구조 변위 후보들을 자동으로 탐색해주는 명령어) -- AI 
    except Exception:
        supercells = phonon.supercells_with_displacements
    if not supercells:
        raise RuntimeError(f"[{phase}] No displaced supercells generated.")

---------------------------------------------------------구조 변위 supercell 후보들을 확보 후 생성---------------------------------------------------------

    F0 = None
    if USE_FZ_RESIDUAL_FORCE_SUBTRACTION:
        ase_sc0 = phonopy_atoms_to_ase(phonon.supercell)
        if DO_RELAX_PERFECT_SUPERCELL:
            ase_sc0, _, _ = relax_atoms_positions(ase_sc0, calc, RELAX_FMAX_SUPERCELL, RELAX_MAXSTEP_SUPERCELL)
        else:
            ase_sc0.calc = calc
        _ = ase_sc0.get_potential_energy()
        F0 = ase_sc0.get_forces()

----------------------------------------------------------perfect supercell force 계산-------------------------------------------------------------------

    forces = []
    for sc in supercells:
        ase_sc = phonopy_atoms_to_ase(sc) (ASE 구조로 변환)
        ase_sc.calc = calc (calculator 진행)
        f = ase_sc.get_forces() (force 계산)
        if F0 is not None: 
            f = f - F0 ( perfect supercell force를 계산한 값을 구조 변위 force와 빼 보정하는 원리) -- AI 
        forces.append(f) (변위 x supercell에 남아있는 force가 데이터 값에 영향을 주면 안되기 때문에 따로 F0로 구한 후 차이 값을 사용하여 순수한 변위에 대한 force를 계산)

----------------------------------------------------------구조 변위 진행한 것들 force 계산-----------------------------------------------------------------

    phonon.forces = forces (phonopy 공식 기반 명령어)
    phonon.produce_force_constants() (계산한 force를 phonopy에 삽입)

    fc = phonon.force_constants (force constants 생성)

    if used_is_sym:
        try: (synnetry 보정 수행, 공식 기반 명령어들을 조합하여 사용) -- AI
            phonon.symmetrize_force_constants()
            fc = phonon.force_constants
        except Exception:
            pass
        try:
            set_permutation_symmetry(fc)
        except Exception:
            pass

    try:
        set_translational_invariance(fc)
    except Exception:
        pass

    if HAS_ROT:
        try:
            set_rotational_invariance(fc)
        except Exception:
            pass

    phonon.force_constants = fc

---------------------------------------------------------force constants 완성----------------------------------------------------------------------------


    np.save(out_fc_npy, phonon.force_constants) (force constants를 파일로 저장, FORCE_CONSTANTS도 저장)
    if HAS_WRITE_FC:
        try:
            write_FORCE_CONSTANTS(phonon.force_constants, filename=out_fc_txt)
        except Exception:
            pass

    try:
        phonon.save(filename=out_params)
    except Exception:
        pass

--------------------------------------------------------force constants, phonopy 설정 저장----------------------------------------------------------------

    mesh_sym = bool(DEFAULT_IS_MESH_SYMMETRY and used_is_sym)
    phonon.run_mesh(MESH, with_eigenvectors=False, is_mesh_symmetry=mesh_sym) (phonopy 공식 기반 명령어, mesh phonon 계산 실행)

    minf, negc = get_mesh_minfreq_and_negcount(phonon) (최소 주파수. 음수 모드 개수 확인 > imaginary mode 진단) -- AI

--------------------------------------------------------mesh 계산, instability 진단-----------------------------------------------------------------------

    run_total_dos_safe(phonon, freq_min=0.0) (DOS 계산)
    f_pos, d_pos = extract_total_dos(phonon) (주파수, DOS 데이터 추출)
    np.savetxt(out_dos, np.c_[f_pos, d_pos], fmt="%.8f %.12e", header="freq_THz dos (freq_min=0.0)") (0이상인 주파수만 저장)
    dos_int_pos = float(np.trapezoid(d_pos, f_pos)) (계산)

-------------------------------------------------------0 이상 주파수 영역 DOS 생성 -----------------------------------------------------------------------

    dos_int_imag = None
    wrote_imag = False
    if SAVE_WITH_IMAG_IF_NEGATIVE and (minf is not None) and (minf < 0.0):
        run_total_dos_safe(phonon, freq_min=-float(IMAG_FREQ_MIN_THz))
        f_im, d_im = extract_total_dos(phonon)
        np.savetxt(out_dos_imag, np.c_[f_im, d_im], fmt="%.8f %.12e",
                   header=f"freq_THz dos (freq_min=-{IMAG_FREQ_MIN_THz})")
        dos_int_imag = float(np.trapezoid(d_im, f_im))
        wrote_imag = True

-------------------------------------------------------imaginary가 있을 때 음수 영역 DOS 추가 저장---------------------------------------------------------

    phonon.run_thermal_properties(t_step=10, t_max=1000, t_min=0) (공식 phonopy 기반, 0 부터 1000K 까지 10K 간격으로 thermal 계산)
    phonon.write_yaml_thermal_properties(filename=out_tp) (thermal.properties.yaml 파일로 저장)

free energy, entropy, heat capaciry 등 저장

-------------------------------------------------------thermal properties 계산 --------------------------------------------------------------------------

    t_all = time.perf_counter() - t_all0

    with open(out_time, "w") as fp:
        fp.write(f"# timing log {now()}\n")
        fp.write(f"phase {phase}\n")
        fp.write(f"TAG {TAG}\n")
        fp.write(f"MODEL {MODEL}\n")
        fp.write(f"MODAL {MODAL}\n")
        fp.write(f"DEVICE {DEVICE}\n")
        fp.write(f"IN_VASP {in_vasp}\n")
        fp.write(f"SUPERCELL_MATRIX {supercell_matrix}\n")
        fp.write(f"DISP_DISTANCE_A {disp}\n")
        fp.write(f"primitive_N_atoms {n_prim}\n")
        fp.write(f"3N_expected {threeN}\n")
        if minf is not None:
            fp.write(f"mesh_min_freq_THz {minf:.8f}\n")
            fp.write(f"mesh_neg_mode_count {negc}\n")
        fp.write(f"dos_int_pos (freq_min=0) {dos_int_pos:.10f}\n")
        if wrote_imag and dos_int_imag is not None:
            fp.write(f"dos_int_with_imag (freq_min=-{IMAG_FREQ_MIN_THz}) {dos_int_imag:.10f}\n")
        fp.write(f"total_wall_s {t_all:.6f}\n")

-------------------------------------------------------계산 시간이 얼마나 걸리는지 ------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) (SevenNet calculator를 끄고 > 모든 phase 반복)
    calc = SevenNetCalculator(model=MODEL, modal=MODAL, device=DEVICE)
    for ph in PHASES:
        run_fcm_one(ph, IN_VASP_MAP[ph], SUPERCELL_MAP[ph], calc)


if __name__ == "__main__":
    main()
-------------------------------------------------------모든 phase 반복 실행 --- --------------------------------------------------------------------------
# Quassi Harmonic Approximation
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re (부피 scale 이름 0.92, 0.94 등 scale 형식 판별 시 사용)
import json
import shutil
import subprocess ( phonopy -qha 같은 CLI 명령어 사용 시 필요)
import numpy as np
from pathlib import Path

from ase.io import read, write (read > VASP 구조 읽기, write > POSCAR 파일 쓰기)
from ase.optimize import BFGS (position only relax BFGS 수행) 

try:
    from sevenn.calculator import SevenNetCalculator (SeveNet calcultor > energy 및 force 계산)
except Exception:
    from sevenn.sevennet_calculator import SevenNetCalculator

import phonopy(phonopy 사용 불러오기)
from phonopy.file_IO import write_FORCE_CONSTANTS(force constant > FORCE_CONSTANTS)
-------------------------------------------------------라이브러리 불러오는 구간---------------------------------------------------------------------------

MODEL  = "7net-omni"
MODAL  = "mp_r2scan"
TAG    = "mp_r2scan_v2"
DEVICE = "cuda:1"

MESH  = (24, 24, 24)(phonon thermal properties 계산용 mesh)
TMIN  = 0
TMAX  = 800
TSTEP = 10

SCALES = [round(x, 2) for x in np.arange(0.92, 1.201, 0.02)] (부피 변위 설정)

DISP_DISTANCE = 0.01(FCM과 동일)
EV_ORDER = "VE"(e-v data > volume energy 순서로 작성)

DO_POS_RELAX = True (position relax) --AI
RELAX_FMAX = 5e-3(relax 수렴 기준)
RELAX_STEPS = 300(relax 최대 step 수)

EV_USE_SUPERCELL_TOTAL = False (e-v data > input cell 기준 사용, True면 supercell 기준 사용)

ENABLE_SPGLIB_STANDARDIZE = True (spglib로 부피 scale에 맞게 space group을 정리) --AI
SYMPREC = 1e-4 (비슷한 대칭을 판단하는 허용오차값) --AI
NO_IDEALIZE = True (현재 계산된 구조를 건드리지 않고 이상적인 구조로 판단 X)--AI

CLEAN_PHASE_DIRS = True
RECREATE_EXISTING_SCALE_DIR = False (폴더 정리)

PREFIX = "re_mp_r2scan_v2_"
OUTROOT = Path(f"re_mp_r2scan_qha_api_{TAG}")
OUTROOT.mkdir(parents=True, exist_ok=True)

-------------------------------------------------실제 계산 조건을 정하는 단계------------------------------------------------------------------------------

IN_VASP_MAP = {
    "alpha": "./r2scan/structure_file/mp_r2scan_alpha_cellopt_7net.vasp",
    "beta":  "./r2scan/structure_file/mp_r2scan_beta_cellopt_7net.vasp",
    "gamma": "./r2scan/structure_file/mp_r2scan_gamma_cellopt_7net.vasp",
    "delta": "./r2scan/structure_file/mp_r2scan_delta_cellopt_7net.vasp",
}

SUPERCELL_MAP = {
    "alpha": [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    "beta":  [[4, 0, 0], [0, 4, 0], [0, 0, 3]],
    "gamma": [[4, 0, 0], [0, 4, 0], [0, 0, 3]],
    "delta": [[6, 0, 0], [0, 4, 0], [0, 0, 2]],
}

PHASES = ["alpha", "beta", "gamma", "delta"]

-------------------------------------------------직접 설정-----------------------------------------------------------------------------------------------

def run(cmd, cwd=None, capture=True):
    print(f"[CMD] {cmd} (cwd={cwd})")
    p = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=capture) (phonopy -qha와 같은 명령어를 subprocess로 사용)
    if p.returncode != 0: (phonopy 성공, 실패 여부를 help text로 읽고 검사) --AI
        if cwd is not None:
            Path(cwd, f"{PREFIX}cmd_failed.txt").write_text(cmd + "\n")
            Path(cwd, f"{PREFIX}stdout_failed.txt").write_text(p.stdout or "")
            Path(cwd, f"{PREFIX}stderr_failed.txt").write_text(p.stderr or "")
        if capture:
            print("----- STDOUT -----")
            print(p.stdout)
            print("----- STDERR -----")
            print(p.stderr)
        raise RuntimeError(f"[CMD FAIL] {cmd}")
    if capture:
        if (p.stdout or "").strip():
            print((p.stdout or "").strip())
        if (p.stderr or "").strip():
            print((p.stderr or "").strip())


def phonopy_help_text() -> str:
    p = subprocess.run("phonopy -h", shell=True, text=True, capture_output=True)
    return (p.stdout or "") + (p.stderr or "")


PHONOPY_HELP = phonopy_help_text()


def phonopy_supports(opt_name: str) -> bool:
    return opt_name in PHONOPY_HELP
----------------------------------------------phonopy CLI 명령어 실행 및 phonopy 버전의 옵션 지원 확인-----------------------------------------------------

def cleanup_phonopy_artifacts(scale_dir: Path): (QHA 처럼 다양한 부피 변위들이 섞이지 않게 정리하는 구간)
    patterns = [
        "POSCAR-*",
        "SPOSCAR*",
        "phonopy_disp.yaml", "phonopy_disp.yml",
        "phonopy.yaml", "phonopy.yml",
        "FORCE_CONSTANTS",
        "FORCE_SETS",
        "thermal_properties.yaml", "thermal_properties.yml",
        "mesh.yaml", "mesh.hdf5",
        "band.yaml", "qpoints.yaml",
    ]
    for pat in patterns:
        for p in scale_dir.glob(pat):
            try:
                p.unlink()
            except Exception:
                pass


def prepare_phase_dir(phase_dir: Path, scales):
    phase_dir.mkdir(exist_ok=True, parents=True)
    keep = set([f"{s:.2f}" for s in scales])

    for child in phase_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.strip()
        if re.fullmatch(r"\d+\.\d+", name) is None:
            continue
        if CLEAN_PHASE_DIRS and (name not in keep):
            print(f"[CLEAN] remove stale scale dir: {child}")
            shutil.rmtree(child, ignore_errors=True)

    if RECREATE_EXISTING_SCALE_DIR:
        for tag in keep:
            sdir = phase_dir / tag
            if sdir.exists():
                print(f"[RESET] recreate scale dir: {sdir}")
                shutil.rmtree(sdir, ignore_errors=True)
            sdir.mkdir(parents=True, exist_ok=True)

------------------------------------------------------------계산 폴더와 이전 계산 폴더가 혼동되지 않게 정리 -----------------------------------------------

def scale_atoms_isotropic(atoms, scale: float): (cell을 isotropic하게 sclae > 여러 부피 구조 변위 생성 전 필수 단계)
    a = atoms.copy()
    a.set_cell(a.cell * scale, scale_atoms=True)
    return a


def relax_positions_only(atoms, calc): (SevenNet calculator 적용, position only relax 진행) --AI
    a = atoms.copy()
    a.calc = calc
    cell_before = a.cell.copy()
    opt = BFGS(a, logfile=None) (BFGS로 relax 후 relaxation 끝난 구조 생성)
    opt.run(fmax=RELAX_FMAX, steps=RELAX_STEPS)
    if np.max(np.abs(a.cell.array - cell_before.array)) > 1e-12: (relax 후 cell이 바뀌었는지 검사 > 바뀔 시 error) --AI
        raise RuntimeError("Cell changed during position-only relax (unexpected).")
    return a

--------------------------------------------------------------구조 scale과 position only relax 구간-----------------------------------------------------

def dim_str_from_sc_matrix(mat3x3):
    return f"{mat3x3[0][0]} {mat3x3[1][1]} {mat3x3[2][2]}" (phonopy CLI의 matrix 문자열을 python에 맞게 변환)


def phonopy_disp_cmd(dim_str: str, disp: float | None) -> str: (변환에 맞게 phonopy -d 명령어를 생성, 구조변위 생성 명령어)
    if disp is None:
        return f'phonopy -d --dim="{dim_str}" -c POSCAR'
    if "--amplitude" in PHONOPY_HELP:
        return f'phonopy -d --dim="{dim_str}" --amplitude {disp} -c POSCAR'
    if "--distance" in PHONOPY_HELP:
        return f'phonopy -d --dim="{dim_str}" --distance={disp} -c POSCAR'
    return f'phonopy -d --dim="{dim_str}" -c POSCAR'


def find_disp_yaml(scale_dir: Path) -> Path: (phonopy_disp.yaml, phonopy.yaml 등 구조 변위가 담긴 yaml 파일 자동 찾기) --AI
    cand = [
        scale_dir / "phonopy_disp.yaml",
        scale_dir / "phonopy_disp.yml",
        scale_dir / "phonopy.yaml",
        scale_dir / "phonopy.yml",
    ]
    for p in cand:
        if p.exists():
            return p
    yamls = sorted(scale_dir.glob("*.yaml")) + sorted(scale_dir.glob("*.yml"))
    if not yamls:
        raise RuntimeError(f"No yaml produced by phonopy -d in {scale_dir}")
    for p in yamls:
        if "disp" in p.name.lower():
            return p
    return yamls[0]


def list_displaced_poscars(scale_dir: Path): (0.02 ~1.20까지 POSCAR 번호가 생성되는데 이때 번호를 순서대로 정렬)
    files = list(scale_dir.glob("POSCAR-*"))
    def key(p: Path):
        m = re.search(r"POSCAR-(\d+)", p.name)
        return int(m.group(1)) if m else 10**9
    return sorted(files, key=key)


def load_phonopy_instance_from_yaml(yaml_path: str): (구조 변위 생성 > phonopy CLI, force 계산은 ASE, force constant는 phonopy로 만들기 때문에 변환하기 위한 방법)
    try:
        ph = phonopy.load(yaml_path)
        if ph is None:
            raise RuntimeError("phonopy.load returned None")
        return ph
    except Exception:
        from phonopy.interface.phonopy_yaml import PhonopyYaml
        pyml = PhonopyYaml()
        pyml.read(yaml_path)
        ph = pyml.get_phonopy()
        if ph is None:
            raise RuntimeError("PhonopyYaml.get_phonopy returned None")
        return ph

-------------------------------------------------------phonopy displacement 구조 생성 준비-------------------------------------------------------------

def ase_to_forces_list(displaced_poscars, calc): (ASE 방법으로 force 계산)
    forces_list = []
    natoms = None
    for i, ppos in enumerate(displaced_poscars, 1):
        sc = read(ppos.as_posix(), format="vasp") (구조변위 파일을 읽기)
        sc.calc = calc (SevenNet calculator 사용)
        f = np.array(sc.get_forces(), float) (force 계산 및 numpy 배열 정리)
        if natoms is None:
            natoms = f.shape[0]
        elif f.shape[0] != natoms:
            raise RuntimeError(f"natoms mismatch at {ppos.name}: {f.shape[0]} vs {natoms}")
        forces_list.append(f) (force 계산한 list 추가)
        print(f"[force] {i:03d}/{len(displaced_poscars):03d} done")
    return forces_list


def calc_nrep(sc_mat) -> int: (supercell 전체 에너지 및 부피 변환 시 사용)
    a = int(sc_mat[0][0]); b = int(sc_mat[1][1]); c = int(sc_mat[2][2])
    return a * b * c

-----------------------------------------------------phonopy 구조 변위 force 계산-------------------------------------------------------------------------

def symmetrize_atoms_spglib(atoms, symprec=SYMPREC): (대칭 분석 및 구조 부피 변위 표준화에 맞게 정리) --AI
    if not ENABLE_SPGLIB_STANDARDIZE:
        return atoms
    try:
        import spglib
    except Exception:
        return atoms

    cell = np.array(atoms.cell.array, dtype=float)
    pos = np.array(atoms.get_scaled_positions(), dtype=float)
    nums = np.array(atoms.get_atomic_numbers(), dtype=int)
    spg_cell = (cell, pos, nums)

    std = spglib.standardize_cell(
        spg_cell,
        to_primitive=False,
        no_idealize=NO_IDEALIZE,
        symprec=symprec
    )
    if std is None:
        return atoms

    std_cell, std_pos, std_nums = std
    from ase import Atoms
    new_atoms = Atoms(
        numbers=list(std_nums),
        scaled_positions=np.array(std_pos, dtype=float),
        cell=np.array(std_cell, dtype=float),
        pbc=True
    )
    return new_atoms

------------------------------------------spglib 대칭 안정화 구간 ---------------------------------------------------------------------------------------

def enforce_fc_asr_and_symmetry(phonon_obj): (force constant 값의 오차나 noise들을 대칭과 같은 후처리로 보정) --AI
    fc = phonon_obj.force_constants
    if fc is None:
        raise RuntimeError("force_constants is None")

    try:
        from phonopy.harmonic.force_constants import (
            set_translational_invariance,
            set_permutation_symmetry,
        )
        set_translational_invariance(fc)
        set_permutation_symmetry(fc)
    except Exception:
        pass

    try:
        if hasattr(phonon_obj, "symmetrize_force_constants"):
            phonon_obj.symmetrize_force_constants()
    except Exception:
        pass

    phonon_obj.force_constants = fc

-------------------------------------------force constants 후처리 구간-----------------------------------------------------------------------------------

def phonopy_t_cmd(disp_yaml_name: str, mesh: str) -> str: (phonopy - t, thermal properties 계산)
    base = (
        f'phonopy -t --readfc -c "{disp_yaml_name}" --mesh="{mesh}" ' (이미 만든 FORCE_COSNTATNS를 읽고, yaml 파일 및 mesh 지정)
        f'--tmin={TMIN} --tmax={TMAX} --tstep={TSTEP}'(온도 범위 지정)
    )
    if phonopy_supports("--asr"):
        return base + " --asr"
    return base

---------------------------------------------thermal properties 명렁 생성 구간---------------------------------------------------------------------------
# QHA 계산 시작
def run_one_scale(phase: str, scale: float, calc: SevenNetCalculator): (alpha phase에 0.92 변위 > ../alpha/0.92/ 디렉토리 생성
    phase_dir = OUTROOT / phase 
    phase_dir.mkdir(exist_ok=True, parents=True)

    scale_tag = f"{scale:.2f}"
    sdir = phase_dir / scale_tag
    sdir.mkdir(exist_ok=True, parents=True)

--------------------------------------------phase 마다 scale 별 작업 폴더 생성 정리-----------------------------------------------------------------------

    in_vasp = IN_VASP_MAP[phase]
    if not os.path.exists(in_vasp): 
        raise FileNotFoundError(in_vasp)

    atoms0 = read(in_vasp, format="vasp")
    atoms = scale_atoms_isotropic(atoms0, scale)

    if DO_POS_RELAX:
        atoms = relax_positions_only(atoms, calc)

    atoms = symmetrize_atoms_spglib(atoms, symprec=SYMPREC)

-------------------------------------------입력 구조 > sclae 적용 > position relax+대칭 안정화------------------------------------------------------------

    write((sdir / "POSCAR").as_posix(), atoms, format="vasp", direct=True, vasp5=True)

    atoms.calc = calc 
    e_static = float(atoms.get_potential_energy()) (atoms.get_potential_energy로 static enery 계산)
    vol = float(atoms.get_volume()) (volume 계산)
    (sdir / "E_static.txt").write_text(f"{e_static:.16f}\n")
    (sdir / "volume.txt").write_text(f"{vol:.16f}\n")

----------------------------------------------POSCAR(scale별) 저장 및 E-V data 및 volume 계산-------------------------------------------------------------

    cleanup_phonopy_artifacts(sdir)

    sc_mat = SUPERCELL_MAP[phase]
    dim_str = dim_str_from_sc_matrix(sc_mat) (supercell에 맞게 각 상의 구조변위 생성)
    run(phonopy_disp_cmd(dim_str, DISP_DISTANCE), cwd=sdir, capture=True)

    displaced = list_displaced_poscars(sdir) (생성된 구조 변위 찾고 yaml 파일을 찾아 각 해당 디렉토리에 저장)
    if not displaced:
        raise RuntimeError(f"[{phase} {scale_tag}] No displaced POSCAR-* generated.")
    disp_yaml = find_disp_yaml(sdir)

    ph = load_phonopy_instance_from_yaml(disp_yaml.as_posix())

--------------------------------------------------phonopy 구조 변위 생성 및 구조 목록 정리, yaml 파일 찾기-------------------------------------------------

    ds = getattr(ph, "dataset", None) (지금까지의 data를 가져오기)
    n_ds = None
    if isinstance(ds, dict) and "first_atoms" in ds:
        n_ds = len(ds["first_atoms"]) (구조 변위 수 정리)

    forces_list = ase_to_forces_list(displaced, calc) (force list 생성하여 구조 변위 개수와 force가 일치하는지 확인)
    if (n_ds is not None) and (len(forces_list) != n_ds):
        raise RuntimeError(
            f"[{phase} {scale_tag}] mismatch: dataset displacements={n_ds} "
            f"but forces_list={len(forces_list)}."
        )

---------------------------------------------------dataset 개수 점검 후 force 계산------------------------------------------------------------------------

    ph.forces = forces_list
    ph.produce_force_constants()
    enforce_fc_asr_and_symmetry(ph)

---------------------------------------------------force를 phonopy에 넣고 force constant 생성-------------------------------------------------------------

    write_FORCE_CONSTANTS(ph.force_constants, filename=(sdir / "FORCE_CONSTANTS").as_posix()) (저장)

    mesh = f"{MESH[0]} {MESH[1]} {MESH[2]}"
    run(phonopy_t_cmd(disp_yaml.name, mesh), cwd=sdir, capture=True) (phonopy -t로 thermal properties 계산)

------------------------------------------------FORCE CONSTANTS 저장 및 thermal 계산---------------------------------------------------------------------

    tp_src = sdir / "thermal_properties.yaml" (thermal properties.yaml 파일을 scale 별로 디렉토리에 저장)
    if not tp_src.exists():
        tp_alt = sdir / "thermal_properties.yml"
        if tp_alt.exists():
            tp_src = tp_alt
        else:
            raise RuntimeError(f"[{phase} {scale_tag}] thermal_properties.yaml(.yml) not found.")

    tp_dst = phase_dir / f"{PREFIX}{scale_tag}_thermal_properties.yaml"
    shutil.copy(tp_src.as_posix(), tp_dst.as_posix())

    meta = { (thermal properties.yaml에 들어가는 내용을 각각 meta.json 파일에 따로 저장, 각 상마다 부피 scale이 많아 thermal properties를 한번에 합친 후 data들을 생성 할 떄 편리하기 위해) --AI
        "phase": phase,
        "scale": scale,
        "volume_inputcell": vol,
        "E_static_inputcell_eV": e_static,
        "supercell_matrix": sc_mat,
        "nrep": calc_nrep(sc_mat),
        "dim_str_for_phonopy": dim_str,
        "disp_yaml": disp_yaml.name,
        "mesh": list(MESH),
        "tmin": TMIN, "tmax": TMAX, "tstep": TSTEP,
        "model": MODEL, "modal": MODAL, "device": DEVICE,
        "n_displacements": len(displaced),
        "pos_relax": {"enabled": DO_POS_RELAX, "fmax": RELAX_FMAX, "steps": RELAX_STEPS},
        "disp_distance": DISP_DISTANCE,
        "symprec": SYMPREC,
        "no_idealize": NO_IDEALIZE,
        "enable_spglib_standardize": ENABLE_SPGLIB_STANDARDIZE,
        "asr_in_fc": True,
        "asr_in_thermal_cli": phonopy_supports("--asr"),
        "ev_use_supercell_total": EV_USE_SUPERCELL_TOTAL,
        "natoms_input_cell": int(len(atoms)),
        "natoms_supercell_expected": int(len(atoms) * calc_nrep(sc_mat)),
    }
    (sdir / f"{PREFIX}meta.json").write_text(json.dumps(meta, indent=2))



def run_qha_for_phase(phase: str): (phonopy -qha 명령어와 같은 것으로 데이터 생성)
    phase_dir = OUTROOT / phase

---------------------------------------------thermal yaml 가져와 정리 후 QHA 데이터 생성-----------------------------------------------------------------

    meta_path = phase_dir / "1.00" / f"{PREFIX}meta.json"
    if not meta_path.exists():
        metas = sorted(phase_dir.glob(f"*/{PREFIX}meta.json"))
        if not metas:
            raise RuntimeError(f"[{phase}] meta.json not found")
        meta_path = metas[0]
    meta = json.loads(meta_path.read_text())
    nrep = int(meta.get("nrep", calc_nrep(SUPERCELL_MAP[phase])))

---------------------------------------------각 부피 scale 마다 data 정리한 meta.json을 수집-------------------------------------------------------------

    ev_rows = []
    tp_files = []

    for s in SCALES:
        tag = f"{s:.2f}"
        sdir = phase_dir / tag

        V = float((sdir / "volume.txt").read_text().strip()) (volume, static energy, thermal 파일 등을 가져옴)
        E = float((sdir / "E_static.txt").read_text().strip())

        if EV_USE_SUPERCELL_TOTAL:
            V *= nrep
            E *= nrep

        ev_rows.append((V, E))

        tp = phase_dir / f"{PREFIX}{tag}_thermal_properties.yaml"
        if not tp.exists():
            raise RuntimeError(f"[{phase}] Missing thermal file: {tp.name}")
        tp_files.append(tp.name)

-------------------------------------------각 sclae의 필수 data 일치한지 확인 후 수집---------------------------------------------------------------------

    ev_rows = sorted(ev_rows, key=lambda x: x[0])

    ev_path = phase_dir / f"{PREFIX}e-v.dat" (e-v data 수집 후 생성)
    if EV_ORDER.upper() == "VE":
        ev_path.write_text("\n".join([f"{V:.16f} {E:.16f}" for (V, E) in ev_rows]) + "\n")
    else:
        ev_path.write_text("\n".join([f"{E:.16f} {V:.16f}" for (V, E) in ev_rows]) + "\n")

    run(f"phonopy-qha -p {PREFIX}e-v.dat " + " ".join(tp_files), cwd=phase_dir, capture=True) (공식 기반 명령어 실행)
    (phase_dir / f"{PREFIX}qha_done.txt").write_text("phonopy-qha finished successfully.\n")

----------------------------------------E-V data 생성 및 phonopy -qha 실행-------------------------------------------------------------------------------

def main():
    print("[INFO] N_SCALES =", len(SCALES))
    print("[INFO] SCALES =", SCALES)
    print("[INFO] supports --amplitude :", phonopy_supports("--amplitude"))
    print("[INFO] supports --distance  :", phonopy_supports("--distance"))
    print("[INFO] supports --asr       :", phonopy_supports("--asr"))
    print("[INFO] PREFIX =", PREFIX)
    print("[INFO] OUTROOT =", OUTROOT.resolve())
    print("[INFO] EV_USE_SUPERCELL_TOTAL =", EV_USE_SUPERCELL_TOTAL)
    print("[INFO] ENABLE_SPGLIB_STANDARDIZE =", ENABLE_SPGLIB_STANDARDIZE)
    print("[INFO] SYMPREC =", SYMPREC, "NO_IDEALIZE =", NO_IDEALIZE)

    calc = SevenNetCalculator(model=MODEL, modal=MODAL, device=DEVICE)

    for phase in PHASES:
        print("\n" + "=" * 90)
        print(f"[PHASE] {phase}  (TAG={TAG}, MODAL={MODAL})")
        print("=" * 90)

        phase_dir = OUTROOT / phase
        prepare_phase_dir(phase_dir, SCALES)

        for s in SCALES:
            print(f"\n--- {phase} scale={s:.2f} ---")
            run_one_scale(phase, s, calc)

        print(f"\n[QHA] running phonopy-qha for phase={phase}")
        run_qha_for_phase(phase)

    print("\nALL DONE")
    print(f"Results root: {OUTROOT.resolve()}")


if __name__ == "__main__":
    main()
--------------------------------------각 phase 마다 위를 똑같이 반복 및 실행----------------------------------------------------------------------------