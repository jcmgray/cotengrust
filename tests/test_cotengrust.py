import pytest

try:
    import cotengra as ctg

    ctg_missing = False
except ImportError:
    ctg_missing = True
    ctg = None

import cotengrust as ctgr


requires_cotengra = pytest.mark.skipif(ctg_missing, reason="requires cotengra")


@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_basic_call(which):
    inputs = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'a')]
    output = ('b', 'd')
    size_dict = {'a': 2, 'b': 3, 'c': 4, 'd': 5}
    path = {
        "greedy": ctgr.optimize_greedy,
        "optimal": ctgr.optimize_optimal,
    }[
        which
    ](inputs, output, size_dict)
    assert all(len(con) <= 2 for con in path)


def find_output_str(lhs):
    tmp_lhs = lhs.replace(",", "")
    return "".join(s for s in sorted(set(tmp_lhs)) if tmp_lhs.count(s) == 1)


def eq_to_inputs_output(eq):
    if "->" not in eq:
        eq += "->" + find_output_str(eq)
    inputs, output = eq.split("->")
    inputs = inputs.split(",")
    inputs = [list(s) for s in inputs]
    output = list(output)
    return inputs, output


def get_rand_size_dict(inputs, d_min=2, d_max=3):
    import random

    size_dict = {}
    for term in inputs:
        for ix in term:
            if ix not in size_dict:
                size_dict[ix] = random.randint(d_min, d_max)
    return size_dict


# these are taken from opt_einsum
test_case_eqs = [
    # Test single-term equations
    "->",
    "a->a",
    "ab->ab",
    "ab->ba",
    "abc->bca",
    "abc->b",
    "baa->ba",
    "aba->b",
    # Test scalar-like operations
    "a,->a",
    "ab,->ab",
    ",ab,->ab",
    ",,->",
    # Test hadamard-like products
    "a,ab,abc->abc",
    "a,b,ab->ab",
    # Test index-transformations
    "ea,fb,gc,hd,abcd->efgh",
    "ea,fb,abcd,gc,hd->efgh",
    "abcd,ea,fb,gc,hd->efgh",
    # Test complex contractions
    "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
    "cd,bdhe,aidb,hgca,gc,hgibcd,hgac",
    "abhe,hidj,jgba,hiab,gab",
    "bde,cdh,agdb,hica,ibd,hgicd,hiac",
    "chd,bde,agbc,hiad,hgc,hgi,hiad",
    "chd,bde,agbc,hiad,bdi,cgh,agdb",
    "bdhe,acad,hiab,agac,hibd",
    # Test collapse
    "ab,ab,c->",
    "ab,ab,c->c",
    "ab,ab,cd,cd->",
    "ab,ab,cd,cd->ac",
    "ab,ab,cd,cd->cd",
    "ab,ab,cd,cd,ef,ef->",
    # Test outer prodcuts
    "ab,cd,ef->abcdef",
    "ab,cd,ef->acdf",
    "ab,cd,de->abcde",
    "ab,cd,de->be",
    "ab,bcd,cd->abcd",
    "ab,bcd,cd->abd",
    # Random test cases that have previously failed
    "eb,cb,fb->cef",
    "dd,fb,be,cdb->cef",
    "bca,cdb,dbf,afc->",
    "dcc,fce,ea,dbf->ab",
    "fdf,cdd,ccd,afe->ae",
    "abcd,ad",
    "ed,fcd,ff,bcf->be",
    "baa,dcf,af,cde->be",
    "bd,db,eac->ace",
    "fff,fae,bef,def->abd",
    "efc,dbc,acf,fd->abe",
    # Inner products
    "ab,ab",
    "ab,ba",
    "abc,abc",
    "abc,bac",
    "abc,cba",
    # GEMM test cases
    "ab,bc",
    "ab,cb",
    "ba,bc",
    "ba,cb",
    "abcd,cd",
    "abcd,ab",
    "abcd,cdef",
    "abcd,cdef->feba",
    "abcd,efdc",
    # Inner than dot
    "aab,bc->ac",
    "ab,bcc->ac",
    "aab,bcc->ac",
    "baa,bcc->ac",
    "aab,ccb->ac",
    # Randomly built test caes
    "aab,fa,df,ecc->bde",
    "ecb,fef,bad,ed->ac",
    "bcf,bbb,fbf,fc->",
    "bb,ff,be->e",
    "bcb,bb,fc,fff->",
    "fbb,dfd,fc,fc->",
    "afd,ba,cc,dc->bf",
    "adb,bc,fa,cfc->d",
    "bbd,bda,fc,db->acf",
    "dba,ead,cad->bce",
    "aef,fbc,dca->bde",
]


@requires_cotengra
@pytest.mark.parametrize("eq", test_case_eqs)
@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_manual_cases(eq, which):
    inputs, output = eq_to_inputs_output(eq)
    size_dict = get_rand_size_dict(inputs)
    path = {
        "greedy": ctgr.optimize_greedy,
        "optimal": ctgr.optimize_optimal,
    }[
        which
    ](inputs, output, size_dict)
    assert all(len(con) <= 2 for con in path)
    tree = ctg.ContractionTree.from_path(
        inputs, output, size_dict, path=path, check=True
    )
    assert tree.is_complete()


@requires_cotengra
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_basic_rand(seed, which):
    inputs, output, shapes, size_dict = ctg.utils.rand_equation(
        n=10,
        reg=4,
        n_out=2,
        n_hyper_in=1,
        n_hyper_out=1,
        d_min=2,
        d_max=3,
        seed=seed,
    )
    path = {
        "greedy": ctgr.optimize_greedy,
        "optimal": ctgr.optimize_optimal,
    }[
        which
    ](inputs, output, size_dict)
    assert all(len(con) <= 2 for con in path)
    tree = ctg.ContractionTree.from_path(
        inputs, output, size_dict, path=path, check=True
    )
    assert tree.is_complete()


@requires_cotengra
def test_optimal_lattice_eq():
    inputs, output, _, size_dict = ctg.utils.lattice_equation(
        [4, 5], d_max=2, seed=42
    )

    path = ctgr.optimize_optimal(inputs, output, size_dict, minimize='flops')
    tree = ctg.ContractionTree.from_path(
        inputs, output, size_dict, path=path
    )
    assert tree.is_complete()
    assert tree.contraction_cost() == 964

    path = ctgr.optimize_optimal(inputs, output, size_dict, minimize='size')
    assert all(len(con) <= 2 for con in path)
    tree = ctg.ContractionTree.from_path(
        inputs, output, size_dict, path=path
    )
    assert tree.contraction_width() == pytest.approx(5)


@requires_cotengra
def test_optimize_random_greedy_log_flops():
    inputs, output, _, size_dict = ctg.utils.lattice_equation(
        [10, 10], d_max=3, seed=42
    )

    path, cost1 = ctgr.optimize_random_greedy_track_flops(
        inputs, output, size_dict, ntrials=4, seed=42
    )
    _, cost2 = ctgr.optimize_random_greedy_track_flops(
        inputs, output, size_dict, ntrials=4, seed=42
    )
    assert cost1 == cost2
    tree = ctg.ContractionTree.from_path(
        inputs, output, size_dict, path=path
    )
    assert tree.is_complete()
    assert tree.contraction_cost(log=10) == pytest.approx(cost1)