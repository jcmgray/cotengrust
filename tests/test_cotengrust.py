import pytest
import numpy as np
from numpy.testing import assert_allclose
import cotengra as ctg
import cotengrust as ctgr


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
    size_dict = {}
    for term in inputs:
        for ix in term:
            if ix not in size_dict:
                size_dict[ix] = np.random.randint(d_min, d_max + 1)
    return size_dict


def build_arrays(inputs, size_dict):
    return [
        np.random.randn(*[size_dict[ix] for ix in term]) for term in inputs
    ]


# these are taken from opt_einsum
test_case_eqs = [
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


@pytest.mark.parametrize("eq", test_case_eqs)
@pytest.mark.parametrize("which", ["greedy", "optimal"])
def test_manual_cases(eq, which):
    inputs, output = eq_to_inputs_output(eq)
    size_dict = get_rand_size_dict(inputs)
    arrays = build_arrays(inputs, size_dict)
    expected = np.einsum(eq, *arrays, optimize=True)
    path = {
        "greedy": ctgr.optimize_greedy,
        "optimal": ctgr.optimize_optimal,
    }[
        which
    ](inputs, output, size_dict)
    assert all(len(con) <= 2 for con in path)
    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    assert_allclose(tree.contract(arrays), expected)


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
    eq = ",".join(map("".join, inputs)) + "->" + "".join(output)

    path = {
        "greedy": ctgr.optimize_greedy,
        "optimal": ctgr.optimize_optimal,
    }[
        which
    ](inputs, output, size_dict)
    assert all(len(con) <= 2 for con in path)
    tree = ctg.ContractionTree.from_path(inputs, output, size_dict, path=path)
    arrays = [np.random.randn(*s) for s in shapes]
    assert_allclose(
        tree.contract(arrays), np.einsum(eq, *arrays, optimize=True)
    )


def test_optimal_lattice_eq():
    inputs, output, _, size_dict = ctg.utils.lattice_equation(
        [4, 5], d_max=3, seed=42
    )

    path = ctgr.optimize_optimal(inputs, output, size_dict, minimize='flops')
    tree = ctg.ContractionTree.from_path(
        inputs, output, size_dict, path=path
    )
    assert tree.contraction_cost() == 3628

    path = ctgr.optimize_optimal(inputs, output, size_dict, minimize='size')
    assert all(len(con) <= 2 for con in path)
    tree = ctg.ContractionTree.from_path(
        inputs, output, size_dict, path=path
    )
    assert tree.contraction_width() == pytest.approx(6.754887502163468)
