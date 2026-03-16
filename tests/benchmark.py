import cotengra as ctg
import cotengrust as ctgr


def _auto_min_time(timer, min_t=0.2, repeats=2, get="min"):
    tot_t = 0
    number = 1

    while True:
        tot_t = timer.timeit(number)
        if tot_t > min_t:
            break
        number *= 2

    results = [tot_t] + timer.repeat(repeats - 1, number)

    if get == "mean":
        return sum(results) / (number * len(results))

    return min(t / number for t in results)


def benchmark(fn, setup=None, n=None, min_t=0.2, repeats=2, get="min"):
    from timeit import Timer

    if n is None:
        n = ""

    if setup is None:
        setup_str = ""
        stmnt_str = "fn({})".format(n)
    else:
        setup_str = "X=setup({})".format(n)
        stmnt_str = "fn(X)"

    timer = Timer(setup=setup_str, stmt=stmnt_str, globals={"setup": setup, "fn": fn})

    return _auto_min_time(timer, min_t=min_t, repeats=repeats, get=get)


def benchmark_optimal(c, minimize="flops"):
    def fn():
        return ctgr.optimize_optimal(
            c.inputs,
            c.output,
            c.size_dict,
            minimize=minimize,
        )

    return benchmark(fn)


def benchmark_random_greedy(c, ntrials=128, seed=42):
    def fn():
        return ctgr.optimize_random_greedy_track_flops(
            c.inputs,
            c.output,
            c.size_dict,
            ntrials=ntrials,
            use_ssa=True,
            seed=seed,
        )

    return benchmark(fn)


if __name__ == "__main__":
    print("[Benchmark random-greedy-128 on 32x32 lattice...]")
    c = ctg.utils.lattice_equation([32, 32])
    t = benchmark_random_greedy(c, ntrials=128, seed=42)
    print(f"    Random-greedy-128 time:    {t:.3e} s")

    print("[Benchmark random-greedy-128 on N=512 random 3-regular graph...]")
    c = ctg.utils.randreg_equation(512, 3, seed=42)
    t = benchmark_random_greedy(c, ntrials=128, seed=42)
    print(f"    Random-greedy-128 time:    {t:.3e} s")

    print("[Benchmark random-greedy-128 on rand hypergraph equation...]")
    c = ctg.utils.rand_equation(256, 5, 5, 5, 5, d_min=2, d_max=10, seed=42)
    t = benchmark_random_greedy(c, ntrials=128, seed=42)
    print(f"    Random-greedy-128 time:    {t:.3e} s")

    print("[Benchmark optimal on 5x5 lattice...]")
    c = ctg.utils.lattice_equation([5, 5])
    t = benchmark_optimal(c, minimize="flops")
    print(f"    Optimal (flops) time:      {t:.3e} s")
    t = benchmark_optimal(c, minimize="max")
    print(f"    Optimal (max)   time:      {t:.3e} s")
    t = benchmark_optimal(c, minimize="size")
    print(f"    Optimal (size)  time:      {t:.3e} s")
    t = benchmark_optimal(c, minimize="write")
    print(f"    Optimal (write) time:      {t:.3e} s")
    t = benchmark_optimal(c, minimize="combo")
    print(f"    Optimal (combo) time:      {t:.3e} s")
    t = benchmark_optimal(c, minimize="limit")
    print(f"    Optimal (limit) time:      {t:.3e} s")

    print("[Benchmark optimal on N=20 random 3-regular graph...]")
    c = ctg.utils.randreg_equation(20, 3, seed=42)
    t = benchmark_optimal(c, minimize="flops")
    print(f"    Optimal (flops) time:      {t:.3e} s")
    t = benchmark_optimal(c, minimize="max")
    print(f"    Optimal (max)   time:      {t:.3e} s")
    t = benchmark_optimal(c, minimize="size")
    print(f"    Optimal (size)  time:      {t:.3e} s")
    t = benchmark_optimal(c, minimize="write")
    print(f"    Optimal (write) time:      {t:.3e} s")
    t = benchmark_optimal(c, minimize="combo")
    print(f"    Optimal (combo) time:      {t:.3e} s")
    t = benchmark_optimal(c, minimize="limit")
    print(f"    Optimal (limit) time:      {t:.3e} s")
