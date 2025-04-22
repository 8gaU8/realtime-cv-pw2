from pathlib import Path
import subprocess

import pandas as pd


def get_ips(cmd: list[str]) -> float:
    print("Executing...", " ".join(cmd))
    stdout = subprocess.run(cmd, capture_output=True, text=True).stdout
    print(stdout)

    ips_str = stdout.splitlines()[-1][5:]
    ips = float(ips_str)
    return ips


def main():
    # consts
    impath = "./assets/sf.png"
    anaglyph_method = "true"
    sigma = "1"
    gaussian_factor_ratio = "1"

    # common parmas
    kernel_size_grid = ["1", "4", "8", "16"]
    neighbor_size_div2_grid = ["1", "4", "8", "16"]

    # omp params
    nb_threads_grid = [str(i) for i in range(2, 65, 2)]
    nb_threads_grid.append("-1")

    # cuda params
    blockDimX_grid = ["1", "2", "4", "8", "16", "32", "64"]
    blockDimY_grid = ["1", "2", "4", "8", "16", "32", "64"]

    # benchmark files
    csv_root = Path("./csvs")
    if not csv_root.exists():
        csv_root.mkdir(parents=True)

    # omp:
    ## ex1-1-anaglyph
    csv_path = csv_root / "ex1-1-anaglyph.csv"
    print(csv_path)
    if not csv_path.exists():
        nb_threads_results = []
        ips_results = []
        for nb_threads in nb_threads_grid:
            cmd = [
                "./pw2-omp/ex1-1-anaglyph",
                impath,
                anaglyph_method,
                nb_threads,
            ]
            ips = get_ips(cmd)
            nb_threads_results.append(nb_threads)
            ips_results.append(ips)
        ex11_df = pd.DataFrame(
            {
                "nb_threads": nb_threads_results,
                "ips": ips_results,
            }
        )
        ex11_df.to_csv(csv_path, index=False)

    ## ex1-2-gaussian
    csv_path = csv_root / "ex1-2-gaussian.csv"
    print(csv_path)
    if not csv_path.exists():
        kernel_size_results = []
        nb_threads_results = []
        ips_results = []
        for kernel_size in kernel_size_grid:
            for nb_threads in nb_threads_grid:
                cmd = [
                    "./pw2-omp/ex1-2-gaussian",
                    impath,
                    anaglyph_method,
                    kernel_size,
                    sigma,
                    nb_threads,
                ]
                ips = get_ips(cmd)

                # store results to list
                kernel_size_results.append(kernel_size)
                nb_threads_results.append(nb_threads)
                ips_results.append(ips)

        ex12_df = pd.DataFrame(
            {
                "kernel_size": kernel_size_results,
                "nb_threads": nb_threads_results,
                "ips": ips_results,
            }
        )
        ex12_df.to_csv(csv_path, index=False)

    ## ex1-3-denoising
    csv_path = csv_root / "ex1-3-denoising.csv"
    print(csv_path)
    if not csv_path.exists():
        neighbor_size_results = []
        nb_threads_results = []
        ips_results = []
        for nb_threads in nb_threads_grid:
            for neighbor_size_div_2 in neighbor_size_div2_grid:
                cmd = [
                    "./pw2-omp/ex1-3-denoising",
                    impath,
                    anaglyph_method,
                    neighbor_size_div_2,
                    gaussian_factor_ratio,
                    sigma,
                    nb_threads,
                ]
                ips = get_ips(cmd)
                neighbor_size_results.append(neighbor_size_div_2)
                nb_threads_results.append(nb_threads)
                ips_results.append(ips)

        ex12_df = pd.DataFrame(
            {
                "neighbor_size": neighbor_size_results,
                "nb_threads": nb_threads_results,
                "ips": ips_results,
            }
        )
        ex12_df.to_csv(csv_path, index=False)

    # =============================================================

    # cuda:
    ## ex2-1-anaglyph
    csv_path = csv_root / "ex2-1-anaglyph.csv"
    print(csv_path)
    if not csv_path.exists():
        blockDimX_results = []
        blockDimY_results = []

        ips_results = []
        for blockDimX in blockDimX_grid:
            for blockDimY in blockDimY_grid:
                cmd = [
                    "./pw2-cuda/ex2-1-anaglyph",
                    impath,
                    anaglyph_method,
                    blockDimX,
                    blockDimY,
                ]
                ips = get_ips(cmd)
                blockDimX_results.append(blockDimX)
                blockDimY_results.append(blockDimY)
                ips_results.append(ips)
        ex21_df = pd.DataFrame(
            {
                "blockDimX": blockDimX_results,
                "blockDimY": blockDimY_results,
                "ips": ips_results,
            }
        )
        ex21_df.to_csv(csv_path, index=False)

    ## ex2-2-gaussian
    csv_path = csv_root / "ex2-2-gaussian.csv"
    print(csv_path)
    if not csv_path.exists():
        kernel_size_results = []
        blockDimX_results = []
        blockDimY_results = []

        ips_results = []
        for kernel_size in kernel_size_grid:
            for blockDimX in blockDimX_grid:
                for blockDimY in blockDimY_grid:
                    cmd = [
                        "./pw2-cuda/ex2-2-gaussian",
                        impath,
                        anaglyph_method,
                        kernel_size,
                        sigma,
                        blockDimX,
                        blockDimY,
                    ]
                    ips = get_ips(cmd)

                    # store results to list
                    kernel_size_results.append(kernel_size)
                    blockDimX_results.append(blockDimX)
                    blockDimY_results.append(blockDimY)
                    ips_results.append(ips)

        ex22_df = pd.DataFrame(
            {
                "kernel_size": kernel_size_results,
                "blockDimX": blockDimX_results,
                "blockDimY": blockDimY_results,
                "ips": ips_results,
            }
        )
        ex22_df.to_csv(csv_path, index=False)

    ## ex2-3-denoising
    csv_path = csv_root / "ex2-3-denoising.csv"
    print(csv_path)
    if not csv_path.exists():
        neighbor_size_results = []
        blockDimX_results = []
        blockDimY_results = []

        ips_results = []
        for blockDimX in blockDimX_grid:
            for blockDimY in blockDimY_grid:
                for neighbor_size_div_2 in neighbor_size_div2_grid:
                    cmd = [
                        "./pw2-cuda/ex2-3-denoising",
                        impath,
                        anaglyph_method,
                        neighbor_size_div_2,
                        gaussian_factor_ratio,
                        sigma,
                        blockDimX,
                        blockDimY,
                    ]
                    ips = get_ips(cmd)
                    neighbor_size_results.append(neighbor_size_div_2)
                    blockDimX_results.append(blockDimX)
                    blockDimY_results.append(blockDimY)
                    ips_results.append(ips)

        ex22_df = pd.DataFrame(
            {
                "neighbor_size": neighbor_size_results,
                "blockDimX": blockDimX_results,
                "blockDimY": blockDimY_results,
                "ips": ips_results,
            }
        )
        ex22_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
