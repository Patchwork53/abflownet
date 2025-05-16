import pandas as pd

def rank_top1_and_compute_mean(csv_path, method):


    df = pd.read_csv(csv_path)
    df = df.loc[df['cdr'] == 'H_CDR3', :]


    df["rank_E"]  = df.groupby(["structure", "cdr"])["E_total"].rank(method="first", ascending=True)
    df["rank_dG"] = df.groupby(["structure", "cdr"])["dG_gen"].rank(method="first", ascending=True)
    df["composite_rank"] = df["rank_E"] + df["rank_dG"]


    top1 = df.loc[
        df.groupby(["structure", "cdr"])["composite_rank"].idxmin(),
        ["structure", "cdr", "filename", "E_total", "dG_gen"],
    ]


    mean_E_total = top1["E_total"].mean()
    mean_dG_gen = top1["dG_gen"].mean()

    print(method)
    print(f"Mean CDR E_total : {mean_E_total:.3f}")
    print(f"Mean CDR-Ag ΔG : {mean_dG_gen:.3f}\n")


rank_top1_and_compute_mean("./energymetric.csv",method="AbFlowNet")
