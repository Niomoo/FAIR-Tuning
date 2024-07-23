import pandas as pd
import numpy as np
import os
import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        default="./significance_test_results/",
        help="Path to the test results",
    )
    if input_args:
        return parser.parse_args(input_args)
    return parser.parse_args()

def significant(value, s_null, s_alter):
    try:
        if pd.isna(value):
            return value
        elif value < 0.05:
            return s_alter
        else:
            return s_null
    except TypeError:
        return value

def get_significance_test_results(df, s_null, s_alter):
    result_df = pd.DataFrame()
    fold_index = df.columns.get_loc("fold")
    for column in df.columns[fold_index + 1 :]:
        result_df[column] = df[column].apply(significant, args=(s_null, s_alter))
    return result_df

def get_comparison_results(df_before, df_after, cancer, s_null, s_alter):
    df_comparison = pd.DataFrame(columns=df_before.columns, index=[cancer])
    count = {"improved": 0, "degraded": 0, "no change": 0, "--": 0}
    for column in df_before.columns:
        if column != "Unnamed: 0" and column != "fold":
            for index, row in df_before.iterrows():
                if row[column] == s_alter and df_after.iloc[-1][column] == s_null:
                    df_comparison.loc[cancer, column] = "improved"
                    count["improved"] += 1
                elif row[column] == s_null and df_after.iloc[-1][column] == s_alter:
                    df_comparison.loc[cancer, column] = "degraded"
                    count["degraded"] += 1
                elif row[column] == s_alter and df_after.iloc[-1][column] == s_alter:
                    df_comparison.loc[cancer, column] = "no change"
                    count["no change"] += 1
                else:
                    df_comparison.loc[cancer, column] = "--"
                    count["--"] += 1
    return df_comparison, count

def main(args):
    summary_1 = pd.DataFrame()
    summary_2 = pd.DataFrame()
    for sens_folder in os.listdir(args.results_path):
        for cancer_folder in os.listdir(args.results_path + sens_folder):
            if cancer_folder == "summary_1.csv" or cancer_folder == "summary_2.csv":
                continue
            df1 = pd.read_csv(
                args.results_path + sens_folder + "/" + cancer_folder + "/bias_baseline.csv"
            )
            df2 = pd.read_csv(
                args.results_path
                + sens_folder
                + "/"
                + cancer_folder
                + "/bias_corrected.csv"
            )
            df3 = pd.read_csv(
                args.results_path + sens_folder + "/" + cancer_folder + "/improvement.csv"
            )

            df1_significance = get_significance_test_results(df1, "fair", "biased")
            df2_significance = get_significance_test_results(df2, "fair", "biased")
            df3_significance = get_significance_test_results(
                df3, "no significant", "significant"
            )
            # print(df1_significance)
            # print(df2_significance)
            df_before = df1_significance.iloc[-1:]
            df_after = df2_significance.iloc[-1:]
            df_improvements = df3_significance.iloc[-2:]

            # print("\n\n---Method 1---")
            comparison_results, count = get_comparison_results(
                df_before, df_after, cancer_folder, "fair", "biased"
            )
            # print(comparison_results)
            print("\n", cancer_folder)
            print("Improved: ", count["improved"])
            print("Degraded: ", count["degraded"])
            print("No change: ", count["no change"])
            print("No Fairness Issue: ", count["--"])
            summary_1 = pd.concat([summary_1, comparison_results])

            # print("---Method 2---")
            df_improvements.index = [f"{cancer_folder}_0", f"{cancer_folder}_1"]
            # print(df_improvements)
            summary_2 = pd.concat([summary_2, df_improvements])

        summary_1.to_csv(args.results_path + sens_folder + "/summary_1.csv")
        summary_2.to_csv(args.results_path + sens_folder + "/summary_2.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
