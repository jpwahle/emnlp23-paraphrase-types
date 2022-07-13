"""Usage: python test_significance.py <file1> <file2> <alpha>
Each file contains example-based scores (e.g., accuracy or BLEU) for a model.

Adjusted from https://github.com/rtmdrr/testSignificanceNLP"""

import sys

import numpy as np
from scipy import stats


def normality_check(data_a, data_b, name, alpha):
    """Performs a normality check on the difference between two datasets.

    Args:
        data_a: The first dataset.
        data_b: The second dataset.
        name (str): The name of the normality test to perform.
            Options are "Shapiro-Wilk", "Anderson-Darling", and "Kolmogorov-Smirnov".
        alpha: The significance level for the test.

    Returns:
        float: The p-value of the normality test.

    Example:
        ```python
        data_a = [1, 2, 3, 4, 5]
        data_b = [2, 3, 4, 5, 6]
        name = "Shapiro-Wilk"
        alpha = 0.05

        p_value = normality_check(data_a, data_b, name, alpha)
        print(p_value)
        ```
    """

    if name == "Shapiro-Wilk":
        # Shapiro-Wilk: Perform the Shapiro-Wilk test for normality.
        shapiro_results = stats.shapiro([a - b for a, b in zip(data_a, data_b)])
        return shapiro_results[1]

    elif name == "Anderson-Darling":
        # Anderson-Darling: Anderson-Darling test for data coming from a particular distribution
        anderson_results = stats.anderson(
            [a - b for a, b in zip(data_a, data_b)], "norm"
        )
        sig_level = 2
        if float(alpha) <= 0.01:
            sig_level = 4
        elif float(alpha) <= 0.025:
            sig_level = 3
        elif float(alpha) <= 0.05:
            sig_level = 2
        elif float(alpha) <= 0.1:
            sig_level = 1
        else:
            sig_level = 0

        return anderson_results[1][sig_level]

    else:
        # Kolmogorov-Smirnov: Perform the Kolmogorov-Smirnov test for goodness of fit.
        ks_results = stats.kstest([a - b for a, b in zip(data_a, data_b)], "norm")
        return ks_results[1]


def calculate_contingency(data_a, data_b, n):
    """Calculates the contingency table for two binary datasets.

    Args:
        data_a: The first binary dataset.
        data_b: The second binary dataset.
        n: The number of data points in the datasets.

    Returns:
        numpy.ndarray: The contingency table as a 2x2 numpy array.

    Example:
        ```python
        data_a = [1, 0, 1, 1, 0]
        data_b = [0, 1, 1, 0, 1]
        n = 5

        contingency_table = calculate_contingency(data_a, data_b, n)
        print(contingency_table)
        ```
    """

    ABrr = 0
    ABrw = 0
    ABwr = 0
    ABww = 0
    for i in range(n):
        if data_a[i] == 1 and data_b[i] == 1:
            ABrr = ABrr + 1
        if data_a[i] == 1 and data_b[i] == 0:
            ABrw = ABrw + 1
        if data_a[i] == 0 and data_b[i] == 1:
            ABwr = ABwr + 1
        else:
            ABww = ABww + 1
    return np.array([[ABrr, ABrw], [ABwr, ABww]])


def mc_nemar(table):
    """Calculates the Mc_nemar test statistic for a contingency table.

    Args:
        table: The contingency table as a 2x2 numpy array.

    Returns:
        float: The p-value of the Mc_nemar test.

    Example:
        ```python
        table = np.array([[10, 5], [3, 8]])

        p_value = mc_nemar(table)
        print(p_value)
        ```
    """
    statistic = float(np.abs(table[0][1] - table[1][0])) ** 2 / (
        table[1][0] + table[0][1]
    )
    return 1 - stats.chi2.cdf(statistic, 1)


def rand_permutation(data_a, data_b, n, R):
    """Performs a random permutation test to calculate the p-value.

    Args:
        data_a: The first dataset.
        data_b: The second dataset.
        n: The number of data points in the datasets.
        R: The number of permutations to perform.

    Returns:
        float: The p-value of the random permutation test.

    Example:
        ```python
        data_a = [1, 2, 3, 4, 5]
        data_b = [2, 3, 4, 5, 6]
        n = 5
        R = 1000

        p_value = rand_permutation(data_a, data_b, n, R)
        print(p_value)
        ```
    """
    delta_orig = float(sum(x - y for x, y in zip(data_a, data_b))) / n
    r = 0
    for _ in range(R):
        temp_a = data_a
        temp_b = data_b
        samples = [np.random.randint(1, 3) for _ in range(n)]
        swap_ind = [i for i, val in enumerate(samples) if val == 1]
        for ind in swap_ind:
            temp_b[ind], temp_a[ind] = temp_a[ind], temp_b[ind]
        delta = float(sum(x - y for x, y in zip(temp_a, temp_b))) / n
        if delta <= delta_orig:
            r = r + 1
    return float(r + 1.0) / (R + 1.0)


def bootstrap(data_a, data_b, n, R):
    """Performs a bootstrap test to calculate the p-value.

    Args:
        data_a: The first dataset.
        data_b: The second dataset.
        n: The number of data points in the datasets.
        R: The number of bootstrap iterations to perform.

    Returns:
        float: The p-value of the bootstrap test.

    Example:
        ```python
        data_a = [1, 2, 3, 4, 5]
        data_b = [2, 3, 4, 5, 6]
        n = 5
        R = 1000

        p_value = bootstrap(data_a, data_b, n, R)
        print(p_value)
        ```
    """
    delta_orig = float(sum(x - y for x, y in zip(data_a, data_b))) / n
    r = 0
    for _ in range(R):
        temp_a = []
        temp_b = []
        samples = np.random.randint(
            0, n, n
        )  # which samples to add to the subsample with repetitions
        for samp in samples:
            temp_a.append(data_a[samp])
            temp_b.append(data_b[samp])
        delta = float(sum(x - y for x, y in zip(temp_a, temp_b))) / n
        if delta > 2 * delta_orig:
            r = r + 1
    return float(r) / (R)


def main():
    """Executes the main function for performing statistical tests.

    This function reads data from files, performs statistical tests based on user input,
    and prints the test results.

    Args:
        None

    Returns:
        None

    Example:
        ```python
        main()
        ```
    """

    if len(sys.argv) < 3:
        print("You did not give enough arguments\n ")
        sys.exit(1)
    filename_a = sys.argv[1]
    filename_b = sys.argv[2]
    alpha = sys.argv[3]

    with open(filename_a, encoding="utf-8") as f:
        data_a = f.read().splitlines()

    with open(filename_b, encoding="utf-8") as f:
        data_b = f.read().splitlines()

    data_a = list(map(float, data_a))
    data_b = list(map(float, data_b))

    print(
        "\nPossible statistical tests: Shapiro-Wilk, Anderson-Darling,"
        " Kolmogorov-Smirnov, t-test, Wilcoxon, Mc_nemar, Permutation,"
        " Bootstrap"
    )
    name = input("\nEnter name of statistical test: ")

    # Normality Check
    if name in ["Shapiro-Wilk", "Anderson-Darling", "Kolmogorov-Smirnov"]:
        output = normality_check(data_a, data_b, name, alpha)

        if float(output) > float(alpha):
            answer = input(
                "\nThe normal test is significant, would you like to perform a"
                " t-test for checking significance of difference between"
                " results? (Y/N) "
            )
            if answer == "Y":
                # two sided t-test
                t_results = stats.ttest_rel(data_a, data_b)
                # correct for one sided test
                pval = t_results[1] / 2
                if float(pval) <= float(alpha):
                    print(f"\nTest result is significant with p-value: {pval}")
                else:
                    print("\nTest result is not significant with p-value:" f" {pval}")
                return
            else:
                answer2 = input(
                    "\nWould you like to perform a different test (permutation"
                    " or bootstrap)? If so enter name of test, otherwise type"
                    " 'N' "
                )
                if answer2 == "N":
                    print("\nbye-bye")
                    return
                else:
                    name = answer2
        else:
            answer = input(
                "\nThe normal test is not significant, would you like to"
                " perform a non-parametric test for checking significance of"
                " difference between results? (Y/N) "
            )
            if answer == "Y":
                answer2 = input("\nWhich test (Permutation or bootstrap)? ")
                name = answer2
            else:
                print("\nbye-bye")
                return

    # Statistical tests

    # Paired Student's t-test: Calculate the T-test on TWO RELATED samples of scores, a and b
    # for one sided test we multiply p-value by half
    if name == "t-test":
        t_results = stats.ttest_rel(data_a, data_b)
        # correct for one sided test
        pval = float(t_results[1]) / 2
        if float(pval) <= float(alpha):
            print(f"\nTest result is significant with p-value: {pval}")
        else:
            print(f"\nTest result is not significant with p-value: {pval}")
        return
    # Wilcoxon: Calculate the Wilcoxon signed-rank test.
    if name == "Wilcoxon":
        wilcoxon_results = stats.wilcoxon(data_a, data_b)
        if float(wilcoxon_results[1]) <= float(alpha):
            print(
                "\nTest result is significant with p-value:" f" {wilcoxon_results[1]}"
            )
        else:
            print(
                "\nTest result is not significant with p-value:"
                f" {wilcoxon_results[1]}"
            )
        return
    if name == "mc_nemar":
        print(
            "\nThis test requires the results to be binary : A[1, 0, 0, 1,"
            " ...], B[1, 0, 1, 1, ...] for success or failure on the i-th"
            " example."
        )
        f_obs = calculate_contingency(data_a, data_b, len(data_a))
        mc_nemar_results = mc_nemar(f_obs)
        if float(mc_nemar_results) <= float(alpha):
            print("\nTest result is significant with p-value:" f" {mc_nemar_results}")
        else:
            print(
                "\nTest result is not significant with p-value:" f" {mc_nemar_results}"
            )
        return
    if name == "Permutation":
        R = max(10000, int(len(data_a) * (1 / float(alpha))))
        pval = rand_permutation(data_a, data_b, len(data_a), R)
        if float(pval) <= float(alpha):
            print(f"\nTest result is significant with p-value: {pval}")
        else:
            print(f"\nTest result is not significant with p-value: {pval}")
        return
    if name == "Bootstrap":
        R = max(10000, int(len(data_a) * (1 / float(alpha))))
        pval = bootstrap(data_a, data_b, len(data_a), R)
        if float(pval) <= float(alpha):
            print(f"\nTest result is significant with p-value: {pval}")
        else:
            print(f"\nTest result is not significant with p-value: {pval}")
        return
    else:
        print("\nInvalid name of statistical test")
        sys.exit(1)


if __name__ == "__main__":
    main()
