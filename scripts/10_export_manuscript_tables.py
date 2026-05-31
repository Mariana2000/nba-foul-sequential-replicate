"""Export manuscript Tables 2--4 from analysis CSV outputs."""

from whistle_balance.manuscript_tables import (
    table2_latex,
    table2_markdown,
    table2_word_csv,
    table3_latex,
    table3_markdown,
    table3_word_csv,
    table4_latex,
    table4_markdown,
    table4_word_csv,
    table5_latex,
    table5_markdown,
    table5_word_csv,
    table6_latex,
    table6_markdown,
    table6_word_csv,
)


def main() -> None:
    table2_word_csv()
    table2_latex()
    table2_markdown()
    table3_word_csv()
    table3_latex()
    table3_markdown()
    table4_word_csv()
    table4_latex()
    table4_markdown()
    table5_word_csv()
    table5_latex()
    table5_markdown()
    table6_word_csv()
    table6_latex()
    table6_markdown()
    print("Wrote Table 2: docs/tables/table2_main_regression.*")
    print("Wrote Table 3: docs/tables/table3_game_fe.*")
    print("Wrote Table 4: docs/tables/table4_marginal_effects.*")
    print("Wrote Table 5: docs/tables/table5_placebo.*")
    print("Wrote Table 6: docs/tables/table6_heterogeneity.*")
    print("Wrote Word CSVs under outputs/tables/")


if __name__ == "__main__":
    main()
