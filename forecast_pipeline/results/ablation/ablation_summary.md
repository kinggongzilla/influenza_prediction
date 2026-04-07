# Covariate Ablation Results

Generated: 2026-04-07 22:24
Training: 4000 steps, lr=5e-06, batch_size=32
Evaluation: eval_start=2025-10-01, horizon=4wk
Influcast season: 2025-26

## Rolling Evaluation (all countries)

| # | Covariates | rWIS | MAPE% | WIS | Cov50% | Cov95% | MAE | Countries |
|---|-----------|------|-------|-----|--------|--------|-----|-----------|
| 3 | hemisphere | 2.982 | 327621.2 | 1018.5 | 40.1 | 91.1 | 1739.92 | 86 |

## Influcast Evaluation (Italy)

| # | Covariates | Simple rWIS | MAPE% | Pairwise rWIS | Cov50% | Cov90% |
|---|-----------|------------|-------|--------------|--------|--------|
| 3 | hemisphere | 0.723 | 16.7 | 0.708 | 24.1 | 83.3 |

## Per-Country rWIS

Columns: #3=hemisphere

| Country | #3 |
|---------|------|
| Afghanistan | 0.932 |
| American Samoa | — |
| Argentina | 0.369 |
| Australia | 0.515 |
| Austria | 0.643 |
| Azerbaijan | 0.749 |
| Bangladesh | 0.568 |
| Belarus | 70.602 |
| Belgium | 0.777 |
| Bhutan | 0.914 |
| Brazil | 0.626 |
| Bulgaria | 0.496 |
| Cambodia | 0.903 |
| Cameroon | 0.775 |
| Canada | — |
| Chile | 0.554 |
| Colombia | 78.712 |
| Costa Rica | 1.056 |
| Croatia | 0.787 |
| Czechia | 1.358 |
| Côte d’Ivoire | 0.951 |
| Denmark | 0.915 |
| Estonia | 0.767 |
| Ethiopia | — |
| Fiji | 0.839 |
| Finland | 0.884 |
| France | 0.881 |
| French Polynesia | — |
| Georgia | 0.862 |
| Germany | 0.485 |
| Ghana | — |
| Greece | 0.702 |
| Guinea | 0.841 |
| Honduras | 0.642 |
| Hungary | 0.659 |
| Indonesia | 0.881 |
| Ireland | 0.546 |
| Israel | 0.513 |
| Italy | 0.476 |
| Jamaica | 0.947 |
| Kazakhstan | 0.671 |
| Kiribati | 0.759 |
| Kosovo (in accordance with UN Security Council resolution 1244 (1999)) | 1.250 |
| Lao People's Democratic Republic | 1.006 |
| Lebanon | 0.979 |
| Lithuania | 0.801 |
| Madagascar | 0.796 |
| Malta | 0.983 |
| Mexico | 3.605 |
| Micronesia (Federated States of) | 0.791 |
| Mongolia | — |
| Morocco | — |
| Nepal | 0.775 |
| Netherlands (Kingdom of the) | 0.994 |
| New Caledonia | 0.763 |
| New Zealand | — |
| Niger | 0.804 |
| Nigeria | 0.958 |
| North Macedonia | 0.894 |
| Northern Mariana Islands | 0.772 |
| Norway | 0.950 |
| Oman | 4.012 |
| Paraguay | 0.917 |
| Peru | 6.411 |
| Poland | 0.581 |
| Qatar | 0.730 |
| Russian Federation | 0.902 |
| Samoa | 0.821 |
| Saudi Arabia | 0.878 |
| Senegal | 1.548 |
| Serbia | 0.749 |
| Singapore | 1.088 |
| Slovakia | 0.661 |
| Solomon Islands | 0.882 |
| Somalia | 1.094 |
| Spain | 0.905 |
| Switzerland | 0.859 |
| Togo | 0.746 |
| Tonga | 0.890 |
| Türkiye | 0.653 |
| Ukraine | 10.553 |
| United States of America | 0.525 |
| Vanuatu | 0.885 |
| Viet Nam | 1.139 |
| Wallis and Futuna | 0.797 |
| Zambia | 0.705 |

## Rolling Eval Per-Horizon rWIS

| # | Covariates | H1 | H2 | H3 | H4 |
|---|-----------|----|----|----|----|
| 3 | hemisphere | 2.438 | 2.185 | 2.010 | 2.890 |

## Influcast Per-Horizon Simple rWIS

| # | Covariates | H1 | H2 | H3 | H4 |
|---|-----------|----|----|----|----|
| 3 | hemisphere | 1.072 | 0.803 | 0.658 | 0.575 |

## Influcast Per-Horizon MAPE%

| # | Covariates | H1 | H2 | H3 | H4 |
|---|-----------|----|----|----|----|
| 3 | hemisphere | 14.2 | 15.9 | 17.8 | 19.7 |

## Best Configurations

- **Best rolling rWIS:** #3 (hemisphere) = 2.982
- **Best rolling MAPE:** #3 (hemisphere) = 327621.2%
- **Best Influcast rWIS:** #3 (hemisphere) = 0.723
- **Best Influcast MAPE:** #3 (hemisphere) = 16.7%
- **Best 95% coverage:** #3 (hemisphere) = 91.1%

## Individual Covariate Impact

Average rWIS improvement when adding each covariate (across all combinations):

