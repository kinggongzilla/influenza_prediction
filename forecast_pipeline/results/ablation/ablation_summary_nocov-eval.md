# Covariate Ablation Results (No-Covariate Evaluation)

Generated: 2026-04-07 22:26
Training: 4000 steps, lr=5e-06, batch_size=32
Evaluation: eval_start=2025-10-01, horizon=4wk
Influcast season: 2025-26

## Rolling Evaluation (all countries)

| # | Covariates | rWIS | MAPE% | WIS | Cov50% | Cov95% | MAE | Countries |
|---|-----------|------|-------|-----|--------|--------|-----|-----------|
| 3 | hemisphere | 3.159 | 288890.6 | 1065.7 | 39.6 | 91.7 | 1826.74 | 86 |

## Influcast Evaluation (Italy)

| # | Covariates | Simple rWIS | MAPE% | Pairwise rWIS | Cov50% | Cov90% |
|---|-----------|------------|-------|--------------|--------|--------|
| 3 | hemisphere | 0.733 | 16.9 | 0.718 | 25.9 | 85.2 |

## Per-Country rWIS

Columns: #3=hemisphere

| Country | #3 |
|---------|------|
| Afghanistan | 1.051 |
| American Samoa | — |
| Argentina | 0.350 |
| Australia | 0.518 |
| Austria | 0.651 |
| Azerbaijan | 0.761 |
| Bangladesh | 0.579 |
| Belarus | 71.433 |
| Belgium | 0.795 |
| Bhutan | 0.901 |
| Brazil | 0.718 |
| Bulgaria | 0.569 |
| Cambodia | 0.954 |
| Cameroon | 0.814 |
| Canada | — |
| Chile | 0.559 |
| Colombia | 91.110 |
| Costa Rica | 1.023 |
| Croatia | 0.807 |
| Czechia | 1.396 |
| Côte d’Ivoire | 0.952 |
| Denmark | 0.930 |
| Estonia | 0.827 |
| Ethiopia | — |
| Fiji | 0.835 |
| Finland | 0.901 |
| France | 0.925 |
| French Polynesia | — |
| Georgia | 0.894 |
| Germany | 0.538 |
| Ghana | — |
| Greece | 0.717 |
| Guinea | 0.874 |
| Honduras | 0.594 |
| Hungary | 0.714 |
| Indonesia | 0.862 |
| Ireland | 0.544 |
| Israel | 0.542 |
| Italy | 0.542 |
| Jamaica | 0.913 |
| Kazakhstan | 0.681 |
| Kiribati | 0.766 |
| Kosovo (in accordance with UN Security Council resolution 1244 (1999)) | 1.308 |
| Lao People's Democratic Republic | 1.032 |
| Lebanon | 1.025 |
| Lithuania | 0.821 |
| Madagascar | 0.830 |
| Malta | 0.980 |
| Mexico | 3.567 |
| Micronesia (Federated States of) | 0.799 |
| Mongolia | — |
| Morocco | — |
| Nepal | 0.758 |
| Netherlands (Kingdom of the) | 0.991 |
| New Caledonia | 0.784 |
| New Zealand | — |
| Niger | 0.849 |
| Nigeria | 0.974 |
| North Macedonia | 0.972 |
| Northern Mariana Islands | 0.779 |
| Norway | 1.021 |
| Oman | 4.479 |
| Paraguay | 0.971 |
| Peru | 6.303 |
| Poland | 0.552 |
| Qatar | 0.722 |
| Russian Federation | 0.821 |
| Samoa | 0.846 |
| Saudi Arabia | 0.886 |
| Senegal | 1.643 |
| Serbia | 0.777 |
| Singapore | 1.018 |
| Slovakia | 0.707 |
| Solomon Islands | 0.927 |
| Somalia | 1.047 |
| Spain | 0.974 |
| Switzerland | 0.877 |
| Togo | 0.693 |
| Tonga | 0.918 |
| Türkiye | 0.696 |
| Ukraine | 9.521 |
| United States of America | 0.534 |
| Vanuatu | 0.866 |
| Viet Nam | 1.092 |
| Wallis and Futuna | 0.793 |
| Zambia | 0.709 |

## Rolling Eval Per-Horizon rWIS

| # | Covariates | H1 | H2 | H3 | H4 |
|---|-----------|----|----|----|----|
| 3 | hemisphere | 2.701 | 2.372 | 2.167 | 3.039 |

## Influcast Per-Horizon Simple rWIS

| # | Covariates | H1 | H2 | H3 | H4 |
|---|-----------|----|----|----|----|
| 3 | hemisphere | 1.062 | 0.814 | 0.673 | 0.589 |

## Influcast Per-Horizon MAPE%

| # | Covariates | H1 | H2 | H3 | H4 |
|---|-----------|----|----|----|----|
| 3 | hemisphere | 13.9 | 16.2 | 18.2 | 20.2 |

## Best Configurations

- **Best rolling rWIS:** #3 (hemisphere) = 3.159
- **Best rolling MAPE:** #3 (hemisphere) = 288890.6%
- **Best Influcast rWIS:** #3 (hemisphere) = 0.733
- **Best Influcast MAPE:** #3 (hemisphere) = 16.9%
- **Best 95% coverage:** #3 (hemisphere) = 91.7%

## Individual Covariate Impact

Average rWIS improvement when adding each covariate (across all combinations):

