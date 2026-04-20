# Data Audit Report

## Sanity Checks

| check | status | detail |
| --- | --- | --- |
| Duplicate dates removed from asset panel | PASS | 0 duplicate dates removed |
| No non-positive asset prices | PASS | No zero or negative prices in asset columns |
| Currency labels already USD | WARN | 4 non-USD labels flagged for review |
| FRED one-business-day lag applied | PASS | All FRED features are shifted one business-day row before signal use |

## Asset Inception Dates

| asset | first_valid_date | last_valid_date | observation_count | currency | asset_class |
| --- | --- | --- | --- | --- | --- |
| USDJPY | 1976-01-01 | 2026-03-19 | 13095 | JPY | FX |
| XAU | 1976-01-02 | 2026-03-19 | 12892 | USD | Commodities |
| COFFEE_FUT | 1979-12-31 | 2026-04-14 | 11471 | USD | Commodities |
| COTTON_FUT | 1979-12-31 | 2026-04-15 | 11646 | USD | Commodities |
| SILVER_FUT | 1979-12-31 | 2026-04-15 | 11929 | USD | Commodities |
| SOYBEAN_FUT | 1979-12-31 | 2026-04-15 | 11671 | USD | Commodities |
| WHEAT_SPOT | 1979-12-31 | 2026-04-14 | 11673 | USD | Commodities |
| NIKKEI225 | 1980-12-26 | 2026-04-14 | 11129 | USD | Equities |
| AUD | 1980-12-31 | 2026-04-15 | 11788 | USD | FX |
| CAD | 1980-12-31 | 2026-04-15 | 11816 | USD | FX |
| CHF_FRANC | 1980-12-31 | 2026-04-15 | 11816 | USD | FX |
| GBP_POUND | 1980-12-31 | 2026-04-15 | 11816 | USD | FX |
| CNY | 1981-01-02 | 2026-04-15 | 11535 | USD | FX |
| SHEKEL | 1981-04-13 | 2026-04-14 | 11741 | USD | FX |
| FTSE100 | 1983-12-30 | 2026-04-14 | 10700 | USD | Equities |
| COCOAINDEXSPOT | 1984-01-06 | 2026-04-14 | 10659 | USD | Commodities |
| COPPERSPOT | 1986-04-01 | 2026-04-14 | 10108 | USD | Commodities |
| LBUSTRUU | 1988-12-30 | 2026-03-18 | 9332 | USD | Fixed Income |
| SPXT | 1989-09-11 | 2026-03-18 | 9201 | USD | Equities |
| NATURALGAS | 1990-04-03 | 2026-04-15 | 9053 | USD | Commodities |
| TA-125_ISRAEL | 1991-12-31 | 2026-04-14 | 8420 | USD | Equities |
| BROAD_TIPS | 1998-03-31 | 2026-04-14 | 7036 | USD | Fixed Income |
| LBEATREU_EUROBONDAGG | 1998-07-31 | 2026-04-14 | 7042 | EUR | Fixed Income |
| EURO | 1998-12-31 | 2026-04-15 | 7120 | USD | FX |
| BCEE1T_EUROAREA | 2000-01-03 | 2026-04-14 | 6609 | EUR | Fixed Income |
| I02923JP_JAPAN_BOND | 2000-07-03 | 2026-04-14 | 6563 | JPY | Fixed Income |
| CSI300_CHINA | 2002-01-04 | 2026-04-14 | 5887 | USD | Equities |
| B3REITT | 2003-03-31 | 2026-03-18 | 5993 | USD | Real Estate |
| 0_5Y_TIPS | 2010-05-31 | 2026-04-14 | 3999 | USD | Fixed Income |
| BITCOIN | 2010-07-19 | 2026-03-19 | 5024 | USD | Crypto |
| BAIGTRUU_ASIACREDIT | 2014-05-22 | 2026-04-14 | 3093 | USD | Fixed Income |
| ETHEREUM | 2018-02-08 | 2026-04-15 | 2878 | USD | Crypto |

## Gap Summary

| asset | gap_count | missing_calendar_days | longest_gap_days | extended_gap_count_gt4d |
| --- | --- | --- | --- | --- |
| TA-125_ISRAEL | 1908 | 4104 | 6 | 116 |
| CSI300_CHINA | 1240 | 2980 | 17 | 95 |
| NIKKEI225 | 2532 | 5417 | 11 | 84 |
| COPPERSPOT | 2113 | 4516 | 6 | 80 |
| FTSE100 | 2237 | 4747 | 5 | 75 |
| CNY | 2353 | 5005 | 11 | 60 |
| COFFEE_FUT | 2471 | 5436 | 45 | 52 |
| XAU | 2638 | 5448 | 7 | 44 |
| LBEATREU_EUROBONDAGG | 1467 | 3078 | 5 | 40 |
| BCEE1T_EUROAREA | 1411 | 2990 | 13 | 29 |
| SILVER_FUT | 2437 | 4979 | 6 | 27 |
| NATURALGAS | 1947 | 4109 | 5 | 25 |
| COTTON_FUT | 2513 | 5262 | 8 | 24 |
| I02923JP_JAPAN_BOND | 1389 | 2854 | 6 | 14 |
| SOYBEAN_FUT | 2525 | 5237 | 5 | 3 |
| SPXT | 1995 | 4137 | 5 | 2 |
| WHEAT_SPOT | 2523 | 5234 | 5 | 2 |
| AUD | 2373 | 4754 | 5 | 1 |
| BITCOIN | 494 | 699 | 10 | 1 |
| BROAD_TIPS | 1534 | 3206 | 15 | 1 |
| LBUSTRUU | 2042 | 4261 | 5 | 1 |
| SHEKEL | 2349 | 4697 | 5 | 1 |
| 0_5Y_TIPS | 867 | 1799 | 4 | 0 |
| B3REITT | 1198 | 2396 | 3 | 0 |
| BAIGTRUU_ASIACREDIT | 626 | 1253 | 4 | 0 |
| CAD | 2363 | 4726 | 3 | 0 |
| CHF_FRANC | 2363 | 4726 | 3 | 0 |
| COCOAINDEXSPOT | 2308 | 4781 | 4 | 0 |
| ETHEREUM | 101 | 111 | 4 | 0 |
| EURO | 1424 | 2848 | 3 | 0 |
| GBP_POUND | 2363 | 4726 | 3 | 0 |
| USDJPY | 2621 | 5246 | 4 | 0 |

## Longest Observed Gaps

| asset | prev_valid_date | next_valid_date | calendar_gap_days | missing_calendar_days | gap_class |
| --- | --- | --- | --- | --- | --- |
| COFFEE_FUT | 1988-10-31 | 1988-12-15 | 45 | 44 | extended_gap |
| COFFEE_FUT | 1987-02-27 | 1987-03-23 | 24 | 23 | extended_gap |
| COFFEE_FUT | 1987-06-30 | 1987-07-23 | 23 | 22 | extended_gap |
| COFFEE_FUT | 1987-08-31 | 1987-09-22 | 22 | 21 | extended_gap |
| COFFEE_FUT | 1987-11-30 | 1987-12-22 | 22 | 21 | extended_gap |
| COFFEE_FUT | 1987-04-30 | 1987-05-20 | 20 | 19 | extended_gap |
| COFFEE_FUT | 1988-02-29 | 1988-03-17 | 17 | 16 | extended_gap |
| COFFEE_FUT | 1988-04-29 | 1988-05-16 | 17 | 16 | extended_gap |
| CSI300_CHINA | 2002-02-08 | 2002-02-25 | 17 | 16 | extended_gap |
| COFFEE_FUT | 1988-08-31 | 1988-09-16 | 16 | 15 | extended_gap |
| BROAD_TIPS | 1998-03-31 | 1998-04-15 | 15 | 14 | extended_gap |
| COFFEE_FUT | 1988-06-30 | 1988-07-15 | 15 | 14 | extended_gap |
| BCEE1T_EUROAREA | 2001-01-19 | 2001-02-01 | 13 | 12 | extended_gap |
| BCEE1T_EUROAREA | 2001-11-20 | 2001-12-03 | 13 | 12 | extended_gap |
| BCEE1T_EUROAREA | 2003-01-21 | 2003-02-03 | 13 | 12 | extended_gap |
| CSI300_CHINA | 2004-01-16 | 2004-01-29 | 13 | 12 | extended_gap |
| BCEE1T_EUROAREA | 2000-09-20 | 2000-10-02 | 12 | 11 | extended_gap |
| BCEE1T_EUROAREA | 2000-12-20 | 2001-01-01 | 12 | 11 | extended_gap |
| BCEE1T_EUROAREA | 2002-11-20 | 2002-12-02 | 12 | 11 | extended_gap |
| BCEE1T_EUROAREA | 2002-12-20 | 2003-01-01 | 12 | 11 | extended_gap |
| BCEE1T_EUROAREA | 2003-12-19 | 2003-12-31 | 12 | 11 | extended_gap |
| CSI300_CHINA | 2003-01-29 | 2003-02-10 | 12 | 11 | extended_gap |
| CSI300_CHINA | 2003-04-30 | 2003-05-12 | 12 | 11 | extended_gap |
| CSI300_CHINA | 2005-02-04 | 2005-02-16 | 12 | 11 | extended_gap |
| CSI300_CHINA | 2006-01-25 | 2006-02-06 | 12 | 11 | extended_gap |

## Currency Exceptions

| Column_Name | Full_Name | Currency | Asset_Class |
| --- | --- | --- | --- |
| USDJPY | US Dollar / Japanese Yen Spot | JPY | FX |
| BCEE1T_EUROAREA | Bloomberg Euro Aggregate Treasury 1-3Y Total Return Index | EUR | Fixed Income |
| I02923JP_JAPAN_BOND | Bloomberg Japan Treasury Bond Index | JPY | Fixed Income |
| LBEATREU_EUROBONDAGG | Bloomberg Euro Aggregate Bond Total Return Index | EUR | Fixed Income |

## FRED Publication-Lag Policy

- Source file: `data/consolidated_csvs/fred/master/fred_data.csv`
- Lag rule: every FRED feature is shifted by `1` business day before any signal can consume it.
- Alignment rule: after lagging on the FRED business-day calendar, values may be forward-filled onto the asset calendar because the last published macro observation remains the latest observable value until a newer release arrives.

| series | first_usable_date |
| --- | --- |
| BAMLC0A0CM | 2001-01-03 |
| BAMLH0A0HYM2 | 2001-01-03 |
| DFII10 | 2003-01-03 |
| NFCI | 2001-01-08 |
| T10Y2Y | 2001-01-03 |
| T10Y3M | 2001-01-03 |
| T10YIE | 2003-01-03 |
| UNRATE | 2001-02-02 |
| USRECD | 2001-01-03 |
| VIXCLS | 2001-01-03 |
