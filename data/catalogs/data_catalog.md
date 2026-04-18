# Data Catalog

- Consolidated CSV files cataloged: 79
- Asset data CSV files cataloged: 2
- Asset panel columns cataloged: 41
- Portfolio start date satisfying hard availability constraints: 2000-01-03

## File Inventory

| source_group | dataset_name | start_date | end_date | frequency | measurement | sector | relative_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Asset Data | Asset Data Key | N/A | N/A | Metadata | Metadata table | Metadata | data/asset_data/data_key.csv |
| Asset Data | Whitmore Daily Asset Panel | 1976-01-01 | 2026-04-15 | Daily | Price / index level, plus *_VOL trading volume columns | Multi-Asset | data/asset_data/whitmore_daily.csv |
| Bloomberg | Bloomberg Data Manifest | 1990-01-02 | 2026-04-07 | Metadata | Metadata table | Metadata | data/consolidated_csvs/bloomberg/master/_data_manifest_bloomberg.csv |
| Bloomberg | Bloomberg Data Quality Report | 1990-01-02 | N/A | Metadata | Metadata table | Metadata | data/consolidated_csvs/bloomberg/master/_data_quality_bloomberg.csv |
| Bloomberg | master_logrets_bloomberg | 1990-01-03 | 2026-04-07 | Daily | Daily log return | Multi-Asset | data/consolidated_csvs/bloomberg/master/master_logrets_bloomberg.csv |
| Bloomberg | master_prices_bloomberg | 1990-01-02 | 2026-04-07 | Daily | Price / index level | Multi-Asset | data/consolidated_csvs/bloomberg/master/master_prices_bloomberg.csv |
| Bloomberg | taa_log_returns | 2001-01-03 | 2025-12-31 | Daily | Daily log return | Multi-Asset | data/consolidated_csvs/bloomberg/master/taa_log_returns.csv |
| Bloomberg | taa_prices | 2001-01-02 | 2025-12-31 | Daily | Price / index level | Multi-Asset | data/consolidated_csvs/bloomberg/master/taa_prices.csv |
| FRED | FRED Master Table | 2001-01-02 | 2025-12-31 | Daily | Mixed levels, rates, spreads, and indicators | Multi-Sector | data/consolidated_csvs/fred/master/fred_data.csv |
| FRED | FRED Pull Log | 1900-01-01 | 2026-04-03 | Per pull | Metadata log | Metadata | data/consolidated_csvs/fred/raw/_pull_log.csv |
| FRED | Brent Crude Oil Price | 1987-05-20 | 2026-03-30 | Daily | Price level | Commodity | data/consolidated_csvs/fred/raw/commodity_Brent_Crude_Oil_Price.csv |
| FRED | Copper Price USD | 1992-01-01 | 2026-02-01 | Monthly | Price level | Commodity | data/consolidated_csvs/fred/raw/commodity_Copper_Price_USD.csv |
| FRED | WTI Crude Oil Price | 1986-01-02 | 2026-03-30 | Daily | Price level | Commodity | data/consolidated_csvs/fred/raw/commodity_WTI_Crude_Oil_Price.csv |
| FRED | Chicago Fed National Financial Conditions Index | 1971-01-08 | 2026-03-27 | Weekly | Level | Credit | data/consolidated_csvs/fred/raw/credit_Chicago_Fed_NFCI.csv |
| FRED | Chicago Fed NFCI Credit Subindex | 1971-01-08 | 2026-03-27 | Weekly | Index / level | Credit | data/consolidated_csvs/fred/raw/credit_Chicago_Fed_NFCI_Credit_Subindex.csv |
| FRED | ICE BofA HY Effective Yield | 1996-12-31 | 2026-04-03 | Daily | Percent / rate | Credit | data/consolidated_csvs/fred/raw/credit_ICE_BofA_HY_Effective_Yield.csv |
| FRED | ICE BofA HY OAS | 1996-12-31 | 2026-04-03 | Daily | Spread / percentage points | Credit | data/consolidated_csvs/fred/raw/credit_ICE_BofA_HY_OAS.csv |
| FRED | ICE BofA IG Effective Yield | 1996-12-31 | 2026-04-03 | Daily | Percent / rate | Credit | data/consolidated_csvs/fred/raw/credit_ICE_BofA_IG_Effective_Yield.csv |
| FRED | ICE BofA IG OAS | 1996-12-31 | 2026-04-03 | Daily | Spread / percentage points | Credit | data/consolidated_csvs/fred/raw/credit_ICE_BofA_IG_OAS.csv |
| FRED | S&P 500 Index Level | 2016-04-04 | 2026-04-02 | Daily | Price / index level | Equity | data/consolidated_csvs/fred/raw/equity_SP500_Index_Level.csv |
| FRED | VIX Close | 1990-01-02 | 2026-04-02 | Daily | Price / index level | Equity | data/consolidated_csvs/fred/raw/equity_VIX_Close.csv |
| FRED | JPY / USD Exchange Rate | 1971-01-04 | 2026-03-27 | Daily | FX spot rate | FX | data/consolidated_csvs/fred/raw/fx_JPY_USD_Exchange_Rate.csv |
| FRED | USD / AUD Exchange Rate | 1971-01-04 | 2026-03-27 | Daily | FX spot rate | FX | data/consolidated_csvs/fred/raw/fx_USD_AUD_Exchange_Rate.csv |
| FRED | USD Broad Trade-Weighted Index | 2006-01-02 | 2026-03-27 | Daily | FX spot rate | FX | data/consolidated_csvs/fred/raw/fx_USD_Broad_Trade_Weighted_Index.csv |
| FRED | USD / EUR Exchange Rate | 1999-01-04 | 2026-03-27 | Daily | FX spot rate | FX | data/consolidated_csvs/fred/raw/fx_USD_EUR_Exchange_Rate.csv |
| FRED | Bank Prime Loan Rate | 1955-08-04 | 2026-03-31 | Daily | Percent / rate | Liquidity | data/consolidated_csvs/fred/raw/liquidity_Bank_Prime_Loan_Rate.csv |
| FRED | Federal Reserve Discount Window Credit | 2003-01-09 | 2026-04-02 | Daily | Level | Liquidity | data/consolidated_csvs/fred/raw/liquidity_Fed_Discount_Window_Credit.csv |
| FRED | Monetary Base | 1959-01-01 | 2026-02-01 | Monthly | Level | Liquidity | data/consolidated_csvs/fred/raw/liquidity_Monetary_Base.csv |
| FRED | Reserve Balances at the Federal Reserve | 2002-12-18 | 2026-04-01 | Weekly | Level | Liquidity | data/consolidated_csvs/fred/raw/liquidity_Reserve_Balances_at_Fed.csv |
| FRED | Building Permits | 1960-01-01 | 2026-01-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_Building_Permits.csv |
| FRED | CPI All Urban Consumers | 1947-01-01 | 2026-02-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_CPI_All_Urban_Consumers.csv |
| FRED | Capacity Utilization | 1967-01-01 | 2026-02-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_Capacity_Utilization.csv |
| FRED | Case Shiller Home Price Index | 1987-01-01 | 2026-01-01 | Monthly | Price level | Macro | data/consolidated_csvs/fred/raw/macro_Case_Shiller_Home_Price_Index.csv |
| FRED | Conference Board Leading Economic Index | 1982-01-01 | 2020-02-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_Conference_Board_LEI.csv |
| FRED | Continuing Jobless Claims | 1967-01-07 | 2026-03-21 | Weekly | Level | Macro | data/consolidated_csvs/fred/raw/macro_Continuing_Jobless_Claims.csv |
| FRED | Core CPI Ex Food Energy | 1957-01-01 | 2026-02-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_Core_CPI_ex_Food_Energy.csv |
| FRED | Core PCE Price Index | 1959-01-01 | 2026-01-01 | Monthly | Price level | Macro | data/consolidated_csvs/fred/raw/macro_Core_PCE_Price_Index.csv |
| FRED | Housing Starts | 1959-01-01 | 2026-01-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_Housing_Starts.csv |
| FRED | Industrial Production Index | 1919-01-01 | 2026-02-01 | Monthly | Index / level | Macro | data/consolidated_csvs/fred/raw/macro_Industrial_Production_Index.csv |
| FRED | Initial Jobless Claims Weekly | 1967-01-07 | 2026-03-28 | Weekly | Level | Macro | data/consolidated_csvs/fred/raw/macro_Initial_Jobless_Claims_Weekly.csv |
| FRED | LEI | 1982-01-01 | 2020-02-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_LEI.csv |
| FRED | Labor Force Participation Rate | 1948-01-01 | 2026-03-01 | Monthly | Percent / rate | Macro | data/consolidated_csvs/fred/raw/macro_Labour_Force_Participation_Rate.csv |
| FRED | M2 Money Supply | 1959-01-01 | 2026-02-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_M2_Money_Supply.csv |
| FRED | M2 Velocity | 1959-01-01 | 2025-10-01 | Quarterly | Level | Macro | data/consolidated_csvs/fred/raw/macro_M2_Velocity.csv |
| FRED | NFCI | 1971-01-08 | 2026-03-27 | Weekly | Level | Macro | data/consolidated_csvs/fred/raw/macro_NFCI.csv |
| FRED | Nonfarm Payrolls | 1939-01-01 | 2026-03-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_Nonfarm_Payrolls.csv |
| FRED | OECD Composite Leading Indicator - USA | 1955-01-01 | 2024-01-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_OECD_CLI_USA.csv |
| FRED | OECD Consumer Confidence - USA | 1960-01-01 | 2024-01-01 | Monthly | Index / level | Macro | data/consolidated_csvs/fred/raw/macro_OECD_Consumer_Confidence.csv |
| FRED | PCE Price Index | 1959-01-01 | 2026-01-01 | Monthly | Price level | Macro | data/consolidated_csvs/fred/raw/macro_PCE_Price_Index.csv |
| FRED | Personal Consumption Expenditures | 1959-01-01 | 2026-01-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_Personal_Consumption_Expenditures.csv |
| FRED | Retail Sales | 1992-01-01 | 2026-02-01 | Monthly | Level | Macro | data/consolidated_csvs/fred/raw/macro_Retail_Sales.csv |
| FRED | University of Michigan Consumer Sentiment | 1952-11-01 | 2026-02-01 | Quarterly | Index / level | Macro | data/consolidated_csvs/fred/raw/macro_UMich_Consumer_Sentiment.csv |
| FRED | US Real GDP Growth QoQ Annualized | 1947-04-01 | 2025-10-01 | Quarterly | Level | Macro | data/consolidated_csvs/fred/raw/macro_US_GDP_Growth_QoQ_Annualised.csv |
| FRED | US Real GDP Level | 1947-01-01 | 2025-10-01 | Quarterly | Level | Macro | data/consolidated_csvs/fred/raw/macro_US_Real_GDP_Level.csv |
| FRED | Unemployment Rate | 1948-01-01 | 2026-03-01 | Monthly | Percent / rate | Macro | data/consolidated_csvs/fred/raw/macro_Unemployment_Rate.csv |
| FRED | VIX | 1990-01-02 | 2026-04-02 | Daily | Level | Macro | data/consolidated_csvs/fred/raw/macro_VIX.csv |
| FRED | Unemployment | 1948-01-01 | 2026-03-01 | Monthly | Percent / rate | Macro | data/consolidated_csvs/fred/raw/macro_unemployment.csv |
| FRED | Yield Curve 10Y2Y | 1976-06-01 | 2026-04-06 | Daily | Percent / rate | Macro | data/consolidated_csvs/fred/raw/macro_yield_curve_10y2y.csv |
| FRED | Yield Curve 10Y3M | 1982-01-04 | 2026-04-06 | Daily | Percent / rate | Macro | data/consolidated_csvs/fred/raw/macro_yield_curve_10y3m.csv |
| FRED | 10Y Breakeven Inflation | 2003-01-02 | 2026-04-03 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_10Y_Breakeven_Inflation.csv |
| FRED | 10Y TIPS Real Yield | 2003-01-02 | 2026-04-02 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_10Y_TIPS_Real_Yield.csv |
| FRED | 5Y5Y Forward Inflation Breakeven | 2003-01-02 | 2026-04-03 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_5Y5Y_Forward_Inflation_Breakeven.csv |
| FRED | 5Y Breakeven Inflation | 2003-01-02 | 2026-04-03 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_5Y_Breakeven_Inflation.csv |
| FRED | 5Y TIPS Real Yield | 2003-01-02 | 2026-04-02 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_5Y_TIPS_Real_Yield.csv |
| FRED | Fed Funds Effective Rate Daily | 1954-07-01 | 2026-04-02 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_Fed_Funds_Effective_Rate_Daily.csv |
| FRED | Fed Funds Effective Rate Monthly | 1954-07-01 | 2026-03-01 | Monthly | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_Fed_Funds_Effective_Rate_Monthly.csv |
| FRED | Germany 10Y Bund Yield | 1956-05-01 | 2026-01-01 | Monthly | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_Germany_10Y_Bund_Yield.csv |
| FRED | SOFR | 2018-04-03 | 2026-04-02 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_SOFR.csv |
| FRED | TED Spread Historical | 1986-01-02 | 2022-01-21 | Daily | Spread / percentage points | Rates | data/consolidated_csvs/fred/raw/rates_TED_Spread_Historical.csv |
| FRED | US 10Y Treasury Yield | 1962-01-02 | 2026-04-02 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_US_10Y_Treasury_Yield.csv |
| FRED | US 1Y Treasury Yield | 1962-01-02 | 2026-04-02 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_US_1Y_Treasury_Yield.csv |
| FRED | US 2Y Treasury Yield | 1976-06-01 | 2026-04-02 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_US_2Y_Treasury_Yield.csv |
| FRED | US 30Y Treasury Yield | 1977-02-15 | 2026-04-02 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_US_30Y_Treasury_Yield.csv |
| FRED | US 3M Treasury Yield | 1981-09-01 | 2026-04-02 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_US_3M_Treasury_Yield.csv |
| FRED | US 5Y Treasury Yield | 1962-01-02 | 2026-04-02 | Daily | Percent / rate | Rates | data/consolidated_csvs/fred/raw/rates_US_5Y_Treasury_Yield.csv |
| FRED | Yieldcurve 10Y Minus 2Y Spread | 1976-06-01 | 2026-04-03 | Daily | Spread / percentage points | Rates | data/consolidated_csvs/fred/raw/rates_YieldCurve_10Y_minus_2Y_Spread.csv |
| FRED | Yieldcurve 10Y Minus 3M Spread | 1982-01-04 | 2026-04-03 | Daily | Spread / percentage points | Rates | data/consolidated_csvs/fred/raw/rates_YieldCurve_10Y_minus_3M_Spread.csv |
| FRED | Hamilton Recession Probability | 1967-10-01 | 2025-07-01 | Quarterly | Probability | Recession | data/consolidated_csvs/fred/raw/recession_Hamilton_Recession_Probability.csv |
| FRED | NBER Recession Indicator Daily | 1900-01-01 | 2026-04-02 | Daily | Binary indicator (0/1) | Recession | data/consolidated_csvs/fred/raw/recession_NBER_Recession_Indicator_Daily.csv |
| FRED | NBER Recession Indicator Monthly | 1900-01-01 | 2026-03-01 | Monthly | Binary indicator (0/1) | Recession | data/consolidated_csvs/fred/raw/recession_NBER_Recession_Indicator_Monthly.csv |
| FRED | Smoothed US Recession Probability | 1967-06-01 | 2026-02-01 | Monthly | Probability | Recession | data/consolidated_csvs/fred/raw/recession_Smoothed_Recession_Probability.csv |

## Asset Panel Contents

| column_name | description | start_date | end_date | frequency | measurement | sector | currency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0_5Y_TIPS | Bloomberg US Treasury TIPS 0-5Y Total Return Index | 2010-05-31 | 2026-04-14 | Daily | Index level | Fixed Income | USD |
| AUD | Australian Dollar Spot (AUD/USD) | 1980-12-31 | 2026-04-15 | Daily | FX spot rate | FX | USD |
| B3REITT | Bloomberg US REIT Total Return Index | 2003-03-31 | 2026-03-18 | Daily | Index level | Real Estate | USD |
| B3REITT_VOL | Bloomberg US REIT Total Return Index trading volume | 2003-03-31 | 2026-03-18 | Daily | Trading volume | Real Estate volume | USD |
| BAIGTRUU_ASIACREDIT | Bloomberg Asia USD Investment Grade Credit Total Return Index | 2014-05-22 | 2026-04-14 | Daily | Index level | Fixed Income | USD |
| BCEE1T_EUROAREA | Bloomberg Euro Aggregate Treasury 1-3Y Total Return Index | 2000-01-03 | 2026-04-14 | Daily | Index level | Fixed Income | EUR |
| BITCOIN | Bitcoin Spot (BTC/USD) | 2010-07-19 | 2026-03-19 | Daily | Price level | Crypto | USD |
| BROAD_TIPS | Bloomberg US Treasury Inflation-Linked Bonds Total Return Index | 1998-03-31 | 2026-04-14 | Daily | Index level | Fixed Income | USD |
| CAD | Canadian Dollar Spot (CAD/USD) | 1980-12-31 | 2026-04-15 | Daily | FX spot rate | FX | USD |
| CHF_FRANC | Swiss Franc Spot (CHF/USD) | 1980-12-31 | 2026-04-15 | Daily | FX spot rate | FX | USD |
| CNY | Chinese Yuan Renminbi Spot (CNY/USD) | 1981-01-02 | 2026-04-15 | Daily | FX spot rate | FX | USD |
| COCOAINDEXSPOT | S&P GSCI Cocoa Index Spot | 1984-01-06 | 2026-04-14 | Daily | Price level | Commodities | USD |
| COFFEE_FUT | Coffee C Front-Month Generic Futures (ICE) | 1979-12-31 | 2026-04-14 | Daily | Level | Commodities | USd |
| COFFEE_FUT_VOL | Coffee C Front-Month Generic Futures (ICE) trading volume | 1989-07-13 | 2026-04-15 | Daily | Trading volume | Commodities volume | USd |
| COPPERSPOT | LME Copper Cash Spot | 1986-04-01 | 2026-04-14 | Daily | Price level | Commodities | USD |
| COPPERSPOT_VOL | LME Copper Cash Spot trading volume | 2005-07-07 | 2026-04-14 | Daily | Trading volume | Commodities volume | USD |
| COTTON_FUT | Cotton #2 Front-Month Generic Futures (ICE) | 1979-12-31 | 2026-04-15 | Daily | Level | Commodities | USd |
| COTTON_FUT_VOL | Cotton #2 Front-Month Generic Futures (ICE) trading volume | 1989-12-07 | 2026-04-15 | Daily | Trading volume | Commodities volume | USd |
| CSI300_CHINA | CSI 300 Index (China A-shares) | 2002-01-04 | 2026-04-14 | Daily | Index / level | Equities | USD |
| ETHEREUM | Ethereum Spot (ETH/USD) | 2018-02-08 | 2026-04-15 | Daily | Price level | Crypto | USD |
| EURO | Euro Spot (EUR/USD) | 1998-12-31 | 2026-04-15 | Daily | FX spot rate | FX | USD |
| FTSE100 | FTSE 100 Index (UK) | 1983-12-30 | 2026-04-14 | Daily | Index / level | Equities | USD |
| FTSE100_VOL | FTSE 100 Index (UK) trading volume | 2002-02-04 | 2026-04-14 | Daily | Trading volume | Equities volume | USD |
| GBP_POUND | British Pound Spot (GBP/USD) | 1980-12-31 | 2026-04-15 | Daily | FX spot rate | FX | USD |
| I02923JP_JAPAN_BOND | Bloomberg Japan Treasury Bond Index | 2000-07-03 | 2026-04-14 | Daily | Price / index level | Fixed Income | JPY |
| LBEATREU_EUROBONDAGG | Bloomberg Euro Aggregate Bond Total Return Index | 1998-07-31 | 2026-04-14 | Daily | Index level | Fixed Income | EUR |
| LBUSTRUU | Bloomberg US Treasury Total Return Index | 1988-12-30 | 2026-03-18 | Daily | Index level | Fixed Income | USD |
| NATURALGAS | Henry Hub Natural Gas Front-Month Generic Futures (NYMEX) | 1990-04-03 | 2026-04-15 | Daily | Level | Commodities | USD |
| NATURALGAS_VOL | Henry Hub Natural Gas Front-Month Generic Futures (NYMEX) trading volume | 1990-04-03 | 2026-04-15 | Daily | Trading volume | Commodities volume | USD |
| NIKKEI225 | Nikkei 225 Stock Average (Japan) | 1980-12-26 | 2026-04-14 | Daily | Level | Equities | USD |
| NIKKEI225_VOL | Nikkei 225 Stock Average (Japan) trading volume | 2004-01-05 | 2026-04-14 | Daily | Trading volume | Equities volume | USD |
| SHEKEL | Israeli Shekel Spot (ILS/USD) | 1981-04-13 | 2026-04-14 | Daily | FX spot rate | FX | USD |
| SILVER_FUT | Silver Spot (XAG/USD) | 1979-12-31 | 2026-04-15 | Daily | Price level | Commodities | USD |
| SOYBEAN_FUT | Soybean Front-Month Generic Futures (CBOT) | 1979-12-31 | 2026-04-15 | Daily | Level | Commodities | USd |
| SOYBEAN_FUT_VOL | Soybean Front-Month Generic Futures (CBOT) trading volume | 1979-12-31 | 2026-04-15 | Daily | Trading volume | Commodities volume | USd |
| SPXT | S&P 500 Total Return Index | 1989-09-11 | 2026-03-18 | Daily | Index level | Equities | USD |
| SPXT_VOL | S&P 500 Total Return Index trading volume | 1990-01-31 | 2026-03-18 | Daily | Trading volume | Equities volume | USD |
| TA-125_ISRAEL | Tel Aviv 125 Index (Israel) | 1991-12-31 | 2026-04-14 | Daily | Index / level | Equities | USD |
| USDJPY | US Dollar / Japanese Yen Spot | 1976-01-01 | 2026-03-19 | Daily | FX spot rate | FX | JPY |
| WHEAT_SPOT | S&P GSCI Wheat Index Spot | 1979-12-31 | 2026-04-14 | Daily | Price level | Commodities | USD |
| XAU | Gold Spot (XAU/USD) | 1976-01-02 | 2026-03-19 | Daily | Price level | Commodities | USD |

## Policy Asset Availability

| asset | group | classification | start_date | end_date | measurement | sector | available_at_portfolio_start |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0_5Y_TIPS | Opportunistic | Opportunistic | 2010-05-31 | 2026-04-14 | Index level | Fixed Income | no |
| AUD | Opportunistic | Opportunistic | 1980-12-31 | 2026-04-15 | FX spot rate | FX | yes |
| BAIGTRUU_ASIACREDIT | Opportunistic | Opportunistic | 2014-05-22 | 2026-04-14 | Index level | Fixed Income | no |
| BCEE1T_EUROAREA | Opportunistic | Opportunistic | 2000-01-03 | 2026-04-14 | Index level | Fixed Income | yes |
| CAD | Opportunistic | Opportunistic | 1980-12-31 | 2026-04-15 | FX spot rate | FX | yes |
| CNY | Opportunistic | Opportunistic | 1981-01-02 | 2026-04-15 | FX spot rate | FX | yes |
| COCOAINDEXSPOT | Opportunistic | Opportunistic | 1984-01-06 | 2026-04-14 | Price level | Commodities | yes |
| COFFEE_FUT | Opportunistic | Opportunistic | 1979-12-31 | 2026-04-14 | Level | Commodities | yes |
| COPPERSPOT | Opportunistic | Opportunistic | 1986-04-01 | 2026-04-14 | Price level | Commodities | yes |
| COTTON_FUT | Opportunistic | Opportunistic | 1979-12-31 | 2026-04-15 | Level | Commodities | yes |
| ETHEREUM | Opportunistic | Opportunistic | 2018-02-08 | 2026-04-15 | Price level | Crypto | no |
| EURO | Opportunistic | Opportunistic | 1998-12-31 | 2026-04-15 | FX spot rate | FX | yes |
| GBP_POUND | Opportunistic | Opportunistic | 1980-12-31 | 2026-04-15 | FX spot rate | FX | yes |
| I02923JP_JAPAN_BOND | Opportunistic | Opportunistic | 2000-07-03 | 2026-04-14 | Price / index level | Fixed Income | no |
| LBEATREU_EUROBONDAGG | Opportunistic | Opportunistic | 1998-07-31 | 2026-04-14 | Index level | Fixed Income | yes |
| NATURALGAS | Opportunistic | Opportunistic | 1990-04-03 | 2026-04-15 | Level | Commodities | yes |
| SHEKEL | Opportunistic | Opportunistic | 1981-04-13 | 2026-04-14 | FX spot rate | FX | yes |
| SOYBEAN_FUT | Opportunistic | Opportunistic | 1979-12-31 | 2026-04-15 | Level | Commodities | yes |
| TA-125_ISRAEL | Opportunistic | Opportunistic | 1991-12-31 | 2026-04-14 | Index / level | Equities | yes |
| USDJPY | Opportunistic | Opportunistic | 1976-01-01 | 2026-03-19 | FX spot rate | FX | yes |
| WHEAT_SPOT | Opportunistic | Opportunistic | 1979-12-31 | 2026-04-14 | Price level | Commodities | yes |
| US_3M_TREASURY_YIELD | Risk-Free | Risk-Free | 1981-09-01 | 2026-04-02 | Percent / rate | Rates | yes |
| B3REITT | SAA | Satellite | 2003-03-31 | 2026-03-18 | Index level | Real Estate | no |
| BITCOIN | SAA | Non-Traditional | 2010-07-19 | 2026-03-19 | Price level | Crypto | no |
| BROAD_TIPS | SAA | Core | 1998-03-31 | 2026-04-14 | Index level | Fixed Income | yes |
| CHF_FRANC | SAA | Non-Traditional | 1980-12-31 | 2026-04-15 | FX spot rate | FX | yes |
| CSI300_CHINA | SAA | Satellite | 2002-01-04 | 2026-04-14 | Index / level | Equities | no |
| FTSE100 | SAA | Core | 1983-12-30 | 2026-04-14 | Index / level | Equities | yes |
| LBUSTRUU | SAA | Core | 1988-12-30 | 2026-03-18 | Index level | Fixed Income | yes |
| NIKKEI225 | SAA | Satellite | 1980-12-26 | 2026-04-14 | Level | Equities | yes |
| SILVER_FUT | SAA | Satellite | 1979-12-31 | 2026-04-15 | Price level | Commodities | yes |
| SPXT | SAA | Core | 1989-09-11 | 2026-03-18 | Index level | Equities | yes |
| XAU | SAA | Satellite | 1976-01-02 | 2026-03-19 | Price level | Commodities | yes |
