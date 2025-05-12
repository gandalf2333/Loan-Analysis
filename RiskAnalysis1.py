# Save this as e.g., loan_analyzer_app_no_sec.py
import streamlit as st
import pandas as pd
import yfinance as yf
# Assuming sec_10k_scraper.py with fetch_sec_fallback_data is in the same directory
# try:
#     # --- Commented out SEC scraper import ---
#     # from sec_10k_scraper import fetch_sec_fallback_data
#     pass # Pass avoids syntax error from empty try block
# except ImportError:
#     st.error("Error: Could not import `Workspace_sec_fallback_data` from `sec_10k_scraper.py`. Make sure the file exists in the same directory.")
# Define a dummy function anyway to prevent potential NameErrors later if uncommented
def fetch_sec_fallback_data(ticker):
    st.warning("SEC fallback data fetching is DISABLED.")
    return {}

# --- Step 1: Fetch and Load Financials ---
# Modified to accept frequency ('annual' or 'quarterly')
def fetch_financials(ticker, frequency='annual'):
    """Fetches financial data ONLY from Yahoo Finance."""
    st.write(f"Fetching {frequency.capitalize()} data for {ticker} from Yahoo Finance...")
    try:
        stock = yf.Ticker(ticker)

        if frequency == 'annual':
            bs = stock.balance_sheet
            is_ = stock.financials
            cf = stock.cashflow
        elif frequency == 'quarterly':
            bs = stock.quarterly_balance_sheet
            is_ = stock.quarterly_financials
            cf = stock.quarterly_cashflow
        else:
            st.error("Invalid frequency specified.")
            return None

        if bs.empty or is_.empty or cf.empty:
            st.warning(f"Could not retrieve sufficient {frequency} data from Yahoo Finance for {ticker}.")
            # No SEC fallback attempt here

        # Get the most recent column
        if not bs.columns.empty:
            col = bs.columns[0]
        elif not is_.columns.empty:
            col = is_.columns[0]
        elif not cf.columns.empty:
            col = cf.columns[0]
        else:
            st.error(f"No data columns found for {ticker} ({frequency}).")
            return None # No data available

        # --- Extract Data ---
        data = {}
        missing = []

        # Helper function to safely get data
        def get_data(df, label, column):
            if label in df.index and column in df.columns and pd.notna(df.loc[label, column]):
                return df.loc[label, column]
            return None

        # Balance Sheet Items
        data['Current Assets'] = get_data(bs, 'Current Assets', col)
        data['Current Liabilities'] = get_data(bs, 'Current Liabilities', col)
        total_liab_label = 'Total Liabilities Net Minority Interest' if 'Total Liabilities Net Minority Interest' in bs.index else 'Total Liabilities'
        data['Total Liabilities'] = get_data(bs, total_liab_label, col)
        equity_label = 'Stockholders Equity' if 'Stockholders Equity' in bs.index else 'Total Equity Gross Minority Interest'
        data['Shareholder Equity'] = get_data(bs, equity_label, col)
        data['Total Assets'] = get_data(bs, 'Total Assets', col)
        data['Cash & Cash Equivalents'] = get_data(bs, 'Cash And Cash Equivalents', col)

        # Income Statement Items
        ebit_label = 'Operating Income' if 'Operating Income' in is_.index else 'Ebit'
        data['EBIT'] = get_data(is_, ebit_label, col)
        data['Interest Expense'] = abs(get_data(is_, 'Interest Expense', col)) if get_data(is_, 'Interest Expense', col) is not None else None
        data['Net Income'] = get_data(is_, 'Net Income', col)
        data['Revenue'] = get_data(is_, 'Total Revenue', col)
        data['Gross Profit'] = get_data(is_, 'Gross Profit', col)

        # Cash Flow Items
        depreciation_label = 'Depreciation And Amortization' if 'Depreciation And Amortization' in cf.index else 'Depreciation'
        depreciation = get_data(cf, depreciation_label, col)

        # Calculate EBITDA
        if data['EBIT'] is not None and depreciation is not None:
            data['EBITDA'] = data['EBIT'] + depreciation
        else:
             data['EBITDA'] = None

        # --- Check for Missing Data (No Fallback) ---
        for key, val in data.items():
            if val is None:
                missing.append(key)

        if missing:
            st.warning(f"Missing fields from Yahoo Finance ({frequency}): {', '.join(missing)}. SEC fallback is disabled.")

        # --- SEC Fallback Logic Block - START (Commented Out) ---
        # if missing:
        #     st.warning(f"Missing fields from Yahoo Finance ({frequency}): {', '.join(missing)}. Trying SEC fallback (if applicable).")
        #     # Note: SEC fallback primarily provides ANNUAL data. Using it for QUARTERLY gaps might be inaccurate.
        #     # Only fetch SEC if critical data is missing, could refine this logic.
        #     # For simplicity here, we try if anything is missing.
        #     sec_data = {}
        #     try:
        #          # This part assumes fetch_sec_fallback_data gets the latest annual data
        #          sec_data = fetch_sec_fallback_data(ticker)
        #          st.info(f"SEC Fallback data fetched: {list(sec_data.keys())}")
        #     except Exception as e:
        #          st.error(f"Failed to fetch SEC fallback data: {e}")

            # # Attempt to fill gaps ONLY if the sec_data has the corresponding key
            # for field in missing:
            #     # Map our desired field names to potential SEC data names
            #     sec_mapping = {
            #         'Total Liabilities': 'TotalLiabilities', # Example mapping, adjust based on your scraper output
            #         'Shareholder Equity': 'StockholdersEquity',
            #         'EBIT': 'OperatingIncomeLoss',
            #         'Interest Expense': 'InterestExpense',
            #         'Depreciation': 'DepreciationAndAmortization', # Need this intermediate for EBITDA calc
            #         'Revenue': 'Revenues', # Example
            #         'Net Income': 'NetIncomeLoss', # Example
            #         # Add other mappings as needed based on your scraper's output keys
            #     }
            #     # Specific handling for fields needed for calculations first
            #     if field == 'EBIT' and sec_mapping['EBIT'] in sec_data:
            #         data[field] = sec_data[sec_mapping['EBIT']]
            #         st.write(f"Filled EBIT from SEC fallback.")
            #     elif field == 'Interest Expense' and sec_mapping['Interest Expense'] in sec_data:
            #          # SEC data might have interest expense as negative, keep it positive for ratio calcs
            #         data[field] = abs(sec_data[sec_mapping['Interest Expense']])
            #         st.write(f"Filled Interest Expense from SEC fallback.")

            #     # Attempt to recalculate EBITDA if components were filled
            #     if data['EBITDA'] is None and data['EBIT'] is not None:
            #         # Try to get depreciation from SEC if not already found
            #         if depreciation is None and sec_mapping['Depreciation'] in sec_data:
            #             depreciation = sec_data[sec_mapping['Depreciation']]
            #             st.write(f"Filled Depreciation from SEC fallback.")
            #         if depreciation is not None:
            #             data['EBITDA'] = data['EBIT'] + depreciation
            #             st.write(f"Recalculated EBITDA using SEC fallback data.")
            #             if 'EBITDA' in missing: missing.remove('EBITDA') # No longer missing

            #     # Fill other fields if still missing and available in SEC data
            #     if data[field] is None and field in sec_mapping and sec_mapping[field] in sec_data:
            #          data[field] = sec_data[sec_mapping[field]]
            #          st.write(f"Filled {field} from SEC fallback.")

            # # Final check after fallback
            # final_missing = [k for k, v in data.items() if v is None]
            # if final_missing:
            #      st.warning(f"Still missing after fallback: {', '.join(final_missing)}")
        # --- SEC Fallback Logic Block - END (Commented Out) ---

        return pd.DataFrame([data])

    except Exception as e:
        st.error(f"❌ Error fetching or processing data for {ticker}: {e}")
        import traceback
        st.error(traceback.format_exc()) # Show detailed error in app for debugging
        return None

# --- Step 2: Calculate Financial Ratios ---
# (Safely handles potential division by zero or None values)
def calculate_ratios(df):
    """Calculates financial ratios from the DataFrame."""
    ratios = {}
    data = df.iloc[0] # Get the single row of data

    try:
        # Liquidity Ratios
        if pd.notnull(data['Current Assets']) and pd.notnull(data['Current Liabilities']) and data['Current Liabilities'] != 0:
            ratios['Current Ratio'] = data['Current Assets'] / data['Current Liabilities']
        if pd.notnull(data['Cash & Cash Equivalents']) and pd.notnull(data['Current Liabilities']) and data['Current Liabilities'] != 0:
            ratios['Quick Ratio'] = data['Cash & Cash Equivalents'] / data['Current Liabilities']

        # Leverage Ratios
        if pd.notnull(data['Total Liabilities']) and pd.notnull(data['Shareholder Equity']) and data['Shareholder Equity'] != 0:
            ratios['Debt-to-Equity'] = data['Total Liabilities'] / data['Shareholder Equity']
        if pd.notnull(data['Total Liabilities']) and pd.notnull(data['Total Assets']) and data['Total Assets'] != 0:
            ratios['Leverage Ratio (Debt/Assets)'] = data['Total Liabilities'] / data['Total Assets']

        # Coverage Ratios
        if pd.notnull(data['EBIT']) and pd.notnull(data['Interest Expense']) and data['Interest Expense'] != 0:
            ratios['Interest Coverage Ratio (EBIT/Int)'] = data['EBIT'] / data['Interest Expense']
        # Note: Sometimes DSCR is defined using EBITDA or other cash flow measures. Sticking to EBIT as per original code.

        # Profitability Ratios
        if pd.notnull(data['Net Income']) and pd.notnull(data['Revenue']) and data['Revenue'] != 0:
            ratios['Net Profit Margin'] = data['Net Income'] / data['Revenue']
        if pd.notnull(data['Gross Profit']) and pd.notnull(data['Revenue']) and data['Revenue'] != 0:
            ratios['Gross Margin'] = data['Gross Profit'] / data['Revenue']
        if pd.notnull(data['EBITDA']) and pd.notnull(data['Revenue']) and data['Revenue'] != 0:
            ratios['EBITDA Margin'] = data['EBITDA'] / data['Revenue']

        # Return Ratios
        if pd.notnull(data['Net Income']) and pd.notnull(data['Total Assets']) and data['Total Assets'] != 0:
            ratios['ROA (Return on Assets)'] = data['Net Income'] / data['Total Assets']
        if pd.notnull(data['Net Income']) and pd.notnull(data['Shareholder Equity']) and data['Shareholder Equity'] != 0:
            ratios['ROE (Return on Equity)'] = data['Net Income'] / data['Shareholder Equity']

    except Exception as e:
        st.error(f"❌ Error calculating ratios: {e}")

    # Ensure all potential ratio keys exist, even if calculation failed
    all_ratio_keys = [
        'Current Ratio', 'Quick Ratio', 'Debt-to-Equity', 'Leverage Ratio (Debt/Assets)',
        'Interest Coverage Ratio (EBIT/Int)', 'Net Profit Margin', 'Gross Margin',
        'EBITDA Margin', 'ROA (Return on Assets)', 'ROE (Return on Equity)'
    ]
    for key in all_ratio_keys:
        if key not in ratios:
            ratios[key] = None # Mark as None if calculation failed or data was missing

    return ratios


# --- Step 3: Loan Scenario Modeling ---
def simulate_loan_impact(df, loan_amount, interest_rate):
    """Simulates the impact of a new loan on interest coverage."""
    results = {
        'New Annual Interest': None,
        'Pro-Forma Interest Expense': None,
        'Pro-Forma Interest Coverage Ratio': None,
        'Acceptable Coverage (>1.25x)?': None # Common covenant threshold
    }
    data = df.iloc[0]

    try:
        if pd.notnull(data['EBIT']) and pd.notnull(data['Interest Expense']):
            annual_interest_increase = loan_amount * interest_rate
            new_total_interest_expense = data['Interest Expense'] + annual_interest_increase

            results['New Annual Interest'] = annual_interest_increase
            results['Pro-Forma Interest Expense'] = new_total_interest_expense

            if new_total_interest_expense != 0:
                new_icr = data['EBIT'] / new_total_interest_expense
                results['Pro-Forma Interest Coverage Ratio'] = new_icr
                results['Acceptable Coverage (>1.25x)?'] = new_icr > 1.25
            else:
                # Handle case where interest expense becomes zero (unlikely but possible)
                 results['Pro-Forma Interest Coverage Ratio'] = float('inf') # Or handle as very high coverage
                 results['Acceptable Coverage (>1.25x)?'] = True

    except Exception as e:
        st.error(f"❌ Error simulating loan impact: {e}")

    return results


# --- Step 4: Risk Scoring ---
# (Using slightly adjusted ratio names from calculate_ratios)
def assess_risk(ratios):
    """Assesses risk level based on calculated ratios."""
    score = 0
    thresholds = {
        'Current Ratio': 1.5,
        'Debt-to-Equity': 2.0, # Lower is better
        'Interest Coverage Ratio (EBIT/Int)': 3.0,
        'Net Profit Margin': 0.05, # Adjusted threshold slightly
        'EBITDA Margin': 0.10, # Adjusted threshold slightly
        'Quick Ratio': 1.0,
        'ROA (Return on Assets)': 0.05
    }
    achieved = []

    if ratios.get('Current Ratio') is not None and ratios['Current Ratio'] >= thresholds['Current Ratio']:
        score += 1; achieved.append("Good Current Ratio")
    if ratios.get('Debt-to-Equity') is not None and ratios['Debt-to-Equity'] <= thresholds['Debt-to-Equity']: # Note: Less than or equal
        score += 1; achieved.append("Manageable Debt-to-Equity")
    if ratios.get('Interest Coverage Ratio (EBIT/Int)') is not None and ratios['Interest Coverage Ratio (EBIT/Int)'] >= thresholds['Interest Coverage Ratio (EBIT/Int)']:
        score += 1; achieved.append("Strong Interest Coverage")
    if ratios.get('Net Profit Margin') is not None and ratios['Net Profit Margin'] >= thresholds['Net Profit Margin']:
        score += 1; achieved.append("Healthy Net Profit Margin")
    if ratios.get('EBITDA Margin') is not None and ratios['EBITDA Margin'] >= thresholds['EBITDA Margin']:
        score += 1; achieved.append("Good EBITDA Margin")
    if ratios.get('Quick Ratio') is not None and ratios['Quick Ratio'] >= thresholds['Quick Ratio']:
        score += 1; achieved.append("Solid Quick Ratio")
    if ratios.get('ROA (Return on Assets)') is not None and ratios['ROA (Return on Assets)'] >= thresholds['ROA (Return on Assets)']:
        score += 1; achieved.append("Positive ROA")

    max_score = len(thresholds) # Max possible score is 7

    if score >= 5:
        risk_level = 'Low Risk'
        color = 'green'
    elif score >= 3:
        risk_level = 'Medium Risk'
        color = 'orange'
    else:
        risk_level = 'High Risk'
        color = 'red'

    summary = f"Score: {score}/{max_score}. Strengths: {', '.join(achieved) if achieved else 'None identified based on thresholds.'}"
    return risk_level, color, summary


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("📊 Loan Risk Analyzer (Yahoo Finance Only)") # Updated title
st.caption("Analyzes financial health and loan impact for borrowers using ONLY Yahoo Finance data.") # Updated caption

# --- Inputs in Sidebar ---
st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").upper()
frequency = st.sidebar.radio("Select Data Frequency:", ('Annual', 'Quarterly'), index=0) # Default Annual

st.sidebar.header("Loan Scenario (Optional)")
loan_amount = st.sidebar.number_input("Loan Amount ($):", min_value=0, value=1000000, step=100000)
interest_rate = st.sidebar.slider("Annual Interest Rate (%):", min_value=0.0, max_value=25.0, value=7.0, step=0.1) / 100.0 # Convert to decimal
# term_years = st.sidebar.number_input("Loan Term (Years):", min_value=1, value=5, step=1) # Original code didn't use term_years

analyze_button = st.sidebar.button("🚀 Analyze Borrower")

# --- Main Analysis Area ---
if analyze_button and ticker:
    with st.spinner(f"Analyzing {ticker} ({frequency})... Please wait."):
        # 1. Fetch Data
        df_financials = fetch_financials(ticker, frequency.lower())

        if df_financials is not None and not df_financials.empty:
            st.subheader(f"🏦 Financial Snapshot ({frequency}) for {ticker}")
            # Format with commas and 0 decimal places, N/A for missing
            st.dataframe(df_financials.T.rename(columns={0:'Value'}).style.format("${:,.0f}", na_rep="N/A"))

            # 2. Calculate Ratios
            st.subheader("📊 Key Financial Ratios")
            ratios = calculate_ratios(df_financials)
            col1, col2, col3 = st.columns(3)
            ratio_cols = [col1, col2, col3]
            i = 0
            for k, v in ratios.items():
                with ratio_cols[i % 3]: # Cycle through columns
                     # Format ratios to 2 decimal places, N/A for missing
                     st.metric(label=k, value=f"{v:.2f}" if v is not None else "N/A")
                i += 1
            if any(v is None for v in ratios.values()):
                st.caption("N/A indicates missing data or calculation error for that ratio.")


            # 3. Loan Impact Simulation
            st.subheader("💸 Loan Impact Simulation")
            if loan_amount > 0:
                loan_results = simulate_loan_impact(df_financials, loan_amount, interest_rate)
                col_l1, col_l2, col_l3, col_l4 = st.columns(4)
                with col_l1:
                    # Format currency values
                    st.metric("New Annual Interest", f"${loan_results.get('New Annual Interest'):,.0f}" if loan_results.get('New Annual Interest') is not None else "N/A")
                with col_l2:
                    st.metric("Pro-Forma Total Interest Exp.", f"${loan_results.get('Pro-Forma Interest Expense'):,.0f}" if loan_results.get('Pro-Forma Interest Expense') is not None else "N/A")
                with col_l3:
                    # Format ratio value
                    st.metric("Pro-Forma Interest Coverage", f"{loan_results.get('Pro-Forma Interest Coverage Ratio'):.2f}x" if loan_results.get('Pro-Forma Interest Coverage Ratio') is not None else "N/A")
                with col_l4:
                    accept_cov = loan_results.get('Acceptable Coverage (>1.25x)?')
                    st.metric("Coverage Acceptable (>1.25x)?", "✅ Yes" if accept_cov is True else ("❌ No" if accept_cov is False else "N/A") )

            else:
                st.info("Enter a loan amount > 0 in the sidebar to simulate impact.")

            # 4. Risk Assessment
            st.subheader("🚦 Overall Risk Assessment")
            risk_level, risk_color, risk_summary = assess_risk(ratios)
            st.markdown(f"**<font color='{risk_color}'>{risk_level}</font>**", unsafe_allow_html=True)
            st.caption(risk_summary)


        else:
            st.error(f"Could not retrieve or process financial data for {ticker} using Yahoo Finance. Please check the ticker and try again, or check the logs above for errors.")

elif not ticker:
    st.info("Please enter a stock ticker in the sidebar and click 'Analyze Borrower'.")