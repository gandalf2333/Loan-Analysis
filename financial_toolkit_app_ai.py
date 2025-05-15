# Save this as e.g., financial_toolkit_app_ai.py
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import google.generativeai as genai # For Google Gemini API
import traceback # For more detailed error logging
from curl_cffi import requests as cffi_requests # For custom user-agent with yfinance

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Financial Analysis Toolkit")

# --- Helper Functions ---
def format_value(value, format_type="currency", na_rep="N/A"):
    """Formats numbers for display."""
    if pd.isna(value):
        return na_rep
    if format_type == "currency":
        return f"${value:,.0f}" # For general currency with no decimals
    if format_type == "currency_precise": # Added for things like EPS and stock prices
        return f"${value:,.2f}"
    if format_type == "ratio":
        return f"{value:.2f}"
    if format_type == "percent":
        # Expects a decimal fraction (e.g., 0.25 for 25%), multiplies by 100
        return f"{value:.2%}"
    if format_type == "number":
        return f"{value:,.0f}"
    return str(value) # Ensure it's a string if no format matches

def get_safe_value(data_structure, keys, default=None):
    """
    Safely get a value from a Pandas Series or a dictionary.
    `keys` can be a single key string or a list of potential keys to try.
    """
    if not isinstance(keys, list):
        keys = [keys] # Convert single key to list

    if isinstance(data_structure, pd.Series):
        for key in keys:
            if key in data_structure.index and pd.notna(data_structure[key]):
                return data_structure[key]
    elif isinstance(data_structure, dict):
        for key in keys:
            if key in data_structure and pd.notna(data_structure[key]):
                return data_structure[key]
    return default

# --- AI Summary Function (Google Gemini) ---
def generate_ai_summary_gemini(stock_info_dict, latest_ratios_series):
    """
    Generates a financial health summary using Google's Gemini API,
    with an improved prompt to reduce data mixing.
    """
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            return "Error: Google API key not found in Streamlit secrets. Please add `GOOGLE_API_KEY = \"YOUR_KEY\"` to `.streamlit/secrets.toml`."

        genai.configure(api_key=api_key)
        model_name = 'gemini-1.5-flash-latest' # Or 'gemini-1.0-pro-latest'
        model = genai.GenerativeModel(model_name)

        company_name = get_safe_value(stock_info_dict, 'shortName', 'the company')
        sector = get_safe_value(stock_info_dict, 'sector', 'N/A')
        
        ratios_text_list = []
        if latest_ratios_series is not None and not latest_ratios_series.empty:
            for ratio_name, value in latest_ratios_series.dropna().items():
                # Using your existing format_value logic for consistency
                if "Margin" in ratio_name or "ROE" in ratio_name or "ROA" in ratio_name:
                    ratios_text_list.append(f"- {ratio_name}: {format_value(value, 'percent')}")
                elif "EPS" in ratio_name: 
                    ratios_text_list.append(f"- {ratio_name}: {format_value(value, 'currency_precise')}")
                else: 
                    ratios_text_list.append(f"- {ratio_name}: {format_value(value, 'ratio')}")
        ratios_as_text = "\n".join(ratios_text_list) if ratios_text_list else "No detailed financial ratio data was provided for this period."

        # Revised Prompt Structure:
        prompt = f"""
        You are a concise financial analyst assistant.
        Your task is to analyze the provided data for {company_name} (Sector: {sector}).

        Please analyze the following two distinct sections of data:

        SECTION 1: GENERAL STOCK INFORMATION
        --- Start of General Stock Information ---
        - Market Cap: {format_value(get_safe_value(stock_info_dict, 'marketCap'), 'currency')}
        - Trailing P/E: {format_value(get_safe_value(stock_info_dict, 'trailingPE'), 'ratio')}
        - Forward P/E: {format_value(get_safe_value(stock_info_dict, 'forwardPE'), 'ratio')}
        - Beta: {format_value(get_safe_value(stock_info_dict, 'beta'), 'ratio')}
        --- End of General Stock Information ---

        SECTION 2: LATEST FINANCIAL RATIOS
        --- Start of Latest Financial Ratios ---
        {ratios_as_text}
        --- End of Latest Financial Ratios ---

        Instructions for your summary:
        1.  First, provide a brief (1-2 sentences) interpretation of what the 'General Stock Information' (SECTION 1) suggests about the company's market valuation and perceived risk.
        2.  Next, analyze the 'Latest Financial Ratios' (SECTION 2). Identify 2-3 key financial strengths and 2-3 potential areas for attention or weaknesses evident from these ratios. If ratio data is sparse or missing from SECTION 2, please state that clearly.
        3.  Combine these analyses into a single, coherent summary of the company's overall financial health, approximately 150-200 words in total.
        4.  Crucially, ensure that your discussion of EPS or other specific ratios from SECTION 2 uses ONLY the values provided in SECTION 2, and does not mix them with values from SECTION 1 (like Market Cap).
        5.  Do NOT provide any investment advice, buy/sell/hold recommendations, or future price predictions.
        6.  Base your analysis ONLY on the numerical data provided in the sections above. Do not infer or add external information.
        
        Please provide your analytical summary:
        """

        generation_config = genai.types.GenerationConfig(
            temperature=0.2, # Slightly lower temperature for more focused output
            max_output_tokens=350 
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        
        summary = ""
        # Robustly extract text from response
        if hasattr(response, 'text') and response.text:
            summary = response.text
        elif response.parts:
            for part in response.parts:
                if hasattr(part, 'text'):
                    summary += part.text
        
        if not summary: # If summary is still empty after trying common attributes
            # st.warning(f"Gemini response object structure for debugging: {response}") # For debugging
            return "Error: Could not extract text from Gemini response. The response might be empty or the API structure has changed."
            
        return summary.strip()

    except Exception as e:
        st.error(f"An error occurred while generating AI summary with Gemini: {e}")
        st.error(f"Traceback: {traceback.format_exc()}") 
        return "Error: Could not generate AI summary with Gemini. Check console for details.

# --- Step 1: Fetch Financial Data (Multi-Year) & Stock Info (MODIFIED) ---
@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_financial_data_multi_year(ticker_symbol, frequency='annual'):
    st.write(f"Fetching {frequency.capitalize()} data for {ticker_symbol} from Yahoo Finance...")
    try:
        # Create a session with a common browser user-agent
        session = cffi_requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        stock = yf.Ticker(ticker_symbol, session=session) # Pass the session here
        stock_info = stock.info # This is where the error occurred

        if not stock_info or stock_info.get('regularMarketPrice') is None:
            st.error(f"Could not retrieve valid stock information for {ticker_symbol} after attempting with custom headers. It might be delisted or an incorrect ticker.")
            return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

        bs, is_, cf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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
            return stock_info, bs, is_, cf, None # Return info even if freq is bad

        if bs.empty and is_.empty and cf.empty:
            st.warning(f"No {frequency} financial statement data found for {ticker_symbol} on Yahoo Finance.")
        
        hist = stock.history(period="4y")
        return stock_info, bs, is_, cf, hist

    except cffi_requests.RequestsError as http_err: # Catching curl_cffi specific HTTP errors
        st.error(f"❌ HTTP Error fetching data for {ticker_symbol}: {http_err}")
        st.error(f"This often happens when Yahoo Finance blocks requests from cloud platforms. The custom user-agent trick may not always work.")
        st.error(traceback.format_exc())
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None
    except Exception as e:
        st.error(f"❌ An unexpected error occurred while fetching data for {ticker_symbol}: {e}")
        st.error(traceback.format_exc())
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

# --- Step 2: Calculate Financial Ratios (Multi-Year) ---
def calculate_ratios_multi_year(balance_sheet, income_statement, cashflow_statement, stock_info_latest):
    ratios_over_time = pd.DataFrame()
    if balance_sheet.empty or income_statement.empty:
        return ratios_over_time

    common_periods = balance_sheet.columns.intersection(income_statement.columns)
    if not cashflow_statement.empty:
        common_periods = common_periods.intersection(cashflow_statement.columns)
    if common_periods.empty:
        return ratios_over_time
    if isinstance(common_periods, pd.DatetimeIndex):
        common_periods = common_periods.sort_values(ascending=True) 

    for period in common_periods:
        ratios = {}
        bs_period = balance_sheet[period]
        is_period = income_statement[period]
        cf_period = cashflow_statement[period] if not cashflow_statement.empty and period in cashflow_statement.columns else pd.Series(dtype='float64')

        current_assets = get_safe_value(bs_period, ['Current Assets', 'Total Current Assets'])
        current_liabilities = get_safe_value(bs_period, ['Current Liabilities', 'Total Current Liabilities'])
        cash_equivalents = get_safe_value(bs_period, ['Cash And Cash Equivalents', 'Cash', 'Cash And Short Term Investments'])
        inventory = get_safe_value(bs_period, ['Inventory'])
        accounts_receivable = get_safe_value(bs_period, ['Net Receivables', 'Accounts Receivable'])
        total_liabilities_val = get_safe_value(bs_period, ['Total Liabilities Net Minority Interest', 'Total Liab', 'Total Liabilities'])
        shareholder_equity = get_safe_value(bs_period, ['Stockholders Equity', 'Total Stockholder Equity'])
        total_assets = get_safe_value(bs_period, ['Total Assets'])
        long_term_debt = get_safe_value(bs_period, ['Long Term Debt'])
        short_term_debt = get_safe_value(bs_period, ['Short Term Debt', 'Current Debt', 'Short Long Term Debt'])
        
        extracted_total_debt = get_safe_value(bs_period, ['Total Debt'])
        calculated_total_debt = (long_term_debt if pd.notna(long_term_debt) else 0) + \
                                (short_term_debt if pd.notna(short_term_debt) else 0)
        total_debt = extracted_total_debt if pd.notna(extracted_total_debt) and extracted_total_debt != 0 else calculated_total_debt
        if total_debt == 0 and pd.notna(extracted_total_debt):
            total_debt = extracted_total_debt

        ebit = get_safe_value(is_period, ['Operating Income', 'Ebit', 'Earnings Before Interest And Taxes'])
        interest_expense_val = get_safe_value(is_period, ['Interest Expense'])
        interest_expense = abs(interest_expense_val) if pd.notna(interest_expense_val) else None
        net_income = get_safe_value(is_period, ['Net Income', 'Net Income Applicable To Common Shares'])
        revenue = get_safe_value(is_period, ['Total Revenue', 'Revenues'])
        gross_profit = get_safe_value(is_period, ['Gross Profit'])
        cogs = get_safe_value(is_period, ['Cost Of Revenue', 'Cost of Goods Sold'])
        
        ebitda = get_safe_value(is_period, ['EBITDA', 'Normalized EBITDA'])
        if pd.isna(ebitda): 
            depreciation_amortization_is = get_safe_value(is_period, ['Depreciation And Amortization', 'Depreciation & Amortization'])
            depreciation_amortization_cf = get_safe_value(cf_period, ['Depreciation And Amortization', 'Depreciation'])
            depreciation = depreciation_amortization_is if pd.notna(depreciation_amortization_is) else depreciation_amortization_cf
            if pd.notna(ebit) and pd.notna(depreciation):
                ebitda = ebit + depreciation
        
        basic_eps = get_safe_value(is_period, ['Basic EPS'])
        diluted_eps = get_safe_value(is_period, ['Diluted EPS'])
        ratios['EPS (Basic)'] = basic_eps
        ratios['EPS (Diluted)'] = diluted_eps if pd.notna(diluted_eps) else basic_eps

        op_cash_flow = get_safe_value(cf_period, ['Total Cash From Operating Activities', 'Operating Cash Flow'])
        cap_ex_val = get_safe_value(cf_period, ['Capital Expenditures', 'Capital Expenditure'])
        cap_ex = abs(cap_ex_val) if pd.notna(cap_ex_val) else None

        if pd.notna(current_assets) and pd.notna(current_liabilities) and current_liabilities != 0:
            ratios['Current Ratio'] = current_assets / current_liabilities
        if pd.notna(cash_equivalents) and pd.notna(current_liabilities) and current_liabilities != 0:
            ratios['Quick Ratio (Cash based)'] = cash_equivalents / current_liabilities
        if pd.notna(current_assets) and pd.notna(inventory) and pd.notna(current_liabilities) and current_liabilities != 0:
            ratios['Quick Ratio (Excl. Inventory)'] = (current_assets - inventory) / current_liabilities
        if pd.notna(total_debt) and pd.notna(shareholder_equity) and shareholder_equity != 0:
            ratios['Debt-to-Equity'] = total_debt / shareholder_equity
        if pd.notna(total_liabilities_val) and pd.notna(total_assets) and total_assets != 0: 
            ratios['Debt-to-Assets'] = total_liabilities_val / total_assets
        if pd.notna(total_assets) and pd.notna(shareholder_equity) and shareholder_equity != 0:
             ratios['Financial Leverage (Assets/Equity)'] = total_assets / shareholder_equity
        if pd.notna(ebit) and pd.notna(interest_expense) and interest_expense != 0:
            ratios['Interest Coverage Ratio (EBIT/Int)'] = ebit / interest_expense
        if pd.notna(net_income) and pd.notna(revenue) and revenue != 0:
            ratios['Net Profit Margin'] = net_income / revenue
        if pd.notna(gross_profit) and pd.notna(revenue) and revenue != 0:
            ratios['Gross Profit Margin'] = gross_profit / revenue
        if pd.notna(ebitda) and pd.notna(revenue) and revenue != 0:
            ratios['EBITDA Margin'] = ebitda / revenue
        if pd.notna(ebit) and pd.notna(revenue) and revenue != 0:
            ratios['Operating Margin'] = ebit / revenue
        if pd.notna(net_income) and pd.notna(total_assets) and total_assets != 0:
            ratios['ROA (Return on Assets)'] = net_income / total_assets
        if pd.notna(net_income) and pd.notna(shareholder_equity) and shareholder_equity != 0:
            ratios['ROE (Return on Equity)'] = net_income / shareholder_equity
        if pd.notna(revenue) and pd.notna(total_assets) and total_assets != 0:
            ratios['Asset Turnover'] = revenue / total_assets
        if pd.notna(cogs) and pd.notna(inventory) and inventory != 0:
            inv_turnover = cogs / inventory
            ratios['Inventory Turnover'] = inv_turnover
            if inv_turnover != 0: ratios['Days Inventory Outstanding (DIO)'] = 365 / inv_turnover
        if pd.notna(revenue) and pd.notna(accounts_receivable) and accounts_receivable != 0:
            rec_turnover = revenue / accounts_receivable
            ratios['Receivables Turnover'] = rec_turnover
            if rec_turnover != 0: ratios['Days Sales Outstanding (DSO)'] = 365 / rec_turnover
        if pd.notna(op_cash_flow) and pd.notna(cap_ex):
            ratios['Free Cash Flow (FCF)'] = op_cash_flow - cap_ex
        
        period_label = period.strftime('%Y-%m-%d') if isinstance(period, pd.Timestamp) else str(period)
        ratios_over_time[period_label] = pd.Series(ratios)
    return ratios_over_time.sort_index(axis=1, ascending=False)

# --- Step 3: Loan Scenario Modeling ---
def simulate_loan_impact(latest_financial_data, loan_amount, interest_rate_decimal):
    results = {'New Annual Interest': None, 'Pro-Forma Interest Expense': None,
               'Pro-Forma Interest Coverage Ratio': None, 'Acceptable Coverage (>1.25x)?': None}
    if not latest_financial_data: 
        st.warning("Loan simulation: essential financial data (EBIT, Interest Expense) missing.")
        return results
    ebit = get_safe_value(latest_financial_data, 'EBIT')
    current_interest_expense = get_safe_value(latest_financial_data, 'Interest Expense', default=0.0)

    if pd.notna(ebit) and pd.notna(current_interest_expense):
        annual_interest_increase = loan_amount * interest_rate_decimal
        new_total_interest_expense = current_interest_expense + annual_interest_increase
        results['New Annual Interest'] = annual_interest_increase
        results['Pro-Forma Interest Expense'] = new_total_interest_expense
        if new_total_interest_expense != 0:
            new_icr = ebit / new_total_interest_expense
            results['Pro-Forma Interest Coverage Ratio'] = new_icr
            results['Acceptable Coverage (>1.25x)?'] = new_icr > 1.25
        else: 
            if ebit > 0:
                results['Pro-Forma Interest Coverage Ratio'] = float('inf')
                results['Acceptable Coverage (>1.25x)?'] = True
            elif ebit < 0:
                results['Pro-Forma Interest Coverage Ratio'] = float('-inf')
                results['Acceptable Coverage (>1.25x)?'] = False
            else: 
                results['Pro-Forma Interest Coverage Ratio'] = 0.0 
                results['Acceptable Coverage (>1.25x)?'] = False 
    return results

# --- Step 4: Risk Scoring ---
def assess_risk_from_ratios(latest_ratios_series):
    score = 0
    thresholds = {'Current Ratio': 1.5, 'Debt-to-Equity': 2.0, 'Interest Coverage Ratio (EBIT/Int)': 3.0,
                  'Net Profit Margin': 0.05, 'EBITDA Margin': 0.10, 
                  'Quick Ratio (Excl. Inventory)': 1.0, 
                  'ROA (Return on Assets)': 0.05}
    achieved, max_score = [], len(thresholds)
    if latest_ratios_series is None or latest_ratios_series.empty: 
        return 'N/A', 'grey', "Not enough data for risk assessment."

    def check_ratio(name, threshold, is_greater_better=True):
        nonlocal score
        val = get_safe_value(latest_ratios_series, name)
        if pd.notnull(val):
            formatted_val_display = f"{val:.2f}"
            if (is_greater_better and val >= threshold) or (not is_greater_better and val <= threshold):
                score += 1; achieved.append(f"{('Good' if is_greater_better else 'Manageable')} {name} ({formatted_val_display})")
    
    for name, thresh in thresholds.items():
        check_ratio(name, thresh, is_greater_better=(name != 'Debt-to-Equity'))
    
    if score >= max_score * 0.7: risk_level, color = 'Low Risk', 'green'
    elif score >= max_score * 0.4: risk_level, color = 'Medium Risk', 'orange'
    else: risk_level, color = 'High Risk', 'red'
    summary_text = f"Score: {score}/{max_score}. Strengths: {', '.join(achieved) if achieved else 'None identified based on thresholds.'}"
    return risk_level, color, summary_text

# --- Streamlit App UI ---
st.title("🏢 Financial Analysis Toolkit")
st.caption("Comprehensive financial analysis using Yahoo Finance data. All monetary values are in the currency reported by Yahoo Finance (usually USD unless specified).")

st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Enter Stock Ticker:", "MSFT").upper()
frequency_options = {'Annual': 'annual', 'Quarterly': 'quarterly'}
selected_frequency_label = st.sidebar.radio("Select Data Frequency:", list(frequency_options.keys()), index=0, key="data_frequency_radio")
frequency_value = frequency_options[selected_frequency_label]
analyze_button = st.sidebar.button("🚀 Analyze Company")

if analyze_button and ticker:
    with st.spinner(f"Fetching and analyzing {ticker} ({selected_frequency_label})... This may take a moment."):
        stock_info, bs_data, is_data, cf_data, stock_hist = fetch_financial_data_multi_year(ticker, frequency_value)

    if stock_info and get_safe_value(stock_info, 'shortName'):
        st.header(f"{get_safe_value(stock_info, 'shortName', ticker)} ({ticker}) - {selected_frequency_label} Data")
        
        ratios_df = pd.DataFrame()
        if bs_data is not None and is_data is not None and not bs_data.empty and not is_data.empty:
             with st.spinner("Calculating financial ratios..."):
                ratios_df = calculate_ratios_multi_year(bs_data, is_data, cf_data, stock_info)
        
        tabs_list_names = ["🔑 Key Metrics & Stock Info", "📜 Historical Financials", "📊 Financial Ratios (YoY)"]
        if ratios_df is not None and not ratios_df.empty:
            tabs_list_names.append("🤖 AI Financial Summary")
        tabs_list_names.append("💸 Loan Impact & Risk")
        
        created_tabs = st.tabs(tabs_list_names)
        tab_idx = 0 

        with created_tabs[tab_idx]: # Key Metrics & Stock Info
            tab_idx += 1
            st.subheader("Company Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Sector", get_safe_value(stock_info, 'sector', 'N/A'))
            col2.metric("Industry", get_safe_value(stock_info, 'industry', 'N/A'))
            col3.metric("Country", get_safe_value(stock_info, 'country', 'N/A'))
            website = get_safe_value(stock_info, 'website')
            if website: st.markdown(f"**Website:** [{website}]({website})")
            summary_text_val = get_safe_value(stock_info, 'longBusinessSummary', 'N/A')
            st.markdown(f"**Business Summary:** {summary_text_val[:500]}{'...' if len(summary_text_val) > 500 else ''}")

            st.subheader("Current Stock Performance")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            current_price = get_safe_value(stock_info, ['currentPrice', 'regularMarketPrice'])
            col_s1.metric("Current Price", format_value(current_price, 'currency_precise'))
            col_s2.metric("Market Cap", format_value(get_safe_value(stock_info, 'marketCap'), 'currency'))
            day_low = format_value(get_safe_value(stock_info, 'regularMarketDayLow'), 'currency_precise')
            day_high = format_value(get_safe_value(stock_info, 'regularMarketDayHigh'), 'currency_precise')
            col_s3.metric("Day Low / High", f"{day_low} / {day_high}")
            w52_low = format_value(get_safe_value(stock_info, 'fiftyTwoWeekLow'), 'currency_precise')
            w52_high = format_value(get_safe_value(stock_info, 'fiftyTwoWeekHigh'), 'currency_precise')
            col_s4.metric("52 Week Low / High", f"{w52_low} / {w52_high}")

            st.subheader("Valuation & Dividends")
            col_v1, col_v2, col_v3, col_v4 = st.columns(4)
            col_v1.metric("Trailing P/E", format_value(get_safe_value(stock_info, 'trailingPE'), 'ratio'))
            col_v2.metric("Forward P/E", format_value(get_safe_value(stock_info, 'forwardPE'), 'ratio'))
            col_v3.metric("Trailing EPS", format_value(get_safe_value(stock_info, 'trailingEps'), 'currency_precise'))
            col_v4.metric("Forward EPS", format_value(get_safe_value(stock_info, 'forwardEps'), 'currency_precise'))

            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            raw_div_yield = get_safe_value(stock_info, 'dividendYield')
            formatted_div_yield_display = f"{raw_div_yield:.2f}%" if pd.notna(raw_div_yield) else "N/A"
            col_d1.metric("Dividend Yield", formatted_div_yield_display)
            col_d2.metric("Payout Ratio", format_value(get_safe_value(stock_info, 'payoutRatio'), 'percent'))
            col_d3.metric("Beta", format_value(get_safe_value(stock_info, 'beta'), 'ratio'))
            col_d4.metric("Shares Outstanding", format_value(get_safe_value(stock_info, 'sharesOutstanding'), 'number'))

        with created_tabs[tab_idx]: # Historical Financials
            tab_idx += 1
            st.subheader("Balance Sheet (Key Items)")
            if bs_data is not None and not bs_data.empty:
                st.dataframe(bs_data.head(20).style.format(lambda x: format_value(x, 'currency', 'N/A') if isinstance(x, (int, float)) else str(x), na_rep="N/A"))
            else: st.info("Balance Sheet data not available.")
            st.subheader("Income Statement (Key Items)")
            if is_data is not None and not is_data.empty:
                st.dataframe(is_data.head(20).style.format(lambda x: format_value(x, 'currency', 'N/A') if isinstance(x, (int, float)) else str(x), na_rep="N/A"))
            else: st.info("Income Statement data not available.")
            st.subheader("Cash Flow Statement (Key Items)")
            if cf_data is not None and not cf_data.empty:
                st.dataframe(cf_data.head(20).style.format(lambda x: format_value(x, 'currency', 'N/A') if isinstance(x, (int, float)) else str(x), na_rep="N/A"))
            else: st.info("Cash Flow Statement data not available.")
        
        with created_tabs[tab_idx]: # Financial Ratios (YoY)
            tab_idx += 1
            st.subheader("Key Financial Ratios Over Time")
            if ratios_df is not None and not ratios_df.empty:
                st.dataframe(ratios_df.style.format("{:.2f}", na_rep="N/A"))
                st.subheader("Ratio Trends")
                default_ratios = ['Current Ratio', 'Debt-to-Equity', 'Net Profit Margin', 'ROE (Return on Equity)', 'EPS (Diluted)']
                available_ratios = [r for r in default_ratios if r in ratios_df.index]
                selected_ratios = st.multiselect("Select ratios to plot:", options=list(ratios_df.index), default=available_ratios, key="ratio_plot_multiselect")
                if selected_ratios:
                    plot_df = ratios_df.loc[selected_ratios].T.reset_index().rename(columns={'index': 'Period'})
                    try:
                        plot_df['Period'] = pd.to_datetime(plot_df['Period'])
                        plot_df = plot_df.sort_values(by='Period')
                    except Exception: st.warning("Could not parse period dates for optimal chart sorting.")
                    for ratio_name_plot in selected_ratios:
                        if ratio_name_plot in plot_df.columns:
                            fig = px.line(plot_df, x='Period', y=ratio_name_plot, title=f'{ratio_name_plot} Trend', markers=True)
                            if any(p_term in ratio_name_plot for p_term in ["Margin", "ROE", "ROA", "Yield"]) and "EPS" not in ratio_name_plot :
                                fig.update_layout(yaxis_tickformat='.2%') 
                            fig.update_layout(xaxis_title=f"{selected_frequency_label} Period Ending")
                            st.plotly_chart(fig, use_container_width=True)
            else: st.info("Ratio data not available for plotting.")

        if "🤖 AI Financial Summary" in tabs_list_names:
            with created_tabs[tab_idx]:
                tab_idx += 1
                st.subheader("🤖 AI Financial Health Summary (via Google Gemini)")
                st.markdown("""
                **Disclaimer:** This summary is generated by an AI model (Google Gemini) based on the financial data provided. 
                It is for informational purposes only and should not be considered financial advice. 
                Always conduct your own thorough research or consult with a qualified financial advisor.
                AI models can make mistakes or misinterpret data.
                """)
                if ratios_df is not None and not ratios_df.empty:
                    latest_ratios = ratios_df[ratios_df.columns[0]]
                    with st.spinner("🧠 Generating AI summary with Gemini... please wait."):
                        ai_summary = generate_ai_summary_gemini(stock_info, latest_ratios)
                        st.markdown(ai_summary)
                else: st.info("Financial ratios needed to generate AI summary. Check if data was fetched correctly.")
        
        with created_tabs[tab_idx]: # Loan Impact & Risk
            st.subheader("Loan Impact Simulation (Based on Latest Data)")
            st.sidebar.subheader("Loan Scenario Details") 
            loan_amount = st.sidebar.number_input("Loan Amount ($):", min_value=0, value=100000, step=10000, key="loan_amount_sidebar_input")
            interest_rate_perc = st.sidebar.slider("Annual Interest Rate (%):", 0.0, 25.0, 7.0, 0.1, key="interest_rate_sidebar_slider")
            interest_rate_dec = interest_rate_perc / 100.0

            loan_sim_inputs = {}
            if isinstance(is_data, pd.DataFrame) and not is_data.empty and len(is_data.columns) > 0: # Corrected check
                latest_is_col_name = is_data.columns[0]
                is_latest_period_data = is_data[latest_is_col_name]
                loan_sim_inputs['EBIT'] = get_safe_value(is_latest_period_data, ['Operating Income', 'Ebit', 'Earnings Before Interest And Taxes'])
                interest_val = get_safe_value(is_latest_period_data, ['Interest Expense'])
                loan_sim_inputs['Interest Expense'] = abs(interest_val) if pd.notna(interest_val) else 0.0
            else:
                st.warning("Income statement data for the latest period is unavailable for loan simulation.")

            if loan_amount > 0 and pd.notna(get_safe_value(loan_sim_inputs, 'EBIT')):
                loan_results = simulate_loan_impact(loan_sim_inputs, loan_amount, interest_rate_dec)
                col_l1, col_l2, col_l3, col_l4 = st.columns(4)
                col_l1.metric("New Annual Interest", format_value(loan_results.get('New Annual Interest'), 'currency'))
                col_l2.metric("Pro-Forma Total Int. Exp.", format_value(loan_results.get('Pro-Forma Interest Expense'), 'currency'))
                pro_forma_icr = loan_results.get('Pro-Forma Interest Coverage Ratio')
                col_l3.metric("Pro-Forma Int. Coverage", f"{pro_forma_icr:.2f}x" if pd.notna(pro_forma_icr) else "N/A")
                accept_cov = loan_results.get('Acceptable Coverage (>1.25x)?')
                col_l4.metric("Coverage Acceptable (>1.25x)?", "✅ Yes" if accept_cov is True else ("❌ No" if accept_cov is False else "N/A"))
            elif loan_amount > 0: 
                st.warning("Could not perform loan simulation. Ensure EBIT data is available for the latest period.")
            else: 
                st.info("Enter a loan amount > 0 in the sidebar to simulate impact.")

            st.subheader("Overall Risk Assessment (Based on Latest Ratios)")
            if ratios_df is not None and not ratios_df.empty:
                latest_ratios_series = ratios_df[ratios_df.columns[0]]
                risk_level, risk_color, risk_summary = assess_risk_from_ratios(latest_ratios_series)
                st.markdown(f"**<font color='{risk_color}'>{risk_level}</font>**", unsafe_allow_html=True)
                st.caption(risk_summary)
            else: st.info("Risk assessment requires calculated ratios. Check if data was fetched correctly.")

    elif ticker and not (stock_info and get_safe_value(stock_info, 'shortName')):
        st.error(f"Could not retrieve any valid data for the ticker: {ticker}. Please ensure it's a correct and active stock ticker, or try again later.")
    elif not ticker and analyze_button: 
        st.error("Please enter a stock ticker in the sidebar.")
    elif not ticker: 
        st.info("👋 Welcome! Please enter a stock ticker in the sidebar and click 'Analyze Company'.")
