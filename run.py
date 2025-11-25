import pandas as pd
import numpy as np
import zipfile
import os
import time

def load_and_prepare_data(data_dir='data', zip_name='CRSPm19652024.zip', file_name='CRSP_v3.csv',
                          start_date='1965-01-01', end_date='1989-12-31'):
    """Loads and prepares the CRSP data for the specified date range."""
    print("Loading and preparing data...")
    zip_path = os.path.join(data_dir, zip_name)
    file_path = os.path.join(data_dir, file_name)

    if not os.path.exists(file_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"Extracted '{file_name}'.")

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception:
        df = pd.read_csv(file_path, low_memory=False, encoding='latin1')

    df.columns = df.columns.str.lower()

    df['date'] = pd.to_datetime(df['date'])
    
    # Filter by date
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    # Handle special values in return column as per user request
    df['ret'] = df['ret'].replace('C', 0)
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    
    # Drop rows where key data is missing BEFORE filtering
    df.dropna(subset=['ret', 'prc'], inplace=True)
    
    # Filter by exchange code and share code for ordinary common shares
    df = df[df['exchcd'].isin([1, 2])]
    df = df[df['shrcd'].isin([10, 11])]

    # Add year and month columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Keep necessary columns.
    df = df[['permno', 'cusip', 'date', 'year', 'month', 'ret', 'prc', 'shrcd', 'exchcd']].copy()
    
    # Sort values by permno, year, and month to ensure correct time-series calculations.
    df.sort_values(by=['permno', 'year', 'month'], inplace=True)
    
    return df

def calculate_momentum(df, J=12):
    """Calculates past J-month cumulative returns for each stock."""
    print(f"Calculating {J}-month momentum...")
    df['log_ret'] = np.log(1 + df['ret'])
    
    df[f'mom_{J}'] = df.groupby('permno')['log_ret'].rolling(window=J, min_periods=J).sum().shift(1).reset_index(level=0, drop=True)
    
    return df

def form_portfolios_and_calculate_returns(df, J=12, K=3):
    """Forms momentum portfolios and calculates returns for a J/K strategy with overlapping portfolios."""
    
    print(f"Running {J}-month formation / {K}-month holding period strategy...")
    start_time = time.time()
    
    mom_col = f'mom_{J}'
    
    unique_dates = sorted(df['date'].unique())
    
    all_returns = []

    for i in range(J, len(unique_dates) - K + 1):
        formation_date = unique_dates[i]
        
        formation_data = df[df['date'] == formation_date].copy()
        
        formation_data.dropna(subset=[mom_col, 'prc'], inplace=True)
        
        # Filter out stocks with price < $5 at formation date
        formation_data = formation_data[formation_data['prc'].abs() >= 5]
        
        if len(formation_data) < 10:
            continue
            
        try:
            formation_data['decile'] = pd.qcut(formation_data[mom_col], 10, labels=False)
        except ValueError:
            continue

        winner_perms = formation_data[formation_data['decile'] == 9]['permno']
        loser_perms = formation_data[formation_data['decile'] == 0]['permno']
        
        if winner_perms.empty or loser_perms.empty:
            continue
            
        for k_offset in range(0, K):
            holding_date = unique_dates[i + k_offset]
            
            holding_data = df[df['date'] == holding_date]
            
            winner_ret = holding_data[holding_data['permno'].isin(winner_perms)]['ret'].mean()
            loser_ret = holding_data[holding_data['permno'].isin(loser_perms)]['ret'].mean()
            
            if not (np.isnan(winner_ret) or np.isnan(loser_ret)):
                all_returns.append({
                    'formation_date': formation_date,
                    'holding_date': holding_date,
                    'strategy_ret': winner_ret - loser_ret
                })

    if not all_returns:
        print(f"Could not generate any returns for J={J}, K={K}.")
        return None, None, None

    returns_df = pd.DataFrame(all_returns)
    monthly_returns = returns_df.groupby('holding_date')['strategy_ret'].mean()
    
    end_time = time.time()
    print(f"  - Calculation for J={J}, K={K} finished in {end_time - start_time:.2f} seconds.")
    
    avg_return = monthly_returns.mean()
    std_dev = monthly_returns.std()
    n_months = len(monthly_returns)
    t_stat = (avg_return / (std_dev / np.sqrt(n_months))) if std_dev > 0 else np.inf
    
    return avg_return, std_dev, t_stat

def display_results_table(results, paper_avg, paper_t_stat):
    """Prints a formatted table comparing script results with paper results."""
    
    print("\n\n" + "="*80)
    print("Final Results: Comparison with Jegadeesh & Titman (1993) Table 1, Panel A")
    print("="*80)
    
    header = f"{'J / K':<7} | {'My Code AVG (%)':<17} | {'Paper AVG (%)':<15} | {'My Code T-stat':<16} | {'Paper T-stat':<14} | {'My Code STD (%)':<15}"
    print(header)
    print("-"*len(header))

    sorted_keys = sorted(results.keys())

    for j, k in sorted_keys:
        avg, std, t_stat = results[(j,k)]
        
        # Safely get paper results
        paper_avg_val = paper_avg.get(j, {}).get(k, 'N/A')
        paper_t_stat_val = paper_t_stat.get(j, {}).get(k, 'N/A')

        avg_str = f"{avg*100:.2f}"
        std_str = f"{std*100:.2f}"
        t_stat_str = f"{t_stat:.2f}"

        paper_avg_str = f"{paper_avg_val:.2f}" if isinstance(paper_avg_val, float) else paper_avg_val
        paper_t_stat_str = f"{paper_t_stat_val:.2f}" if isinstance(paper_t_stat_val, float) else paper_t_stat_val

        row = f"{f'{j} / {k}':<7} | {avg_str:<17} | {paper_avg_str:<15} | {t_stat_str:<16} | {paper_t_stat_str:<14} | {std_str:<15}"
        print(row)
        
    print("="*80)


if __name__ == '__main__':
    J_PERIODS = [3, 6, 9, 12]
    K_PERIODS = [3, 6, 9, 12]
    
    START_DATE = '1965-01-01'
    END_DATE = '1989-12-31'
    
    # Jegadeesh & Titman (1993), Table 1, Panel A
    PAPER_AVG_RETURNS = {
        3: {3: 0.48, 6: 0.64, 9: 0.70, 12: 0.77},
        6: {3: 0.77, 6: 0.95, 9: 1.02, 12: 0.93},
        9: {3: 1.04, 6: 1.18, 9: 1.11, 12: 0.86},
        12: {3: 1.31, 6: 1.22, 9: 1.10, 12: 0.83},
    }
    
    PAPER_T_STATS = {
        3: {3: 2.08, 6: 3.01, 9: 3.59, 12: 4.19},
        6: {3: 2.76, 6: 3.73, 9: 4.24, 12: 4.02},
        9: {3: 3.74, 6: 4.54, 9: 4.41, 12: 3.53},
        12: {3: 4.43, 6: 4.49, 9: 4.14, 12: 3.19},
    }

    crsp_data = load_and_prepare_data(start_date=START_DATE, end_date=END_DATE)
    
    all_results = {}
    
    for j in J_PERIODS:
        crsp_data_with_mom = calculate_momentum(crsp_data, J=j)
        for k in K_PERIODS:
            avg, std, t_stat = form_portfolios_and_calculate_returns(crsp_data_with_mom, J=j, K=k)
            if avg is not None:
                all_results[(j, k)] = (avg, std, t_stat)
                
    display_results_table(all_results, PAPER_AVG_RETURNS, PAPER_T_STATS)