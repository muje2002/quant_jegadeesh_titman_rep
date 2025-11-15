import pandas as pd
import numpy as np
import zipfile
import os
import time

def load_and_prepare_data(data_dir='data', zip_name='CRSPm19652024.zip', file_name='CRSPm19652024',
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

    df['date'] = pd.to_datetime(df['date'])
    
    # Filter by date
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    
    # Add year and month columns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Keep necessary columns.
    # Note: CUSIP is loaded but we sort by PERMNO as it's a more stable identifier.
    df = df[['permno', 'cusip', 'date', 'year', 'month', 'ret']].copy()
    df.dropna(subset=['ret'], inplace=True)
    
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
        
        formation_data.dropna(subset=[mom_col], inplace=True)
        
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
            
        for k in range(0, K):
            holding_date = unique_dates[i + k]
            
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
        print("Could not generate any returns.")
        return

    returns_df = pd.DataFrame(all_returns)
    monthly_returns = returns_df.groupby('holding_date')['strategy_ret'].mean()
    
    end_time = time.time()
    print(f"Calculation finished in {end_time - start_time:.2f} seconds.")
    
    avg_return = monthly_returns.mean()
    std_dev = monthly_returns.std()
    n_months = len(monthly_returns)
    t_stat = (avg_return / (std_dev / np.sqrt(n_months))) if std_dev > 0 else np.inf
    
    print("\n" + "="*40)
    print(f"Momentum Strategy Results (J={J}, K={K})")
    print("="*40)
    print(f"Period: {monthly_returns.index.min().strftime('%Y-%m')} to {monthly_returns.index.max().strftime('%Y-%m')}")
    print(f"Number of Months in Sample: {n_months}")
    print(f"Avg. Monthly Return (Winner-Loser): {avg_return:.4%}")
    print(f"Standard Deviation of Returns: {std_dev:.4%}")
    print(f"T-statistic: {t_stat:.2f}")
    print("="*40)


if __name__ == '__main__':
    J_PERIOD = [3,6,9,12]
    K_PERIOD = [3,6,9,12]
    
    START_DATE = '1965-01-01'
    END_DATE = '1989-12-31'
    
    crsp_data = load_and_prepare_data(start_date=START_DATE, end_date=END_DATE)
    for j in J_PERIOD:
        crsp_data_with_mom = calculate_momentum(crsp_data, J=j)
        for k in K_PERIOD:
            form_portfolios_and_calculate_returns(crsp_data_with_mom, J=j, K=k)