import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import os

# Define the ETF symbols you want to analyze
etf_symbols = [
    'SPX',    # S&P 500 Index
    'SPY',    # SPDR S&P 500 ETF
    'SPXL',   # Direxion Daily S&P 500 Bull 3X
    'SPXS',   # Direxion Daily S&P 500 Bear 3X
    'UPRO',   # ProShares UltraPro S&P500 (3x Bull)
    'SPXU',   # ProShares UltraPro Short S&P500 (3x Bear)
    'IWM',    # iShares Russell 2000
    'QQQ',    # Invesco QQQ (Nasdaq 100)
    'TQQQ',   # ProShares UltraPro QQQ (3x Bull)
    'SQQQ',   # ProShares UltraPro Short QQQ (3x Bear)
    'TMF',    # Direxion Daily 20+ Year Treasury Bull 3X
    'TMV',    # Direxion Daily 20+ Year Treasury Bear 3X
    'YINN',   # Direxion Daily FTSE China Bull 3X
    'YANG',   # Direxion Daily FTSE China Bear 3X
    'SOXL',   # Direxion Daily Semiconductor Bull 3X
    'SOXS',   # Direxion Daily Semiconductor Bear 3X
    'TNA',    # Direxion Daily Small Cap Bull 3X
    'TZA',    # Direxion Daily Small Cap Bear 3X
    'TECL',   # Direxion Daily Technology Bull 3X
    'TECS',   # Direxion Daily Technology Bear 3X
    'RETL',   # Direxion Daily Retail Bull 3X
    'TSLL',   # Direxion Daily TSLA Bull 1.5X
    'NVDU',   # NVIDIA 2X Long
    'AGG',    # iShares Core U.S. Aggregate Bond
    'VTI'     # Vanguard Total Stock Market
]


def get_etf_data(symbols, start=None, end=None):
    """
    Get ETF data - load from local files if available, otherwise download and save
    If start/end dates not specified, downloads all available historical data
    """
    data = {}
    for symbol in symbols:
        filename = f"data/{symbol}_data.csv"
        
        try:
            # Try to load existing data first
            if os.path.exists(filename):
                df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
                
                if len(df.index) > 0:
                    # Check if we have data for the requested date range
                    if start is not None and end is not None:
                        if isinstance(start, str):
                            start = pd.to_datetime(start)
                        if isinstance(end, str):
                            end = pd.to_datetime(end)
                            
                        if df.index[0] <= start and df.index[-1] >= end:
                            data[symbol] = df
                            print(f"Loaded existing data for {symbol}")
                            continue
            
            # Download if file doesn't exist or date range is insufficient
            print(f"Downloading new data for {symbol}...")
            
            # Only pass start/end if specified
            download_args = {'multi_level_index': False}
            if start is not None:
                download_args['start'] = start
            if end is not None:
                download_args['end'] = end
                
            etf = yf.download(symbol, **download_args)
            
            if len(etf) > 0:
                data[symbol] = etf
                
                # Save downloaded data
                etf.to_csv(filename)
                print(f"Saved data for {symbol}")
            else:
                print(f"No data available for {symbol}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    return data


# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

def analyze_etf_pair_strategy(etf_data, etf1_symbol, etf2_symbol, etf1_amount, etf2_amount, 
                            etf1_short_fee=0.0, etf2_short_fee=0.0,
                            etf1_div_rate=0.0, etf2_div_rate=0.0,
                            start_date=None, end_date=None, plot=True):
    """
    Analyze a pair trading strategy with specified amounts for two ETFs over a given time period
    
    Parameters:
    etf_data (dict): Dictionary containing ETF price data
    etf1_symbol (str): Symbol for first ETF
    etf2_symbol (str): Symbol for second ETF  
    etf1_amount (float): Dollar amount to short/long first ETF (negative for short)
    etf2_amount (float): Dollar amount to short/long second ETF (negative for short)
    etf1_short_fee (float): Annual short fee rate for first ETF (default 0.0)
    etf2_short_fee (float): Annual short fee rate for second ETF (default 0.0)
    etf1_div_rate (float): Annual dividend rate for first ETF (default 0.0)
    etf2_div_rate (float): Annual dividend rate for second ETF (default 0.0)
    start_date (str): Start date in 'YYYY-MM-DD' format (optional)
    end_date (str): End date in 'YYYY-MM-DD' format (optional)
    plot (bool): Whether to generate and save plots (default True)  
    Returns:
    dict: Dictionary containing performance metrics and daily values
    """
    # Get data for both ETFs
    etf1_df = etf_data[etf1_symbol].copy()
    etf2_df = etf_data[etf2_symbol].copy()
    # Handle date range filtering
    start_dt = pd.to_datetime(start_date) if start_date else etf1_df.index[0]
    end_dt = pd.to_datetime(end_date) if end_date else etf1_df.index[-1]    
    # Find closest available dates
    closest_start = etf1_df.index[etf1_df.index.searchsorted(start_dt)]
    closest_end = etf1_df.index[etf1_df.index.searchsorted(end_dt)]    
    if closest_start != start_dt:
        print(f"Using closest available start date: {closest_start.strftime('%Y-%m-%d')}")
    if closest_end != end_dt:
        print(f"Using closest available end date: {closest_end.strftime('%Y-%m-%d')}")
    
    # Filter data
    date_mask = (etf1_df.index >= closest_start) & (etf1_df.index <= closest_end)
    etf1_df = etf1_df[date_mask]
    etf2_df = etf2_df[date_mask]
    # Calculate shares and daily values
    etf1_shares = etf1_amount / etf1_df.iloc[0]['Close']
    etf2_shares = etf2_amount / etf2_df.iloc[0]['Close']
    etf1_daily_value = etf1_shares * etf1_df['Close']
    etf2_daily_value = etf2_shares * etf2_df['Close']
    total_daily_value = etf1_daily_value + etf2_daily_value
    
    # Calculate daily costs/gains from short fees and dividends
    days_elapsed = [(date - closest_start).days for date in etf1_df.index]
    
    # ETF1 costs/gains
    etf1_daily_costs = pd.Series(0.0, index=etf1_df.index)
    if etf1_amount < 0:  # Short position
        etf1_daily_costs += abs(etf1_daily_value) * (etf1_short_fee/365)  # Short fee
        etf1_daily_costs += abs(etf1_daily_value) * (etf1_div_rate/365)   # Dividend payment
    else:  # Long position
        etf1_daily_costs -= abs(etf1_daily_value) * (etf1_div_rate/365)   # Dividend gain
    
    # ETF2 costs/gains
    etf2_daily_costs = pd.Series(0.0, index=etf2_df.index)
    if etf2_amount < 0:  # Short position
        etf2_daily_costs += abs(etf2_daily_value) * (etf2_short_fee/365)  # Short fee
        etf2_daily_costs += abs(etf2_daily_value) * (etf2_div_rate/365)   # Dividend payment
    else:  # Long position
        etf2_daily_costs -= abs(etf2_daily_value) * (etf2_div_rate/365)   # Dividend gain
    
    total_daily_costs = etf1_daily_costs + etf2_daily_costs
    cumulative_costs = total_daily_costs.cumsum()
    
    # Adjust total value for costs
    total_daily_value_with_costs = total_daily_value - cumulative_costs
    
    # Calculate key metrics
    initial_investment = abs(etf1_amount) + abs(etf2_amount)
    initial_value = total_daily_value.iloc[0]
    final_value = total_daily_value_with_costs.iloc[-1]
    total_days = (etf1_df.index[-1] - etf1_df.index[0]).days
    etf1_final_value = etf1_daily_value.iloc[-1]
    etf2_final_value = etf2_daily_value.iloc[-1]
    
    # Calculate returns
    etf1_return = (etf1_final_value - etf1_amount) / abs(etf1_amount)
    etf2_return = (etf2_final_value - etf2_amount) / abs(etf2_amount)
    total_return = (final_value - initial_value) / initial_investment
    annual_return = (1 + total_return) ** (365/total_days) - 1
    
    # Calculate drawdown using cost-adjusted values
    portfolio_peaks = total_daily_value_with_costs.expanding(min_periods=1).max()
    drawdowns = (total_daily_value_with_costs - portfolio_peaks) / portfolio_peaks
    max_drawdown = drawdowns.max()
    
    if plot:
        # Generate plots
        fig = plt.figure(figsize=(12, 15))
        
        # Initial portfolio structure
        ax1 = fig.add_subplot(321)
        initial_labels = [
            f'{etf1_symbol}\n{"Long" if etf1_amount > 0 else "Short"} ${abs(etf1_amount):,.0f}\n'
            f'{etf1_shares:.1f} shares @ ${etf1_df.iloc[0]["Close"]:.2f}\n'
            f'Weight: {abs(etf1_amount)/initial_investment*100:.1f}%',
            
            f'{etf2_symbol}\n{"Long" if etf2_amount > 0 else "Short"} ${abs(etf2_amount):,.0f}\n'
            f'{etf2_shares:.1f} shares @ ${etf2_df.iloc[0]["Close"]:.2f}\n'
            f'Weight: {abs(etf2_amount)/initial_investment*100:.1f}%'
        ]
        ax1.pie([abs(etf1_amount), abs(etf2_amount)], labels=initial_labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Initial Portfolio Structure\nTotal Investment: ${initial_investment:,.0f}')
        
        # Final portfolio structure  
        ax2 = fig.add_subplot(322)
        final_labels = [
            f'{etf1_symbol}\n{"Long" if etf1_amount > 0 else "Short"} ${abs(etf1_final_value):,.0f}\n'
            f'Return: {etf1_return*100:.1f}%\nP&L: ${etf1_final_value - etf1_amount:,.0f}\n'
            f'Weight: {abs(etf1_final_value/final_value)*100:.1f}%',
            
            f'{etf2_symbol}\n{"Long" if etf2_amount > 0 else "Short"} ${abs(etf2_final_value):,.0f}\n'
            f'Return: {etf2_return*100:.1f}%\nP&L: ${etf2_final_value - etf2_amount:,.0f}\n'
            f'Weight: {abs(etf2_final_value/final_value)*100:.1f}%'
        ]
        ax2.pie([abs(etf1_final_value), abs(etf2_final_value)], labels=final_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Final Portfolio Structure\nTotal Value: ${final_value:,.0f}')
        
        # Portfolio performance
        ax3 = fig.add_subplot(312)
        ax3.plot(etf1_daily_value.index, etf1_daily_value.values, 
                label=f'{etf1_symbol} ({"Long" if etf1_amount > 0 else "Short"}) (Return: {etf1_return*100:.1f}%, Annual: {((1+etf1_return)**(365/total_days)-1)*100:.1f}%)')
        ax3.plot(etf2_daily_value.index, etf2_daily_value.values,
                label=f'{etf2_symbol} ({"Long" if etf2_amount > 0 else "Short"}) (Return: {etf2_return*100:.1f}%, Annual: {((1+etf2_return)**(365/total_days)-1)*100:.1f}%)')
        
        # Calculate total return before costs
        total_return_before_costs = (total_daily_value.iloc[-1] - initial_value) / initial_investment
        annual_return_before_costs = (1 + total_return_before_costs) ** (365/total_days) - 1
        
        ax3.plot(total_daily_value.index, total_daily_value.values,
                label=f'Total (Before Costs) (Return: {total_return_before_costs*100:.1f}%, Annual: {annual_return_before_costs*100:.1f}%)', linewidth=2)
        
        # Calculate costs ratio
        costs_ratio = cumulative_costs.iloc[-1] / initial_investment
        annual_costs_ratio = (1 + costs_ratio) ** (365/total_days) - 1
        
        ax3.plot(cumulative_costs.index, -cumulative_costs.values,
                label=f'Cumulative Costs (Ratio: {costs_ratio*100:.1f}%, Annual: {annual_costs_ratio*100:.1f}%)', linewidth=2, linestyle='--')
        ax3.plot(total_daily_value_with_costs.index, total_daily_value_with_costs.values,
                label=f'Total After Costs (Return: {total_return*100:.1f}%, Annual: {annual_return*100:.1f}%)', linewidth=2)
        period_str = f'{closest_start.strftime("%Y-%m-%d")} to {closest_end.strftime("%Y-%m-%d")}'
        ax3.set_title(f'Portfolio Performance Over Time\n{period_str}\n'
                    f'Initial: {initial_value:,.0f} -> Final: {final_value:,.0f}')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Value ($)')
        ax3.legend()
        ax3.grid(True)
        
        # Drawdown analysis
        ax4 = fig.add_subplot(313)
        ax4.plot(drawdowns.index, drawdowns.values * 100, color='red', 
                label=f'Max Drawdown: {max_drawdown*100:.1f}%')
        ax4.fill_between(drawdowns.index, drawdowns.values * 100, 0, color='red', alpha=0.1)
        ax4.set_title('Portfolio Drawdown Analysis (After Costs)')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Drawdown (%)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        filename = (f'{etf1_symbol}_{"short" if etf1_amount < 0 else "long"}_{abs(etf1_amount):.0f}_'
                f'{etf2_symbol}_{"short" if etf2_amount < 0 else "long"}_{abs(etf2_amount):.0f}_'
                f'{closest_start.strftime("%Y-%m-%d")}_to_{closest_end.strftime("%Y-%m-%d")}.png')
        plt.savefig(filename)
        plt.close()
        print(f"Successfully saved plot: {filename}")
    
    return {
        'initial_investment': initial_investment,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'daily_values': total_daily_value,
        'daily_values_with_costs': total_daily_value_with_costs,
        'cumulative_costs': cumulative_costs,
        'etf1_daily_values': etf1_daily_value,
        'etf2_daily_values': etf2_daily_value,
        'etf1_return': etf1_return,
        'etf2_return': etf2_return,
        'start_date': closest_start,
        'end_date': closest_end,
        'total_days': total_days
    }



# Define fee rates as global dict
FEE_RATES = {
    'YINN': {
        'short_fee': 0.006,  # 0.6% annual short fee
        'div_rate': 0.02,    # 2% annual dividend rate
    },
    'YANG': {
        'short_fee': 0.09,   # 9% annual short fee
        'div_rate': 0.0238,  # 2.38% annual dividend rate
    },
    'QQQ': {
        'short_fee': 0.0,    # No short fee for long position
        'div_rate': 0.015,   # 1.5% annual dividend rate
    },
    'TQQQ': {
        'short_fee': 0.02,   # 2% annual short fee
        'div_rate': 0.0,     # No dividend for TQQQ
    }
}


etf_data = get_etf_data(['YINN', 'YANG'], None, None)

result = analyze_etf_pair_strategy(etf_data, 'YINN', 'YANG', -1000, -1000, 
                            etf1_short_fee=0.006, etf2_short_fee=0.09,
                            etf1_div_rate=0.02, etf2_div_rate=0.0238,
                            start_date=None, end_date=None, plot=True)





# Update ETF symbols to include QQQ and TQQQ
etf_symbols = [
    'QQQ',    # Invesco QQQ (Nasdaq 100)
    'TQQQ',   # ProShares UltraPro QQQ (3x Bull)
]

# Get data for QQQ/TQQQ pair
etf_data = get_etf_data(etf_symbols, start_date, end_date)

# Test QQQ long / TQQQ short strategy for different periods
for start_date, end_date in periods:
    analyze_etf_pair_strategy(
        etf_data,
        'QQQ', 'TQQQ', 
        etf1_amount=30000,   # Long $30k of QQQ
        etf2_amount=-10000,  # Short $10k of TQQQ
        etf1_short_fee=0.0,   # No short fee for long position
        etf2_short_fee=0.02,  # 2% annual short fee
        etf1_div_rate=0.015,  # 1.5% annual dividend rate
        etf2_div_rate=0.0,    # No dividend for TQQQ
        start_date=start_date,
        end_date=end_date
    )
