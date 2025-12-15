"""
Stock Market Predictor - Modern Professional GUI
Fixed stock row display with proper grid layout
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from recommend import EnhancedRecommendationEngine


class MetricCard(tk.Frame):
    """Card for displaying metrics"""
    def __init__(self, parent, icon, label, value="--", color="#4caf50", **kwargs):
        super().__init__(parent, bg="#3a3a3a", relief='flat', **kwargs)
        
        tk.Label(self, text=icon, font=('Segoe UI', 28), bg="#3a3a3a", fg=color).pack(pady=(15, 5))
        self.value_label = tk.Label(self, text=value, font=('Segoe UI', 20, 'bold'), bg="#3a3a3a", fg="#ffffff")
        self.value_label.pack(pady=(5, 5))
        tk.Label(self, text=label, font=('Segoe UI', 10), bg="#3a3a3a", fg="#b0b0b0").pack(pady=(0, 15))
    
    def update_value(self, value, color=None):
        self.value_label.config(text=value)
        if color:
            self.value_label.config(fg=color)


class StockRow(tk.Frame):
    """Professional stock recommendation row with grid layout"""
    def __init__(self, parent, rank, ticker, company, shares, amount, weight, return_pct, risk, **kwargs):
        super().__init__(parent, bg="#3a3a3a", relief='flat', height=70, **kwargs)
        self.pack_propagate(False)
        
        # Configure grid columns with specific weights
        self.grid_columnconfigure(0, weight=0, minsize=60)   # Rank
        self.grid_columnconfigure(1, weight=0, minsize=280)  # Ticker/Company
        self.grid_columnconfigure(2, weight=0, minsize=120)  # Shares
        self.grid_columnconfigure(3, weight=0, minsize=140)  # Amount
        self.grid_columnconfigure(4, weight=0, minsize=120)  # Return
        self.grid_columnconfigure(5, weight=0, minsize=100)  # Risk
        
        # Rank badge
        rank_container = tk.Frame(self, bg="#3a3a3a")
        rank_container.grid(row=0, column=0, sticky='nsew', padx=(15, 10))
        
        rank_badge = tk.Frame(rank_container, bg="#0d7377", width=45, height=45)
        rank_badge.place(relx=0.5, rely=0.5, anchor='center')
        rank_badge.pack_propagate(False)
        tk.Label(rank_badge, text=str(rank), font=('Segoe UI', 18, 'bold'), bg="#0d7377", fg='#ffffff').place(relx=0.5, rely=0.5, anchor='center')
        
        # Ticker and Company
        info_container = tk.Frame(self, bg="#3a3a3a")
        info_container.grid(row=0, column=1, sticky='w', padx=10)
        
        tk.Label(info_container, text=ticker, font=('Segoe UI', 13, 'bold'), bg="#3a3a3a", fg='#14ffec', anchor='w').pack(anchor='w', pady=(15, 2))
        company_short = company[:32] + '...' if len(company) > 35 else company
        tk.Label(info_container, text=company_short, font=('Segoe UI', 9), bg="#3a3a3a", fg='#b0b0b0', anchor='w').pack(anchor='w')
        
        # Shares
        shares_container = tk.Frame(self, bg="#3a3a3a")
        shares_container.grid(row=0, column=2, sticky='ew', padx=10)
        
        shares_text = f"{shares:.4f}" if isinstance(shares, float) and shares != int(shares) else f"{int(shares)}"
        tk.Label(shares_container, text=shares_text, font=('Segoe UI', 12, 'bold'), bg="#3a3a3a", fg='#ffffff').pack(pady=(18, 2))
        tk.Label(shares_container, text="shares", font=('Segoe UI', 8), bg="#3a3a3a", fg='#808080').pack()
        
        # Amount and Weight
        amount_container = tk.Frame(self, bg="#3a3a3a")
        amount_container.grid(row=0, column=3, sticky='ew', padx=10)
        
        tk.Label(amount_container, text=f"${amount:,.2f}", font=('Segoe UI', 12, 'bold'), bg="#3a3a3a", fg='#4caf50').pack(pady=(18, 2))
        tk.Label(amount_container, text=f"{weight:.1%} allocation", font=('Segoe UI', 8), bg="#3a3a3a", fg='#808080').pack()
        
        # Expected Return
        return_container = tk.Frame(self, bg="#3a3a3a")
        return_container.grid(row=0, column=4, sticky='ew', padx=10)
        
        return_color = '#4caf50' if return_pct > 0 else '#f44336'
        tk.Label(return_container, text=f"{return_pct:+.2%}", font=('Segoe UI', 12, 'bold'), bg="#3a3a3a", fg=return_color).pack(pady=(18, 2))
        tk.Label(return_container, text="expected", font=('Segoe UI', 8), bg="#3a3a3a", fg='#808080').pack()
        
        # Risk Badge
        risk_container = tk.Frame(self, bg="#3a3a3a")
        risk_container.grid(row=0, column=5, sticky='ew', padx=(10, 15))
        
        if risk < 0.20:
            risk_text, risk_color = "Low Risk", "#4caf50"
        elif risk < 0.30:
            risk_text, risk_color = "Medium", "#ff9800"
        else:
            risk_text, risk_color = "High Risk", "#f44336"
        
        risk_badge = tk.Frame(risk_container, bg=risk_color, height=32, width=95)
        risk_badge.place(relx=0.5, rely=0.5, anchor='center')
        risk_badge.pack_propagate(False)
        tk.Label(risk_badge, text=risk_text, font=('Segoe UI', 9, 'bold'), bg=risk_color, fg='#ffffff').place(relx=0.5, rely=0.5, anchor='center')


class StockPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Stock Market Advisor - Professional Edition")
        self.root.geometry("1500x900")
        self.root.configure(bg='#2b2b2b')
        
        self.engine = None
        self.loading = False
        
        self.colors = {
            'bg_dark': '#2b2b2b',
            'bg_medium': '#3a3a3a', 
            'bg_light': '#4a4a4a',
            'accent': '#0d7377',
            'accent_bright': '#14ffec',
            'success': '#4caf50',
            'warning': '#ff9800',
            'danger': '#f44336',
            'text_primary': '#ffffff',
            'text_secondary': '#b0b0b0',
        }
        
        self.create_widgets()
        self.load_engine()
    
    def create_widgets(self):
        # Top bar
        top_bar = tk.Frame(self.root, bg=self.colors['accent'], height=70)
        top_bar.pack(fill='x')
        top_bar.pack_propagate(False)
        
        title_container = tk.Frame(top_bar, bg=self.colors['accent'])
        title_container.place(relx=0.5, rely=0.5, anchor='center')
        
        tk.Label(title_container, text="📊", font=('Segoe UI', 32), bg=self.colors['accent'], fg='#ffffff').pack(side='left', padx=(0, 15))
        tk.Label(title_container, text="AI Stock Market Advisor", font=('Segoe UI', 26, 'bold'), bg=self.colors['accent'], fg='#ffffff').pack(side='left')
        tk.Label(title_container, text="Professional Edition", font=('Segoe UI', 11), bg=self.colors['accent'], fg=self.colors['accent_bright']).pack(side='left', padx=(15, 0))
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill='both', expand=True, padx=25, pady=25)
        main_container.grid_columnconfigure(0, weight=0, minsize=380)
        main_container.grid_columnconfigure(1, weight=1)
        main_container.grid_rowconfigure(0, weight=1)
        
        # Left panel
        left_panel = tk.Frame(main_container, bg=self.colors['bg_medium'])
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 20))
        
        self.create_clean_input_form(left_panel)
        
        # Right panel
        right_panel = tk.Frame(main_container, bg=self.colors['bg_dark'])
        right_panel.grid(row=0, column=1, sticky='nsew')
        
        # Portfolio header
        header = tk.Frame(right_panel, bg=self.colors['bg_medium'], height=60)
        header.pack(fill='x', pady=(0, 15))
        header.pack_propagate(False)
        tk.Label(header, text="📊  Investment Portfolio", font=('Segoe UI', 18, 'bold'), bg=self.colors['bg_medium'], fg=self.colors['text_primary']).pack(side='left', padx=25, pady=15)
        
        # Metrics
        self.metrics_container = tk.Frame(right_panel, bg=self.colors['bg_dark'])
        self.metrics_container.pack(fill='x', pady=(0, 15))
        self.create_metric_cards()
        
        # Stocks section with header
        stocks_section = tk.Frame(right_panel, bg=self.colors['bg_medium'])
        stocks_section.pack(fill='both', expand=True)
        
        stocks_header = tk.Frame(stocks_section, bg=self.colors['bg_medium'], height=50)
        stocks_header.pack(fill='x')
        stocks_header.pack_propagate(False)
        tk.Label(stocks_header, text="🎯  Top Stock Picks", font=('Segoe UI', 16, 'bold'), bg=self.colors['bg_medium'], fg=self.colors['text_primary']).pack(side='left', padx=25, pady=10)
        
        # Column headers
        headers_frame = tk.Frame(stocks_section, bg=self.colors['bg_light'], height=35)
        headers_frame.pack(fill='x')
        headers_frame.pack_propagate(False)
        
        headers_frame.grid_columnconfigure(0, weight=0, minsize=60)
        headers_frame.grid_columnconfigure(1, weight=0, minsize=280)
        headers_frame.grid_columnconfigure(2, weight=0, minsize=120)
        headers_frame.grid_columnconfigure(3, weight=0, minsize=140)
        headers_frame.grid_columnconfigure(4, weight=0, minsize=120)
        headers_frame.grid_columnconfigure(5, weight=0, minsize=100)
        
        tk.Label(headers_frame, text="Rank", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=0, column=0, sticky='w', padx=(25, 0), pady=8)
        tk.Label(headers_frame, text="Stock", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=0, column=1, sticky='w', padx=10, pady=8)
        tk.Label(headers_frame, text="Shares", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=0, column=2, sticky='w', padx=10, pady=8)
        tk.Label(headers_frame, text="Amount", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=0, column=3, sticky='w', padx=10, pady=8)
        tk.Label(headers_frame, text="Return", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=0, column=4, sticky='w', padx=10, pady=8)
        tk.Label(headers_frame, text="Risk", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_light'], fg=self.colors['text_secondary']).grid(row=0, column=5, sticky='w', padx=10, pady=8)
        
        # Scrollable stock list
        stocks_container = tk.Frame(stocks_section, bg=self.colors['bg_medium'])
        stocks_container.pack(fill='both', expand=True)
        
        canvas = tk.Canvas(stocks_container, bg=self.colors['bg_medium'], highlightthickness=0)
        scrollbar = tk.Scrollbar(stocks_container, orient='vertical', command=canvas.yview)
        
        self.stocks_frame = tk.Frame(canvas, bg=self.colors['bg_medium'])
        self.stocks_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        
        canvas_window = canvas.create_window((0, 0), window=self.stocks_frame, anchor='nw')
        
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind('<Configure>', on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.show_welcome()
        
        # Status bar
        status_bar = tk.Frame(self.root, bg=self.colors['bg_medium'], height=35)
        status_bar.pack(side='bottom', fill='x')
        status_bar.pack_propagate(False)
        self.status_label = tk.Label(status_bar, text="⚡ Ready to generate recommendations", font=('Segoe UI', 10), bg=self.colors['bg_medium'], fg=self.colors['text_secondary'], anchor='w')
        self.status_label.pack(side='left', padx=20, pady=8)
    
    def create_clean_input_form(self, parent):
        """Create clean, professional input form"""
        form_container = tk.Frame(parent, bg=self.colors['bg_medium'])
        form_container.pack(fill='both', expand=True, padx=25, pady=25)
        
        # Capital
        tk.Label(form_container, text="💰  Investment Amount", font=('Segoe UI', 12, 'bold'), bg=self.colors['bg_medium'], fg=self.colors['text_primary'], anchor='w').pack(anchor='w', pady=(0, 5))
        tk.Label(form_container, text="How much do you want to invest?", font=('Segoe UI', 9), bg=self.colors['bg_medium'], fg=self.colors['text_secondary'], anchor='w').pack(anchor='w', pady=(0, 8))
        
        capital_frame = tk.Frame(form_container, bg=self.colors['bg_light'], height=45)
        capital_frame.pack(fill='x', pady=(0, 25))
        capital_frame.pack_propagate(False)
        
        tk.Label(capital_frame, text=" $ ", font=('Segoe UI', 14, 'bold'), bg=self.colors['bg_light'], fg=self.colors['accent_bright']).pack(side='left', padx=(12, 5))
        self.capital_var = tk.StringVar(value="10000")
        tk.Entry(capital_frame, textvariable=self.capital_var, font=('Segoe UI', 14), bg=self.colors['bg_light'], fg=self.colors['text_primary'], insertbackground=self.colors['accent_bright'], relief='flat', bd=0).pack(side='left', fill='both', expand=True, padx=(0, 12))
        
        # Time Horizon
        tk.Label(form_container, text="⏱️  Time Horizon", font=('Segoe UI', 12, 'bold'), bg=self.colors['bg_medium'], fg=self.colors['text_primary'], anchor='w').pack(anchor='w', pady=(0, 5))
        tk.Label(form_container, text="How long will you hold?", font=('Segoe UI', 9), bg=self.colors['bg_medium'], fg=self.colors['text_secondary'], anchor='w').pack(anchor='w', pady=(0, 8))
        
        self.horizon_var = tk.StringVar(value="1w - One Week")
        ttk.Combobox(form_container, textvariable=self.horizon_var, values=['1d - One Day', '1w - One Week', '1m - One Month'], state='readonly', font=('Segoe UI', 11), height=15).pack(fill='x', pady=(0, 25))
        
        # Risk
        tk.Label(form_container, text="🎲  Risk Tolerance", font=('Segoe UI', 12, 'bold'), bg=self.colors['bg_medium'], fg=self.colors['text_primary'], anchor='w').pack(anchor='w', pady=(0, 5))
        tk.Label(form_container, text="How much risk can you handle?", font=('Segoe UI', 9), bg=self.colors['bg_medium'], fg=self.colors['text_secondary'], anchor='w').pack(anchor='w', pady=(0, 8))
        
        self.risk_var = tk.StringVar(value="medium - Balanced")
        ttk.Combobox(form_container, textvariable=self.risk_var, values=['low - Conservative', 'medium - Balanced', 'high - Aggressive'], state='readonly', font=('Segoe UI', 11), height=15).pack(fill='x', pady=(0, 25))
        
        # Goal
        tk.Label(form_container, text="🎯  Investment Goal", font=('Segoe UI', 12, 'bold'), bg=self.colors['bg_medium'], fg=self.colors['text_primary'], anchor='w').pack(anchor='w', pady=(0, 5))
        tk.Label(form_container, text="What are you optimizing for?", font=('Segoe UI', 9), bg=self.colors['bg_medium'], fg=self.colors['text_secondary'], anchor='w').pack(anchor='w', pady=(0, 8))
        
        self.goal_var = tk.StringVar(value="max_sharpe - Best Risk/Reward")
        ttk.Combobox(form_container, textvariable=self.goal_var, values=['max_return - Highest Returns', 'max_sharpe - Best Risk/Reward', 'prob_target - Target Return %'], state='readonly', font=('Segoe UI', 11), height=15).pack(fill='x', pady=(0, 25))
        
        # Portfolio Size
        tk.Label(form_container, text="📊  Portfolio Size", font=('Segoe UI', 12, 'bold'), bg=self.colors['bg_medium'], fg=self.colors['text_primary'], anchor='w').pack(anchor='w', pady=(0, 5))
        tk.Label(form_container, text="Number of stocks to hold", font=('Segoe UI', 9), bg=self.colors['bg_medium'], fg=self.colors['text_secondary'], anchor='w').pack(anchor='w', pady=(0, 8))
        
        self.positions_var = tk.IntVar(value=5)
        tk.Scale(form_container, from_=1, to=10, orient='horizontal', variable=self.positions_var, font=('Segoe UI', 10), bg=self.colors['bg_medium'], fg=self.colors['text_primary'], activebackground=self.colors['accent'], troughcolor=self.colors['bg_light'], highlightthickness=0, relief='flat', length=300).pack(fill='x', pady=(0, 25))
        
        # Fractional shares
        self.fractional_var = tk.BooleanVar(value=True)
        tk.Checkbutton(form_container, text="  Allow fractional shares (e.g., 3.5 shares)", variable=self.fractional_var, font=('Segoe UI', 10), bg=self.colors['bg_medium'], fg=self.colors['text_secondary'], selectcolor=self.colors['bg_light'], activebackground=self.colors['bg_medium'], relief='flat', bd=0).pack(anchor='w', pady=(0, 30))
        
        # Generate button
        self.generate_btn = tk.Button(form_container, text="🚀  Generate Portfolio", command=self.get_recommendations, font=('Segoe UI', 14, 'bold'), bg=self.colors['accent'], fg='#ffffff', activebackground=self.colors['accent_bright'], activeforeground='#000000', relief='flat', bd=0, cursor='hand2', height=2)
        self.generate_btn.pack(fill='x', pady=(10, 0))
    
    def create_metric_cards(self):
        for widget in self.metrics_container.winfo_children():
            widget.destroy()
        
        # Row 1
        row1 = tk.Frame(self.metrics_container, bg=self.colors['bg_dark'])
        row1.pack(fill='x', pady=(0, 15))
        
        self.prob_card = MetricCard(row1, "🎯", "Success Probability", "--")
        self.prob_card.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.return_card = MetricCard(row1, "💹", "Expected Return", "--", color=self.colors['success'])
        self.return_card.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.sharpe_card = MetricCard(row1, "⚖️", "Sharpe Ratio", "--", color=self.colors['accent_bright'])
        self.sharpe_card.pack(side='left', fill='both', expand=True)
        
        # Row 2
        row2 = tk.Frame(self.metrics_container, bg=self.colors['bg_dark'])
        row2.pack(fill='x')
        
        self.vol_card = MetricCard(row2, "📉", "Volatility", "--", color=self.colors['warning'])
        self.vol_card.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.var_card = MetricCard(row2, "⚠️", "Worst Case (5%)", "--", color=self.colors['danger'])
        self.var_card.pack(side='left', fill='both', expand=True)
    
    def show_welcome(self):
        for widget in self.stocks_frame.winfo_children():
            widget.destroy()
        
        welcome = tk.Frame(self.stocks_frame, bg=self.colors['bg_medium'])
        welcome.pack(fill='both', expand=True, pady=60)
        
        tk.Label(welcome, text="👋", font=('Segoe UI', 52), bg=self.colors['bg_medium'], fg=self.colors['accent_bright']).pack(pady=(0, 15))
        tk.Label(welcome, text="Welcome to AI Stock Advisor", font=('Segoe UI', 20, 'bold'), bg=self.colors['bg_medium'], fg=self.colors['text_primary']).pack(pady=(0, 10))
        tk.Label(welcome, text="Fill in your parameters and click 'Generate Portfolio'", font=('Segoe UI', 12), bg=self.colors['bg_medium'], fg=self.colors['text_secondary']).pack(pady=(0, 30))
        
        for feature in ["✓ Analyzes 196 stocks with machine learning", "✓ Predicts returns with 56% accuracy", "✓ Risk-adjusted recommendations", "✓ Daily market updates"]:
            tk.Label(welcome, text=feature, font=('Segoe UI', 11), bg=self.colors['bg_medium'], fg=self.colors['text_secondary']).pack(pady=3)
    
    def load_engine(self):
        def load():
            self.loading = True
            self.update_status("Loading AI models...")
            try:
                self.engine = EnhancedRecommendationEngine()
                self.update_status("✅ Ready to generate recommendations")
            except Exception as e:
                self.update_status(f"❌ Error: {str(e)}")
                messagebox.showerror("Error", f"Failed to load:\n{str(e)}")
            finally:
                self.loading = False
        
        threading.Thread(target=load, daemon=True).start()
    
    def get_recommendations(self):
        if self.loading:
            messagebox.showwarning("Please Wait", "AI models loading...")
            return
        
        if not self.engine:
            messagebox.showerror("Error", "AI models not loaded")
            return
        
        try:
            capital = float(self.capital_var.get())
            if capital <= 0:
                raise ValueError()
        except:
            messagebox.showerror("Invalid Input", "Enter valid amount")
            return
        
        horizon = self.horizon_var.get().split(' - ')[0]
        risk = self.risk_var.get().split(' - ')[0]
        goal = self.goal_var.get().split(' - ')[0]
        positions = self.positions_var.get()
        fractional = self.fractional_var.get()
        
        self.generate_btn.config(state='disabled', text="⏳  Analyzing...", bg=self.colors['bg_light'])
        self.update_status("🔍 Analyzing market data...")
        
        def process():
            try:
                result = self.engine.recommend(capital=capital, horizon=horizon, risk_level=risk, goal=goal, num_positions=positions, allow_fractional=fractional)
                self.root.after(0, lambda: self.display_results(result))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.generate_btn.config(state='normal', text="🚀  Generate Portfolio", bg=self.colors['accent']))
        
        threading.Thread(target=process, daemon=True).start()
    
    def display_results(self, result):
        if not result['success']:
            messagebox.showerror("No Results", result['message'])
            return
        
        portfolio = result['portfolio_metrics']
        recs = result['recommendations']
        inputs = result['inputs']
        
        # Update metrics
        prob = portfolio['probability_profit']
        prob_color = self.colors['success'] if prob >= 0.55 else self.colors['warning'] if prob >= 0.50 else self.colors['danger']
        
        self.prob_card.update_value(f"{prob:.1%}", prob_color)
        self.return_card.update_value(f"{portfolio['expected_return']:.2%}")
        self.sharpe_card.update_value(f"{portfolio['sharpe_estimate']:.2f}")
        self.vol_card.update_value(f"{portfolio['expected_volatility']:.1%}")
        self.var_card.update_value(f"{portfolio['var_5_pct']:.2%}")
        
        # Clear and show stocks
        for widget in self.stocks_frame.winfo_children():
            widget.destroy()
        
        # Add stock rows
        for i, rec in enumerate(recs, 1):
            StockRow(
                self.stocks_frame,
                rank=i,
                ticker=rec['ticker'],
                company=rec['company'],
                shares=rec['shares'],
                amount=rec['allocation'],
                weight=rec['weight'],
                return_pct=rec['predicted_return'],
                risk=rec['predicted_volatility']
            ).pack(fill='x', pady=2)
        
        # Summary
        total = sum(r['allocation'] for r in recs)
        cash = inputs['capital'] - total
        
        summary = tk.Frame(self.stocks_frame, bg=self.colors['bg_light'], height=55)
        summary.pack(fill='x', pady=(15, 0))
        summary.pack_propagate(False)
        tk.Label(summary, text=f"Total Invested: ${total:,.2f}  •  Remaining Cash: ${cash:,.2f}  •  {len(recs)} Stocks", font=('Segoe UI', 12, 'bold'), bg=self.colors['bg_light'], fg=self.colors['text_primary']).place(relx=0.5, rely=0.5, anchor='center')
        
        self.update_status(f"✅ Portfolio generated - {prob:.1%} success probability")
    
    def update_status(self, message):
        self.status_label.config(text=f"⚡ {message}")


if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictorGUI(root)
    root.mainloop()