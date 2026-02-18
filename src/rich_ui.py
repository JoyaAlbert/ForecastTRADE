# -*- coding: utf-8 -*-
"""
ForecastTRADE Rich UI Module
=============================
Provides a beautiful, minimal, and aesthetic terminal UI using Rich.
Handles:
- Asset selection with interactive menu
- Progress tracking for calculations
- Results display with formatted tables and metrics
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn
from rich.layout import Layout
from rich.columns import Columns
from rich import box
from datetime import datetime
import json
import sys

# Initialize Rich console
console = Console()


class TradingRecommendation:
    """Encapsulates volatility-adjusted recommendation levels and config payload."""

    def __init__(self, probability, entry_price, profit_target, stop_loss, dynamic_risk_config=None):
        self.probability = probability
        self.entry_price = entry_price
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.dynamic_risk_config = dynamic_risk_config or {
            "dynamic_risk": {
                "type": "Volatility_Adjusted",
                "params": {
                    "k_tp": 2.5,
                    "k_sl": 1.25,
                    "vol_metric": "rolling_std_20d"
                }
            }
        }

    def target_pct(self):
        return ((self.profit_target - self.entry_price) / self.entry_price) * 100 if self.entry_price else 0.0

    def stop_pct(self):
        return ((self.stop_loss - self.entry_price) / self.entry_price) * 100 if self.entry_price else 0.0

    def to_dict(self):
        payload = {
            "entry": self.entry_price,
            "profit_target": self.profit_target,
            "stop_loss": self.stop_loss,
            "probability": self.probability,
            **self.dynamic_risk_config
        }
        return self._to_json_safe(payload)

    @staticmethod
    def _to_json_safe(value):
        if isinstance(value, dict):
            return {k: TradingRecommendation._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [TradingRecommendation._to_json_safe(v) for v in value]
        if hasattr(value, 'item'):
            try:
                return value.item()
            except Exception:
                pass
        return value

class RichUI:
    """Main class for Rich UI interactions"""
    
    def __init__(self):
        """Initialize the Rich UI"""
        self.console = console
        self.selected_ticker = None
        self.selected_name = None
    
    def display_header(self):
        """Display the main header"""
        header = Text()
        header.append("╔═══════════════════════════════════════════════════════════════════╗\n", style="bright_cyan")
        header.append("║                    ✨  ForecastTRADE v2.0  ✨                     ║\n", style="bright_cyan")
        header.append("║           Hybrid LSTM-XGBoost Stock Return Predictor              ║\n", style="cyan")
        header.append("║                        Powered by Rich UI                         ║\n", style="cyan")
        header.append("╚═══════════════════════════════════════════════════════════════════╝", style="bright_cyan")
        
        self.console.print(header)
        self.console.print()  # Blank line
    
    def select_asset(self):
        """Display asset selection menu with rich formatting"""
        
        # Clear screen (optional)
        # console.clear()
        
        # Create title
        title = Panel(
            Text("📊 SELECT STOCK TO FORECAST", style="bold bright_white", justify="center"),
            border_style="bright_cyan",
            padding=(1, 2),
            title="[bold]Asset Selection[/bold]"
        )
        self.console.print(title)
        
        # Create assets table
        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold bright_cyan",
            padding=(0, 1),
            show_edge=True
        )
        
        table.add_column("Opt", style="bold yellow", width=5, justify="center")
        table.add_column("Ticker", style="bold white", width=10)
        table.add_column("Company Name", style="bright_white", width=35)
        table.add_column("Status", width=15, justify="center")
        
        assets = {
            '1': {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
            '2': {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
            '3': {'symbol': 'AAPL', 'name': 'Apple Inc.'},
            '4': {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
            '5': {'symbol': 'IBE.MC', 'name': 'Iberdrola SA'},
            '6': {'symbol': 'CUSTOM', 'name': 'Enter custom ticker'}
        }
        
        for key, value in assets.items():
            if key == '3':  # Default option
                symbol = f"[bold bright_cyan]{value['symbol']}[/bold bright_cyan]"
                status = "[bold bright_green]⭐ DEFAULT[/bold bright_green]"
            elif value['symbol'] == 'CUSTOM':
                symbol = f"[yellow]{value['symbol']}[/yellow]"
                status = "[dim]Custom Input[/dim]"
            else:
                symbol = f"[white]{value['symbol']}[/white]"
                status = "[dim]Available[/dim]"
            
            table.add_row(f"[bold yellow][{key}][/bold yellow]", symbol, value['name'], status)
        
        self.console.print(table)
        self.console.print()
        
        # Get user input
        prompt = Text()
        prompt.append("Enter your choice ", style="white")
        prompt.append("(1-6) ", style="bright_cyan")
        prompt.append("[", style="dim")
        prompt.append("default: 3", style="bright_green")
        prompt.append("]: ", style="dim")
        
        try:
            choice = self.console.input(prompt).strip()
            
            if not choice:
                choice = '3'
            
            assets_dict = {
                '1': ('NVDA', 'NVIDIA Corporation'),
                '2': ('TSLA', 'Tesla Inc.'),
                '3': ('AAPL', 'Apple Inc.'),
                '4': ('MSFT', 'Microsoft Corporation'),
                '5': ('IBE.MC', 'Iberdrola SA'),
                '6': (None, None)  # Custom
            }
            
            if choice in assets_dict:
                if choice == '6':
                    custom_ticker = self.console.input("[bold cyan]➤ Enter custom ticker symbol[/bold cyan] (e.g., GOOGL, AMZN): ").strip().upper()
                    if custom_ticker:
                        self.selected_ticker = custom_ticker
                        self.selected_name = custom_ticker
                    else:
                        self._use_default()
                else:
                    self.selected_ticker, self.selected_name = assets_dict[choice]
            else:
                self._use_default()
        
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]⚠️  Selection cancelled. Using default: AAPL[/yellow]")
            self._use_default()
        
        # Display confirmation
        self._display_selection_confirmation()
    
    def _use_default(self):
        """Use default ticker (AAPL)"""
        self.selected_ticker = 'AAPL'
        self.selected_name = 'Apple Inc.'
    
    def _display_selection_confirmation(self):
        """Display confirmation panel for selected asset"""
        confirmation = Panel(
            Text(
                f"✅ Selected: {self.selected_name} ({self.selected_ticker})",
                style="bold bright_green",
                justify="center"
            ),
            border_style="bright_green",
            padding=(1, 3),
            width=60
        )
        self.console.print(confirmation)
        self.console.print()
    
    def show_progress_fetching(self, ticker, num_records):
        """Display progress for data fetching"""
        msg = Text()
        msg.append("✅ Datos obtenidos: ", style="bright_green")
        msg.append(f"{num_records}", style="bold bright_cyan")
        msg.append(" registros", style="white")
        self.console.print(msg)
    
    def show_progress_engineering(self, feature_count):
        """Display progress for feature engineering"""
        title = Panel(
            Text("🔧 FEATURE ENGINEERING IN PROGRESS", style="bold bright_yellow", justify="center"),
            border_style="bright_yellow",
            padding=(0, 2),
        )
        self.console.print(title)
        
        # Create a simple progress visualization
        # stages = [
        #     ("Momentum Indicators", 0.15),
        #     ("Volatility Indicators", 0.30),
        #     ("Trend Indicators", 0.45),
        #     ("Volume Indicators", 0.60),
        #     ("Price Patterns", 0.75),
        #     ("Macro Indicators", 0.85),
        #     ("Regime Indicators", 1.0),
        # ]
        
        # for stage, progress in stages:
        #     bar = "█" * int(progress * 30) + "░" * int((1 - progress) * 30)
        #     pct = int(progress * 100)
        #     msg = f"[bright_cyan]{stage:<25}[/bright_cyan] [bright_white]|{bar}|[/bright_white] [bright_green]{pct:3d}%[/bright_green]"
        #     self.console.print(msg)
        
        self.console.print("[dim]Processing technical indicators...[/dim]")
        self.console.print()
        summary = Text()
        summary.append(f"✅ Total features created: ", style="bright_green")
        summary.append(f"{feature_count}", style="bold bright_cyan")
        self.console.print(summary)
        self.console.print()
    
    def show_triple_barrier_info(self, profit_pct, stop_loss_pct, max_days, labels_dist):
        """Display Triple Barrier Method information (DEPRECATED - use percentile labeling)"""
        panel_content = Text()
        panel_content.append("Triple Barrier Labeling Configuration\n", style="bold bright_cyan")
        panel_content.append("\n📊 Adaptive Parameters:\n", style="bright_cyan")
        panel_content.append(f"  • Profit Target:    +{profit_pct*100:.2f}%\n", style="white")
        panel_content.append(f"  • Stop Loss:        -{stop_loss_pct*100:.2f}%\n", style="white")
        panel_content.append(f"  • Max Holding Days: {max_days} days\n\n", style="white")
        panel_content.append("Label Distribution:\n", style="bright_cyan")
        
        for label, count, pct in labels_dist:
            label_emoji = "🎯" if label == "Profit Target" else ("⛔" if label == "Stop Loss" else "⏱️")
            bar = "█" * int(pct / 2) + "░" * int((100 - pct) / 2)
            panel_content.append(f"  {label_emoji} {label:<20} |{bar}| {pct:5.1f}% ({count:6d})\n", style="white")
        
        panel = Panel(
            panel_content,
            border_style="bright_cyan",
            padding=(1, 2),
            title="[bold]Target Definition[/bold]"
        )
        self.console.print(panel)
        self.console.print()
    
    def show_percentile_labeling_info(self, horizon, p_bottom, p_top, threshold_sell, threshold_buy, labels_dist):
        """Display Percentile-Based Labeling information"""
        panel_content = Text()
        panel_content.append("Percentile-Based Labeling Configuration\n", style="bold bright_cyan")
        panel_content.append("\n📊 Strategy: Quantile Classification\n", style="bright_cyan")
        panel_content.append(f"  • Forward Horizon:  {horizon} days\n", style="white")
        panel_content.append(f"  • SELL Percentile:  Bottom {p_bottom}% (≤ {threshold_sell*100:+.2f}%)\n", style="bright_red")
        panel_content.append(f"  • BUY Percentile:   Top {100-p_top}% (≥ {threshold_buy*100:+.2f}%)\n", style="bright_green")
        panel_content.append(f"  • HOLD (Discarded): Middle {p_top-p_bottom}%\n\n", style="dim")
        panel_content.append("Label Distribution:\n", style="bright_cyan")
        
        for label, count, pct in labels_dist:
            if "SELL" in label:
                label_emoji = "📉"
                style_color = "bright_red"
            elif "BUY" in label:
                label_emoji = "📈"
                style_color = "bright_green"
            else:
                label_emoji = "⏸️"
                style_color = "dim"
            
            bar = "█" * int(pct / 2) + "░" * int((100 - pct) / 2)
            panel_content.append(f"  {label_emoji} {label:<20} |{bar}| {pct:5.1f}% ({count:6d})\n", style=style_color)
        
        panel_content.append("\n✅ Guaranteed Balance: 35% SELL / 35% BUY / 30% HOLD\n", style="bold bright_yellow")
        
        panel = Panel(
            panel_content,
            border_style="bright_magenta",
            padding=(1, 2),
            title="[bold]Target Definition (Percentile Method)[/bold]"
        )
        self.console.print(panel)
        self.console.print()
    
    def show_training_progress(self, fold_number, total_folds, train_dates, val_dates, class_dist):
        """Display training progress with formatted information"""
        
        header = Text(f"Fold {fold_number}/{total_folds}", style="bold bright_white")
        train_text = Text()
        train_text.append(f"  Train: ", style="dim")
        train_text.append(train_dates, style="bright_cyan")
        
        val_text = Text()
        val_text.append(f"  Valid: ", style="dim")
        val_text.append(val_dates, style="bright_green")
        
        class_text = Text()
        class_text.append(f"  Classes: ", style="dim")
        for cls, count in class_dist.items():
            class_text.append(f"{cls}→{count} ", style="bright_yellow")
        
        output = f"{header}\n{train_text}\n{val_text}\n{class_text}"
        
        panel = Panel(
            output,
            border_style="bright_cyan" if fold_number % 2 == 0 else "blue",
            padding=(0, 2),
            width=70
        )
        self.console.print(panel)
        self.console.print()
    
    def show_fold_results(self, fold_num, metrics):
        """Display results for a single fold with styled metrics"""
        
        # Create title
        title = Text(f"Fold {fold_num} Results", style="bold bright_white")
        
        # Create metrics table
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold bright_cyan",
            padding=(0, 1),
        )
        
        table.add_column("Metric", style="white", width=20)
        table.add_column("Value", style="bright_cyan", width=12)
        
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                table.add_row(metric_name, f"{value:.4f}")
            else:
                table.add_row(metric_name, str(value))
        
        panel = Panel(
            table,
            border_style="bright_cyan",
            padding=(0, 2),
            title=f"[bold]{title}[/bold]"
        )
        self.console.print(panel)
    
    def show_cv_summary(self, summary_stats):
        """Display cross-validation summary with statistics"""
        
        title = Panel(
            Text("📊 CROSS-VALIDATION SUMMARY", style="bold bright_white", justify="center"),
            border_style="bright_cyan",
            padding=(0, 2),
        )
        self.console.print(title)
        self.console.print()
        
        # Create summary table
        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold bright_cyan",
            padding=(0, 2),
        )
        
        table.add_column("Metric", style="bold white", width=25)
        table.add_column("Mean", style="bright_cyan", width=15)
        table.add_column("Std Dev", style="bright_yellow", width=15)
        
        for metric, (mean, std) in summary_stats.items():
            # Add color based on metric value
            if "Sharpe" in metric:
                mean_style = "bright_green" if mean > 0 else "bright_red"
            else:
                mean_style = "bright_cyan"
            
            table.add_row(
                f"[white]{metric}[/white]",
                f"[{mean_style}]{mean:.4f}[/{mean_style}]",
                f"[bright_yellow]{std:.4f}[/bright_yellow]"
            )
        
        self.console.print(table)
        self.console.print()
    
    def show_recommendation(self, probability, last_price, profit_target, stop_loss, date_str, dynamic_risk_config=None):
        """Display final trading recommendation with emoji and styling"""
        rec = TradingRecommendation(
            probability=probability,
            entry_price=last_price,
            profit_target=profit_target,
            stop_loss=stop_loss,
            dynamic_risk_config=dynamic_risk_config
        )
        
        # Determine recommendation
        if probability > 0.65:
            recommendation = "🟢 BUY (High confidence)"
            color = "bright_green"
            reason = "Strong win probability"
            action_color = "green"
            action_text = "BUY"
        elif probability < 0.35:
            recommendation = "🔴 AVOID/SELL (High risk)"
            color = "bright_red"
            reason = "High loss probability"
            action_color = "red"
            action_text = "AVOID / SELL"
        else:
            recommendation = "🟡 HOLD/WAIT (Uncertain)"
            color = "bright_yellow"
            reason = "Signal not clear enough"
            action_color = "yellow"
            action_text = "HOLD / WAIT"
        
        # Build recommendation panel using Text objects
        panel_content = Text()
        panel_content.append(f"Valid from: ", style="dim")
        panel_content.append(f"{date_str}\n\n", style="bright_cyan")
        
        panel_content.append("Entry Point:\n", style="bright_cyan")
        panel_content.append(f"  💰 Price: ${last_price:.2f}\n\n", style="bold bright_white")
        
        panel_content.append("Target Levels:\n", style="bright_cyan")
        panel_content.append(f"  🎯 Profit Target: ${profit_target:.2f} ", style="white")
        panel_content.append(f"({rec.target_pct():+.2f}%)\n", style="bright_green")
        panel_content.append(f"  🛑 Stop Loss:     ${stop_loss:.2f} ", style="white")
        panel_content.append(f"({rec.stop_pct():+.2f}%)\n\n", style="bright_red")

        panel_content.append("Dynamic Risk Config:\n", style="bright_cyan")
        panel_content.append(json.dumps(rec.dynamic_risk_config, indent=2), style="white")
        panel_content.append("\n\n", style="white")
        
        panel_content.append("Forecast:\n", style="bright_cyan")
        panel_content.append(f"  📊 Win Probability: ", style="white")
        panel_content.append(f"{probability*100:.1f}%\n\n", style=f"bold {action_color}")
        
        # Add recommendation
        panel_content.append("─" * 50 + "\n", style="dim")
        panel_content.append("  RECOMMENDATION: ", style="dim")
        panel_content.append(f"{action_text}", style=f"bold {color}")
        panel_content.append("\n" + "─" * 50, style="dim")
        
        # Create subtitle as Text object
        subtitle_text = Text(reason, style=color)
        
        panel = Panel(
            panel_content,
            border_style=color,
            padding=(1, 3),
            title=Text("🎯 TRADING RECOMMENDATION", style=f"bold {color}"),
            subtitle=subtitle_text
        )
        
        self.console.print(panel)
        self.console.print()

        self.console.print("[dim]Recommendation payload:[/dim]")
        self.console.print(json.dumps(rec.to_dict(), indent=2), style="dim")
        self.console.print()
        
        # Return recommendation text
        return recommendation
    
    def show_final_summary(self, ticker, recommendations):
        """Display final ensemble recommendation"""
        
        if not recommendations:
            return
        
        title = Panel(
            Text(f"🔮 FINAL ENSEMBLE RECOMMENDATION - {ticker}", 
                 style="bold bright_white", justify="center"),
            border_style="bright_cyan",
            padding=(0, 2),
        )
        self.console.print(title)
        self.console.print()
        
        # Calculate statistics
        probs = [r['probability'] for r in recommendations]
        avg_prob = sum(probs) / len(probs)
        max_prob = max(probs)
        min_prob = min(probs)
        confidence = max_prob - min_prob
        
        # Display ensemble stats
        stats_content = Text()
        stats_content.append("Ensemble Statistics:\n", style="bright_cyan")
        stats_content.append(f"  • Number of Models: {len(recommendations)}\n", style="white")
        stats_content.append(f"  • Average Probability: ", style="white")
        stats_content.append(f"{avg_prob*100:.1f}%\n", style="bold bright_green" if avg_prob > 0.5 else "bold")
        stats_content.append(f"  • Max Probability: ", style="white")
        stats_content.append(f"{max_prob*100:.1f}%\n", style="bright_green")
        stats_content.append(f"  • Min Probability: ", style="white")
        stats_content.append(f"{min_prob*100:.1f}%\n", style="bright_red")
        stats_content.append(f"  • Confidence (Spread): ", style="white")
        
        if confidence < 0.1:
            stats_content.append(f"{confidence*100:.1f}%", style="bright_green")
            stats_content.append(" (High consensus)\n", style="bright_green")
        elif confidence < 0.2:
            stats_content.append(f"{confidence*100:.1f}%", style="bright_yellow")
            stats_content.append(" (Moderate consensus)\n", style="bright_yellow")
        else:
            stats_content.append(f"{confidence*100:.1f}%", style="bright_red")
            stats_content.append(" (Low consensus)\n", style="bright_red")
        
        panel = Panel(
            stats_content,
            border_style="bright_cyan",
            padding=(1, 2),
        )
        self.console.print(panel)
        self.console.print()
    
    def show_error(self, error_message):
        """Display error message in red panel"""
        error_panel = Panel(
            Text(f"❌ Error: {error_message}", style="bright_red"),
            border_style="bright_red",
            padding=(0, 2),
        )
        self.console.print(error_panel)
    
    def show_warning(self, warning_message):
        """Display warning message in yellow panel"""
        warning_panel = Panel(
            Text(f"⚠️  {warning_message}", style="bright_yellow"),
            border_style="bright_yellow",
            padding=(0, 2),
        )
        self.console.print(warning_panel)
    
    def show_success(self, success_message):
        """Display success message in green panel"""
        success_panel = Panel(
            Text(f"✅ {success_message}", style="bright_green"),
            border_style="bright_green",
            padding=(0, 2),
        )
        self.console.print(success_panel)
    
    def show_feature_importance(self, top_features):
        """Display top feature importances in a nice table"""
        
        title = Panel(
            Text("🎯 TOP FEATURE IMPORTANCES", style="bold bright_white", justify="center"),
            border_style="bright_cyan",
            padding=(0, 2),
        )
        self.console.print(title)
        self.console.print()
        
        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold bright_cyan",
            padding=(0, 2),
        )
        
        table.add_column("Rank", style="bold yellow", width=6, justify="center")
        table.add_column("Feature", style="bright_white", width=25)
        table.add_column("Importance", style="bright_cyan", width=15)
        table.add_column("Contribution", width=20)
        
        for rank, (feature, importance) in enumerate(top_features, 1):
            bar = "█" * int(importance * 100) + "░" * int((1 - importance) * 100)
            table.add_row(
                str(rank),
                f"[bright_white]{feature}[/bright_white]",
                f"[bright_cyan]{importance:.4f}[/bright_cyan]",
                f"[bright_yellow]{bar}[/bright_yellow]"
            )
        
        self.console.print(table)
        self.console.print()


# Convenience functions for easy access
ui = RichUI()

def select_asset():
    """Get asset selection from user"""
    ui.display_header()
    ui.select_asset()
    return ui.selected_ticker, ui.selected_name


def show_progress_bar(description, total):
    """Show a progress bar for long-running operations"""
    with Progress(
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(f"[bold cyan]{description}[/bold cyan]", total=total)
        return task, progress
