"""
FinancialIQ Visualization Module
Handles visualization of financial data and trends
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np

class FinancialVisualizer:
    """Handles visualization of financial data"""
    
    def __init__(self):
        self.color_palette = {
            'revenue': '#2E86C1',
            'net_income': '#27AE60',
            'eps': '#E67E22',
            'assets': '#8E44AD',
            'liabilities': '#C0392B',
            'equity': '#F1C40F'
        }
    
    def create_financial_trends(self, data: pd.DataFrame, metrics: List[str], 
                              title: str = "Financial Trends") -> go.Figure:
        """Create a line chart showing financial trends over time"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        for metric in metrics:
            fig.add_trace(
                go.Scatter(
                    x=data['date'],
                    y=data[metric],
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=self.color_palette.get(metric, '#000000')),
                    mode='lines+markers'
                ),
                secondary_y=False
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Amount (USD)",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_financial_ratios(self, data: pd.DataFrame, ratios: List[str],
                              title: str = "Financial Ratios") -> go.Figure:
        """Create a bar chart showing financial ratios"""
        fig = go.Figure()
        
        for ratio in ratios:
            fig.add_trace(
                go.Bar(
                    x=data['date'],
                    y=data[ratio],
                    name=ratio.replace('_', ' ').title(),
                    marker_color=self.color_palette.get(ratio, '#000000')
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Ratio",
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_risk_factor_heatmap(self, data: pd.DataFrame,
                                 title: str = "Risk Factor Analysis") -> go.Figure:
        """Create a heatmap showing risk factor frequencies"""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='Reds',
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Risk Categories",
            yaxis_title="Companies",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_financial_metrics_comparison(self, data: pd.DataFrame,
                                          metrics: List[str],
                                          title: str = "Financial Metrics Comparison") -> go.Figure:
        """Create a radar chart comparing financial metrics across companies"""
        fig = go.Figure()
        
        for company in data['company'].unique():
            company_data = data[data['company'] == company]
            fig.add_trace(go.Scatterpolar(
                r=[company_data[metric].iloc[0] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=company
            ))
        
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, data[metrics].max().max()]
                )
            ),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_executive_compensation(self, data: pd.DataFrame,
                                    title: str = "Executive Compensation") -> go.Figure:
        """Create a stacked bar chart showing executive compensation breakdown"""
        fig = go.Figure()
        
        compensation_types = ['salary', 'bonus', 'stock_awards', 'option_awards', 'other']
        
        for comp_type in compensation_types:
            fig.add_trace(go.Bar(
                x=data['executive'],
                y=data[comp_type],
                name=comp_type.replace('_', ' ').title(),
                marker_color=self.color_palette.get(comp_type, '#000000')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Executive",
            yaxis_title="Compensation (USD)",
            barmode='stack',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def create_financial_statement_analysis(self, data: pd.DataFrame,
                                          title: str = "Financial Statement Analysis") -> go.Figure:
        """Create a waterfall chart showing changes in financial metrics"""
        fig = go.Figure(go.Waterfall(
            name="Financial Metrics",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Starting Balance", "Revenue", "Expenses", "Other Income", "Ending Balance"],
            y=[data['starting_balance'].iloc[0],
               data['revenue'].iloc[0],
               -data['expenses'].iloc[0],
               data['other_income'].iloc[0],
               data['ending_balance'].iloc[0]],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Category",
            yaxis_title="Amount (USD)",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig

    def extract_revenue_data(self, filings: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract revenue data from filings"""
        try:
            # Extract relevant columns
            revenue_data = filings[['filedAt', 'formType', 'companyName']].copy()
            
            # Convert filedAt to datetime
            revenue_data['filedAt'] = pd.to_datetime(revenue_data['filedAt'])
            
            # Add placeholder columns for financial data
            revenue_data['revenue'] = np.nan
            revenue_data['net_income'] = np.nan
            revenue_data['eps'] = np.nan
            
            # Sort by date
            revenue_data = revenue_data.sort_values('filedAt')
            
            return revenue_data
        except Exception as e:
            print(f"Error extracting revenue data: {str(e)}")
            return None

    def extract_risk_factors(self, filings: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract risk factors from filings"""
        try:
            # Extract relevant columns
            risk_data = filings[['filedAt', 'formType', 'companyName']].copy()
            
            # Convert filedAt to datetime
            risk_data['filedAt'] = pd.to_datetime(risk_data['filedAt'])
            
            # Add placeholder columns for risk factors
            risk_data['market_risk'] = np.nan
            risk_data['operational_risk'] = np.nan
            risk_data['regulatory_risk'] = np.nan
            risk_data['financial_risk'] = np.nan
            
            # Sort by date
            risk_data = risk_data.sort_values('filedAt')
            
            return risk_data
        except Exception as e:
            print(f"Error extracting risk factors: {str(e)}")
            return None

    def extract_executive_compensation(self, filings: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract executive compensation data from filings"""
        try:
            # Extract relevant columns
            comp_data = filings[['filedAt', 'formType', 'companyName']].copy()
            
            # Convert filedAt to datetime
            comp_data['filedAt'] = pd.to_datetime(comp_data['filedAt'])
            
            # Add placeholder columns for executive compensation
            comp_data['ceo_compensation'] = np.nan
            comp_data['cfo_compensation'] = np.nan
            comp_data['other_executives'] = np.nan
            
            # Sort by date
            comp_data = comp_data.sort_values('filedAt')
            
            return comp_data
        except Exception as e:
            print(f"Error extracting executive compensation: {str(e)}")
            return None

    def create_revenue_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create a revenue trends chart"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add form type indicators
        for form_type in data['formType'].unique():
            form_data = data[data['formType'] == form_type]
            fig.add_trace(
                go.Scatter(
                    x=form_data['filedAt'],
                    y=[0] * len(form_data),
                    name=f'{form_type} Filing',
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color=self.color_palette.get(form_type, '#000000')
                    ),
                    text=form_data['companyName'],
                    hoverinfo='text+x'
                ),
                secondary_y=False
            )
        
        fig.update_layout(
            title="Filing Timeline",
            xaxis_title="Date",
            yaxis_title="",
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        
        return fig

    def create_risk_factors_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create a risk factors timeline chart"""
        fig = go.Figure()
        
        # Add form type indicators
        for form_type in data['formType'].unique():
            form_data = data[data['formType'] == form_type]
            fig.add_trace(
                go.Scatter(
                    x=form_data['filedAt'],
                    y=[0] * len(form_data),
                    name=f'{form_type} Filing',
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color=self.color_palette.get(form_type, '#000000')
                    ),
                    text=form_data['companyName'],
                    hoverinfo='text+x'
                )
            )
        
        fig.update_layout(
            title="Filing Timeline",
            xaxis_title="Date",
            yaxis_title="",
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        
        return fig

    def create_compensation_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create a filing timeline chart"""
        fig = go.Figure()
        
        # Add form type indicators
        for form_type in data['formType'].unique():
            form_data = data[data['formType'] == form_type]
            fig.add_trace(
                go.Scatter(
                    x=form_data['filedAt'],
                    y=[0] * len(form_data),
                    name=f'{form_type} Filing',
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=10,
                        color=self.color_palette.get(form_type, '#000000')
                    ),
                    text=form_data['companyName'],
                    hoverinfo='text+x'
                )
            )
        
        fig.update_layout(
            title="Filing Timeline",
            xaxis_title="Date",
            yaxis_title="",
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        
        return fig 