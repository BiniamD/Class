import React from 'react';

const Presentation = () => {
  const slides = [
    {
      title: "Deep Learning for Enhanced Trading Signal Generation",
      subtitle: "A Hybrid CNN-BiLSTM Model with Attention Mechanism for Stock Market Prediction and Trading Signal Generation",
      author: "Biniam Abebe",
      institution: "University of North Texas",
      department: "Advanced Data Analytics - ADTA 5900 Capstone",
      date: "April 26, 2025",
      logo: "UNT Logo"
    },
    {
      title: "Agenda",
      content: [
        "Introduction and Problem Statement",
        "Research Objectives and Questions",
        "Literature Review Highlights",
        "Methodology and Data Description",
        "Model Architecture and Implementation",
        "Trading Strategy Development",
        "Experimental Results and Performance Analysis",
        "Key Findings and Implementation Insights",
        "Limitations and Future Research",
        "Conclusion and Recommendations"
      ]
    },
    {
      title: "Introduction & Problem Statement",
      content: [
        "Traditional technical analysis is subjective and prone to psychological biases",
        "Deep learning offers objective pattern recognition capabilities",
        "U.S. equity market: $7 trillion market cap with 60-70% algorithmic trading",
        "Proposed solution: Hybrid CNN-BiLSTM model for enhanced trading signal generation",
        "Focus on S&P 500 stocks with comprehensive technical and fundamental indicators"
      ],
      image: "Traditional Analysis vs AI-Driven Analysis",
      notes: "Emphasize the limitations of traditional analysis methods and the need for more objective, data-driven approaches in today's fast-paced markets."
    },
    {
      title: "Research Objectives & Questions",
      content: [
        "Primary objective: Develop and validate a hybrid deep learning model for stock market prediction and trading signal generation",
        "Research Question 1: How effective is a hybrid CNN-BiLSTM model compared to traditional technical analysis?",
        "Research Question 2: Can the hybrid model provide better trading signal reliability and profitability?",
        "Research Question 3: How do different market regimes affect model performance?",
        "Research Question 4: Can risk management be improved through deep learning predictions?"
      ],
      notes: "These research questions directly address the gap between traditional technical analysis and modern deep learning approaches, focusing on both prediction accuracy and practical trading implementation."
    },
    {
      title: "Literature Review Highlights",
      content: [
        "Deep learning excels in handling complex financial data for forecasting (Huang et al., 2020)",
        "Hybrid CNN-LSTM models consistently outperform standalone architectures in stock prediction (Shah et al., 2022)",
        "CNN-BiLSTM-AM achieved lowest error rates (MAE: 21.952, RMSE: 31.694) compared to other models",
        "Technical indicators significantly enhance prediction performance when combined with deep learning (Sezer et al., 2017)",
        "Graph-based CNN-LSTM with leading indicators improves prediction precision (Wu et al., 2023)",
        "Performance metrics standardization: Sharpe ratio, win rate, maximum drawdown (Saud & Shakya, 2024)"
      ],
      chart: "Comparative Model Performance from Literature",
      notes: "The literature consistently shows that hybrid architectures outperform single models, and that combining technical indicators with deep learning yields superior results."
    },
    {
      title: "Methodology & Data Description",
      content: [
        "Rich dataset: 501 S&P 500 companies over 5 years (2019-2024)",
        "Comprehensive feature set: 76 technical and fundamental indicators",
        "High data quality: Only 1.9% missing values",
        "Data sources: Daily price data (Yahoo Finance API), Market metrics (Alpha Vantage API)",
        "Data processing: Imputation, outlier detection, time series alignment",
        "Time series analysis: Volatility clustering, market regimes, seasonal patterns"
      ],
      chart: "Stock Price Return Distribution",
      notes: "This slide highlights the comprehensive nature of our dataset and the rigorous methodology employed for data collection and preprocessing."
    },
    {
      title: "Data Processing & Feature Engineering",
      content: [
        "Price-based features: OHLC, returns, log returns, price ranges",
        "Technical indicators across multiple timeframes: Moving averages (5, 10, 20, 50, 200 days), RSI (9, 14, 25 periods), MACD, Bollinger Bands",
        "Market features: Market returns, volatility measures, VIX data, rolling beta calculations",
        "Fundamental features: PE ratio, PB ratio, dividend yield, profit margin, enterprise value",
        "Class imbalance handling with SMOTE: Balanced trading signals (50-50 distribution)",
        "Mathematical formulations: Moving Averages, RSI, MACD, Bollinger Bands"
      ],
      chart: "Feature Correlation Heatmap",
      notes: "The feature engineering process was critical to model performance, transforming raw market data into meaningful signals that the deep learning model could use effectively."
    },
    {
      title: "Deep Learning Architecture",
      content: [
        "CNN component: Processes local patterns through 64 filters with kernel size 3, followed by max pooling and dropout (0.2)",
        "BiLSTM structure: Three stacked layers (128, 32, 32 units) with bidirectional processing for enhanced temporal feature capture",
        "Attention mechanism: SoftMax-activated scoring system to focus on relevant temporal patterns",
        "Attention formula: Attention Score = softmax(W · ht + b), where ht is the hidden state",
        "Training split: 70% training, 15% validation, 15% testing",
        "Optimization: Adam optimizer (learning rate 0.001), batch size 32, 50 epochs"
      ],
      image: "CNN-BiLSTM with Attention Architecture Diagram",
      notes: "The hybrid architecture combines the strengths of CNNs for local pattern recognition with BiLSTMs for temporal sequence learning, while the attention mechanism helps identify the most relevant features at each time step."
    },
    {
      title: "Trading Strategy Development",
      content: [
        "Signal generation: Probability threshold system (Signal = 1 if probability > 0.60, 0 otherwise)",
        "Position sizing: Dynamic allocation based on model confidence (linear scaling)",
        "Technical confirmation framework: Moving Averages (50, 200-day), RSI, MACD, Bollinger Bands",
        "Risk management protocols: Stop-loss (2% below entry), Take-profit (5% above entry), Maximum holding period (30 days)",
        "Market regime adaptation: Strategy parameters adjusted based on detected market conditions",
        "Performance evaluation: Standard financial metrics (Sharpe Ratio, Maximum Drawdown, Win Rate, Profit Factor)"
      ],
      chart: "Strategy Performance by Market Regime",
      notes: "Our trading strategy integrates the deep learning model predictions with established technical analysis principles and systematic risk management protocols, creating a robust framework for market participation."
    },
    {
      title: "Experimental Results",
      content: [
        "Top performer: Walmart (WMT) - 48.18% return, 72.73% win rate, 3.38% maximum drawdown",
        "Strong performers: Mastercard (MA) - 19.45% risk-adjusted return, 50% win rate",
        "Portfolio average across all stocks: 15.4% return, 1.85 Sharpe ratio, 58.6% win rate",
        "Risk management effectiveness: Maximum drawdowns < 5% for top performers",
        "Market condition sensitivity: Better performance in stable market environments",
        "Trading frequency finding: Selective trading (10-15 trades) outperformed high-frequency trading (40+ trades)"
      ],
      chart: "Performance Metrics Heatmap",
      notes: "These results demonstrate that our hybrid model generates reliable trading signals that translate into significant risk-adjusted returns when implemented with proper risk management."
    },
    {
      title: "Performance Analysis",
      content: [
        "Comparative analysis: Strategy outperformed baseline by 12.3% while maintaining lower volatility",
        "Stock category analysis: Large-cap retail (WMT) and financial (MA, JPM) sectors showed strongest performance",
        "Technology sector challenges: High-volatility stocks like NVIDIA showed mixed results (-21.65% return, 31.82% win rate)",
        "Trading frequency impact: Negative correlation between trade frequency and performance (NVDA: 44 trades, -21.65% return vs. WMT: 11 trades, 48.18% return)",
        "Win rate correlation: Strong positive relationship between win rate and total return (R² = 0.78)",
        "Risk-return profile: Top performers maintained exceptional risk-adjusted returns (Sharpe ratios > 2.0)"
      ],
      chart: "Scatter Plot: Win Rate vs. Total Return",
      notes: "The relationship between win rate and total return was one of our most significant findings, highlighting the importance of trade quality over quantity."
    },
    {
      title: "Key Findings & Implementation Insights",
      content: [
        "Trading Frequency: Quality over quantity - selective trading (10-15 trades) significantly outperforms high-frequency trading",
        "Stock Selection: Large-cap stability (WMT, MA, JPM) delivers consistent performance compared to volatile tech stocks",
        "Attention Mechanism: Successfully identifies the most important temporal patterns and market regime shifts",
        "SMOTE Balancing: Dramatically improves model's ability to identify profitable trading opportunities",
        "Risk Management: Confidence-based position sizing and adaptive stop-losses maintain low drawdowns",
        "Technical Confirmation: Integration of model predictions with technical indicators produces robust trading signals"
      ],
      image: "Cumulative Return Comparison for WMT",
      notes: "These insights provide practical guidance for implementing the model in real-world trading scenarios, emphasizing the importance of selective trading and proper risk management."
    },
    {
      title: "Limitations & Future Research",
      content: [
        "Market Condition Sensitivity: Variable performance across different market regimes",
        "Model Complexity: Potential overfitting in certain market conditions requiring regular recalibration",
        "Volatility Challenges: Limited effectiveness in high-volatility stocks and extreme market conditions",
        "Trading Volume Constraints: Some stocks show insufficient trading activity for reliable signal generation",
        "Risk Management Trade-offs: Balance between return potential and risk control",
        "Future Research Directions: Enhanced market regime detection, adaptive parameter optimization, additional data sources (sentiment, alternative data)"
      ],
      chart: "Performance Variation Across Market Regimes",
      notes: "While acknowledging these limitations, they also provide clear directions for future research and model enhancement."
    },
    {
      title: "Conclusion & Recommendations",
      content: [
        "The hybrid CNN-BiLSTM model with attention mechanism demonstrates superior performance compared to traditional methods",
        "Best results achieved in stable, large-cap stocks (WMT: 48.18% return, 72.73% win rate, 3.38% maximum drawdown)",
        "Selective trading strategy (10-15 trades) with strong risk management consistently outperforms high-frequency approaches",
        "Integration of deep learning predictions with technical analysis indicators produces more robust trading signals",
        "SMOTE balancing technique successfully addresses class imbalance issues in trading signal generation",
        "Recommendations: Implement model with selective focus on stable large-cap stocks, emphasize quality over quantity in trade execution, maintain strict risk management protocols"
      ],
      chart: "Performance Dashboard Summary",
      notes: "Our hybrid model successfully bridges the gap between traditional technical analysis and modern deep learning approaches, providing a framework for enhanced trading signal generation with practical implementation considerations."
    },
    {
      title: "References",
      content: [
        "Huang, J., Chai, J., & Cho, S. (2020). Deep learning in finance and banking: A literature review and classification.",
        "Shah, J., Vaidya, D., & Shah, M. (2022). A comprehensive review of multiple hybrid deep learning approaches for stock prediction.",
        "Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review.",
        "Wu, J. M.-T., Li, Z., Herencsar, N., Vo, B., & Lin, J. C.-W. (2023). A graph-based CNN-LSTM stock price prediction algorithm with leading indicators.",
        "Saud, S. & Shakya, S. (2024). Intelligent stock trading strategies using long short-term memory networks."
      ],
      notes: "These key references provide the theoretical foundation for our hybrid model approach and performance evaluation framework."
    },
    {
      title: "Acknowledgments",
      content: [
        "Dr. [Professor Name] - Project Advisor",
        "UNT Advanced Data Analytics Department",
        "AI Research Assistance: Claude 3.5 Sonnet (Anthropic), GitHub Copilot",
        "Data Sources: Yahoo Finance API, Alpha Vantage API"
      ]
    },
    {
      title: "Thank You & Questions",
      content: [
        "Thank you for your attention!",
        "Questions and Discussion",
        "Contact Information:",
        "Biniam Abebe",
        "University of North Texas",
        "ADTA 5900 - Advanced Data Analytics Capstone",
        "Email: BiniamAbebe@my.unt.edu"
      ],
      notes: "Be prepared to discuss: 1) The practical implementation of the model, 2) The significance of the attention mechanism, 3) Risk management considerations, and 4) Future research directions."
    }
  ];

  return (
    <div className="bg-gray-100 p-4 font-sans">
      {slides.map((slide, index) => (
        <div key={index} className="bg-white rounded-lg shadow-lg mb-8 overflow-hidden">
          <div className="bg-green-700 text-white p-6">
            <h2 className="text-2xl font-bold">{slide.title}</h2>
            {slide.subtitle && <h3 className="text-xl mt-2">{slide.subtitle}</h3>}
            {slide.author && <p className="mt-4">{slide.author}</p>}
            {slide.institution && <p>{slide.institution}</p>}
            {slide.department && <p>{slide.department}</p>}
            {slide.date && <p className="mt-2 text-sm">{slide.date}</p>}
          </div>
          
          <div className="p-6">
            {slide.content && (
              <ul className="list-disc pl-6 space-y-2">
                {slide.content.map((item, i) => (
                  <li key={i} className="text-lg">{item}</li>
                ))}
              </ul>
            )}
            
            {slide.chart && (
              <div className="mt-4 p-4 border border-gray-300 rounded bg-gray-50 text-center h-64 flex items-center justify-center">
                <p className="text-gray-500 italic">[{slide.chart}]</p>
              </div>
            )}
            
            {slide.image && (
              <div className="mt-4 p-4 border border-gray-300 rounded bg-gray-50 text-center h-64 flex items-center justify-center">
                <p className="text-gray-500 italic">[{slide.image}]</p>
              </div>
            )}
            
            {slide.notes && (
              <div className="mt-4 p-2 border-t border-gray-200">
                <p className="text-sm text-gray-600 italic">Speaker Notes: {slide.notes}</p>
              </div>
            )}
            
            <div className="mt-4 text-right text-sm text-gray-400">
              Slide {index + 1}/{slides.length}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default Presentation;
