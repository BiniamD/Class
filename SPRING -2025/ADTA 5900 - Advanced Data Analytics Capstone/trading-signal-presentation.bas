Sub CreateDeepLearningPresentation()
    ' Create a new PowerPoint presentation
    Dim ppApp As Object
    Dim ppPres As Object
    Dim ppSlide As Object
    Dim ppShape As Object
    Dim i As Integer
    Dim slideIndex As Integer
    
    ' Create PowerPoint application instance
    Set ppApp = CreateObject("PowerPoint.Application")
    ppApp.Visible = True
    
    ' Create a new presentation
    Set ppPres = ppApp.Presentations.Add
    
    ' Set UNT green color for title slides
    Dim untGreen As Long
    untGreen = RGB(0, 102, 51) ' UNT Green color
    
    ' -------------------------
    ' Slide 1: Title Slide
    ' -------------------------
    slideIndex = 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 1) ' 1 = Title slide layout
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Deep Learning for Enhanced Trading Signal Generation"
    
    ' Subtitle
    ppSlide.Shapes(2).TextFrame.TextRange.Text = "A Hybrid CNN-BiLSTM Model with Attention Mechanism for Stock Market Prediction and Trading Signal Generation"
    
    ' Add author and institution information
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 320, 500, 100)
    With ppShape.TextFrame.TextRange
        .Text = "Biniam Abebe" & vbCrLf & _
                "University of North Texas" & vbCrLf & _
                "Advanced Data Analytics - ADTA 5900 Capstone" & vbCrLf & _
                "April 26, 2025"
        .Font.Size = 16
        .ParagraphFormat.Alignment = 1 ' Center alignment
    End With
    
    ' Change title slide background color to UNT green
    ppSlide.Background.Fill.ForeColor.RGB = untGreen
    
    ' Change title text color to white
    ppSlide.Shapes.Title.TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)
    ppSlide.Shapes(2).TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)
    ppShape.TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "Good afternoon everyone. Thank you for joining me today for this presentation on my capstone project. I'm Biniam Abebe, and I'll be discussing my research on deep learning applications for enhanced trading signal generation in the stock market. This project combines advanced neural network architectures with traditional technical analysis to create a more robust trading system. My research focuses specifically on a hybrid CNN-BiLSTM model with an attention mechanism applied to S&P 500 stocks. Throughout this presentation, I'll share the methodology, key findings, and practical applications of this approach."
    
    ' -------------------------
    ' Slide 2: Agenda
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Agenda"
    
    ' Content
    Dim agendaItems As String
    agendaItems = " Introduction and Problem Statement" & vbCrLf & _
                  " Research Objectives and Questions" & vbCrLf & _
                  " Literature Review Highlights" & vbCrLf & _
                  " Methodology and Data Description" & vbCrLf & _
                  " Model Architecture and Implementation" & vbCrLf & _
                  " Trading Strategy Development" & vbCrLf & _
                  " Experimental Results and Performance Analysis" & vbCrLf & _
                  " Key Findings and Implementation Insights" & vbCrLf & _
                  " Limitations and Future Research" & vbCrLf & _
                  " Conclusion and Recommendations"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = agendaItems
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "Here's an overview of what we'll cover today. We'll begin with the problem statement that motivated this research, followed by the specific research questions I sought to answer. I'll then highlight key findings from the literature that informed my approach. The core of the presentation will focus on the methodology, including data collection, model architecture, and trading strategy implementation. We'll examine the experimental results in detail, analyzing performance across different stocks and market conditions. I'll share key insights for practical implementation and discuss limitations of the current approach. Finally, we'll conclude with recommendations for both researchers and practitioners in this field. Let's start with the introduction."
    
    ' -------------------------
    ' Slide 3: Introduction & Problem Statement
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Introduction & Problem Statement"
    
    ' Content
    Dim introContent As String
    introContent = " Traditional technical analysis is subjective and prone to psychological biases" & vbCrLf & _
                   " Deep learning offers objective pattern recognition capabilities" & vbCrLf & _
                   " U.S. equity market: $7 trillion market cap with 60-70% algorithmic trading" & vbCrLf & _
                   " Proposed solution: Hybrid CNN-BiLSTM model for enhanced trading signal generation" & vbCrLf & _
                   " Focus on S&P 500 stocks with comprehensive technical and fundamental indicators"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = introContent
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "The financial markets present a challenging environment for traditional technical analysis, which often relies heavily on subjective interpretation of chart patterns and indicators. Human traders are inherently limited by cognitive biases, emotional reactions, and an inability to process multiple indicators simultaneously." & vbCrLf & _
                    "Deep learning approaches offer a promising alternative by providing objective pattern recognition capabilities that can process vast amounts of market data. This is particularly relevant in today's U.S." & vbCrLf & _
                    "equity market, which has a market capitalization of approximately $7 trillion, with 60-70% of daily trading volume conducted algorithmically. " & vbCrLf & _
                    "The competitive landscape demands increasingly sophisticated approaches to gain an edge. " & vbCrLf & _
                    "My research proposes a hybrid CNN-BiLSTM model that combines the strengths of convolutional neural networks for spatial pattern recognition with bidirectional long short-term memory networks for temporal" & vbCrLf & _
                    "sequence learning. This approach is applied to S&P 500 stocks using a comprehensive set of technical and fundamental indicators to generate more reliable trading signals than traditional methods alone."
    
    ' -------------------------
    ' Slide 4: Research Objectives & Questions
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Research Objectives & Questions"
    
    ' Content
    Dim researchContent As String
    researchContent = " Primary objective: Develop and validate a hybrid deep learning model for stock market prediction and trading signal generation" & vbCrLf & _
                      " Research Question 1: How effective is a hybrid CNN-BiLSTM model compared to traditional technical analysis?" & vbCrLf & _
                      " Research Question 2: Can the hybrid model provide better trading signal reliability and profitability?" & vbCrLf & _
                      " Research Question 3: How do different market regimes affect model performance?" & vbCrLf & _
                      " Research Question 4: Can risk management be improved through deep learning predictions?"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = researchContent
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "The primary objective of this research was to develop and validate a hybrid deep learning model specifically for stock market prediction and trading signal generation. This led to four key research questions that guided my investigation. First, I wanted to quantitatively assess how effective a hybrid CNN-BiLSTM model is compared to traditional technical analysis methods in terms of prediction accuracy and signal quality. Second, I examined whether this hybrid approach could provide better trading signal reliability and profitability when implemented in a realistic trading framework with transaction costs and risk management. Third, I investigated how different market regimes—bull markets, bear markets, and sideways markets—affect the model's performance," & vbCrLf & _
                "as adaptability to changing market conditions is crucial for any trading system. Finally, I explored whether risk management could be improved by incorporating deep learning predictions into position " & vbCrLf & _
                "sizing and stop-loss placement decisions. These questions directly address the gap between academic research on market prediction and practical implementation considerations for traders and portfolio managers."
                   
    ' -------------------------
    ' Slide 5: Literature Review Highlights
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Literature Review Highlights"
    
    ' Content
    Dim litReviewContent As String
    litReviewContent = " Deep learning excels in handling complex financial data for forecasting (Huang et al., 2020)" & vbCrLf & _
                       " Hybrid CNN-LSTM models consistently outperform standalone architectures in stock prediction (Shah et al., 2022)" & vbCrLf & _
                       " CNN-BiLSTM-AM achieved lowest error rates (MAE: 21.952, RMSE: 31.694) compared to other models" & vbCrLf & _
                       " Technical indicators significantly enhance prediction performance when combined with deep learning (Sezer et al., 2017)" & vbCrLf & _
                       " Graph-based CNN-LSTM with leading indicators improves prediction precision (Wu et al., 2023)" & vbCrLf & _
                       " Performance metrics standardization: Sharpe ratio, win rate, maximum drawdown (Saud & Shakya, 2024)"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = litReviewContent
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "My research builds upon several key findings from the literature on deep learning applications in finance." & vbCrLf & _
            "Huang and colleagues (2020) conducted a comprehensive review demonstrating that deep learning approaches excel in handling complex, " & vbCrLf & _
            "high-dimensional financial data for forecasting purposes. This provided the theoretical foundation for my work. Of particular relevance was the research by Shah et al. (2022)," & vbCrLf & _
            "which showed that hybrid CNN-LSTM models consistently outperform standalone architectures in stock prediction tasks." & vbCrLf & _
            " Their analysis revealed that a CNN-BiLSTM model with an attention mechanism achieved the lowest error rates among tested architectures, with a Mean Absolute Error of 21.952 and Root Mean Square Error of 31.694." & vbCrLf & _
            "Sezer's work in 2017 demonstrated that technical indicators significantly enhance prediction performance when combined with deep learning approaches, validating my decision to incorporate 76 different indicators." & vbCrLf & _
            "Wu and colleagues (2023) further advanced this field by showing that a graph-based CNN-LSTM model with leading indicators improves prediction precision. " & vbCrLf & _
            "Finally, Saud and Shakya's 2024 research provided standardized performance metrics for evaluating trading strategies, including the Sharpe ratio, win rate, and maximum drawdown, which I adopted in my evaluation framework. Collectively, this literature supports the hybrid approach I've implemented while highlighting the importance of comprehensive evaluation."
               
    ' -------------------------
    ' Slide 6: Methodology & Data Description
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Methodology & Data Description"
    
    ' Content
    Dim methodContent As String
    methodContent = " Rich dataset: 501 S&P 500 companies over 5 years (2019-2024)" & vbCrLf & _
                    " Comprehensive feature set: 76 technical and fundamental indicators" & vbCrLf & _
                    " High data quality: Only 1.9% missing values" & vbCrLf & _
                    " Data sources: Daily price data (Yahoo Finance API), Market metrics (Alpha Vantage API)" & vbCrLf & _
                    " Data processing: Imputation, outlier detection, time series alignment" & vbCrLf & _
                    " Time series analysis: Volatility clustering, market regimes, seasonal patterns"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = methodContent
    
    ' Add placeholder for chart
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 340, 500, 100)
    ppShape.TextFrame.TextRange.Text = "[Stock Price Return Distribution]"
    ppShape.TextFrame.TextRange.Font.Italic = True
    ppShape.TextFrame.TextRange.Font.Color.RGB = RGB(128, 128, 128)
    ppShape.TextFrame.TextRange.ParagraphFormat.Alignment = 2 ' Center
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "For this research, I assembled a comprehensive dataset covering 501 S&P 500 companies over a five-year period from 2019 to 2024. This dataset is particularly valuable because it includes 76 different technical and fundamental indicators, providing a rich feature set for model training. The data quality is exceptionally high, with only 1.9% missing values across all variables. Data was sourced primarily from the Yahoo Finance API for daily price data and the Alpha Vantage API for additional market metrics. The data processing pipeline included several critical steps: imputation of missing values using forward-fill for time series data, outlier detection using rolling z-scores, and time series alignment to ensure consistent trading days across all stocks. Initial exploratory data analysis revealed significant volatility clustering effects, distinct market regimes with different characteristics, and clear seasonal patterns in both returns" & vbCrLf & _
    " and volatility. The chart shown here illustrates the stock price return distribution, which exhibits a slight negative skewness typical of equity markets, with fat tails indicating more extreme events than would be predicted by a normal distribution. This comprehensive dataset provided a solid foundation for developing and testing our hybrid model."
    
    ' -------------------------
    ' Slide 7: Data Processing & Feature Engineering
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Data Processing & Feature Engineering"
    
    ' Content
    Dim featureContent As String
    featureContent = " Price-based features: OHLC, returns, log returns, price ranges" & vbCrLf & _
                     " Technical indicators across multiple timeframes: Moving averages (5, 10, 20, 50, 200 days), RSI (9, 14, 25 periods), MACD, Bollinger Bands" & vbCrLf & _
                     " Market features: Market returns, volatility measures, VIX data, rolling beta calculations" & vbCrLf & _
                     " Fundamental features: PE ratio, PB ratio, dividend yield, profit margin, enterprise value" & vbCrLf & _
                     " Class imbalance handling with SMOTE: Balanced trading signals (50-50 distribution)" & vbCrLf & _
                     " Mathematical formulations: Moving Averages, RSI, MACD, Bollinger Bands"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = featureContent
    
    ' Add placeholder for chart
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 340, 500, 100)
    ppShape.TextFrame.TextRange.Text = "[Feature Correlation Heatmap]"
    ppShape.TextFrame.TextRange.Font.Italic = True
    ppShape.TextFrame.TextRange.Font.Color.RGB = RGB(128, 128, 128)
    ppShape.TextFrame.TextRange.ParagraphFormat.Alignment = 2 ' Center
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "Feature engineering was a critical component of this research, as the quality of input features directly impacts model performance. I categorized features into four main groups. First, price-based features included standard OHLC (Open, High, Low, Close) data along with derived metrics such as returns, log returns, and price ranges. Second, I calculated technical indicators across multiple timeframes, including moving averages spanning 5 to 200 days, RSI with various lookback periods (9, 14, and 25 days), MACD, and Bollinger Bands. Third, market-level features captured broader market dynamics through metrics like market returns, volatility measures, VIX data, and rolling beta calculations. Fourth, fundamental features provided context about company characteristics through metrics like PE ratio, PB ratio, dividend yield, profit margin, and enterprise value. A major challenge in financial prediction is class imbalance," & vbCrLf & _
    "as profitable trading opportunities typically represent a minority class. To address this, I implemented the Synthetic Minority Over-sampling Technique (SMOTE), which balanced the distribution of trading signals to a 50-50 ratio. This significantly improved the model's ability to identify profitable trades. The feature correlation heatmap displayed here shows the relationships between these features, highlighting clusters of related indicators and helping identify redundancies in the feature set. These engineered features provided a comprehensive view of market conditions and stock behavior for the deep learning model."
    
    ' -------------------------
    ' Slide 8: Deep Learning Architecture
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Deep Learning Architecture"
    
    ' Content
    Dim architectureContent As String
    architectureContent = " CNN component: Processes local patterns through 64 filters with kernel size 3, followed by max pooling and dropout (0.2)" & vbCrLf & _
                          " BiLSTM structure: Three stacked layers (128, 32, 32 units) with bidirectional processing for enhanced temporal feature capture" & vbCrLf & _
                          " Attention mechanism: SoftMax-activated scoring system to focus on relevant temporal patterns" & vbCrLf & _
                          " Attention formula: Attention Score = softmax(W · ht + b), where ht is the hidden state" & vbCrLf & _
                          " Training split: 70% training, 15% validation, 15% testing" & vbCrLf & _
                          " Optimization: Adam optimizer (learning rate 0.001), batch size 32, 50 epochs"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = architectureContent
    
    ' Add placeholder for image
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 340, 500, 100)
    ppShape.TextFrame.TextRange.Text = "[CNN-BiLSTM with Attention Architecture Diagram]"
    ppShape.TextFrame.TextRange.Font.Italic = True
    ppShape.TextFrame.TextRange.Font.Color.RGB = RGB(128, 128, 128)
    ppShape.TextFrame.TextRange.ParagraphFormat.Alignment = 2 ' Center
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "The heart of this research is the hybrid deep learning architecture that combines the strengths of multiple neural network types. The model begins with a CNN component designed to process local patterns in the input data. This component uses 64 filters with a kernel size of 3, followed by max pooling to reduce dimensionality and dropout regularization with a rate of 0.2 to prevent overfitting. The CNN layer is particularly effective at identifying local patterns across multiple technical indicators simultaneously. Next, the architecture incorporates a BiLSTM structure consisting of three stacked layers with 128, 32, and 32 units respectively. The bidirectional processing allows the model to consider both past and future context in the time series, providing a more comprehensive view of temporal patterns than a standard LSTM. What makes this architecture particularly powerful is the addition of an attention mechanism," & vbCrLf & _
    "which uses a SoftMax-activated scoring system to focus on the most relevant temporal patterns in the data. The attention score is calculated using the formula Attention Score = softmax(W · ht + b), where ht represents the hidden state at time t, and W and b are learnable parameters. For training, I split the data chronologically with 70% for training, 15% for validation, and 15% for testing, ensuring no look-ahead bias. The model was optimized using the Adam optimizer with a learning rate of 0.001, a batch size of 32, and trained for 50 epochs with early stopping to prevent overfitting. This complex architecture allowed the model to capture both spatial and temporal patterns in the market data, leading to more accurate predictions and trading signals."
    
    ' -------------------------
    ' Slide 9: Trading Strategy Development
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Trading Strategy Development"
    
    ' Content
    Dim strategyContent As String
    strategyContent = " Signal generation: Probability threshold system (Signal = 1 if probability > 0.60, 0 otherwise)" & vbCrLf & _
                      " Position sizing: Dynamic allocation based on model confidence (linear scaling)" & vbCrLf & _
                      " Technical confirmation framework: Moving Averages (50, 200-day), RSI, MACD, Bollinger Bands" & vbCrLf & _
                      " Risk management protocols: Stop-loss (2% below entry), Take-profit (5% above entry), Maximum holding period (30 days)" & vbCrLf & _
                      " Market regime adaptation: Strategy parameters adjusted based on detected market conditions" & vbCrLf & _
                      " Performance evaluation: Standard financial metrics (Sharpe Ratio, Maximum Drawdown, Win Rate, Profit Factor)"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = strategyContent
    
    ' Add placeholder for chart
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 340, 500, 100)
    ppShape.TextFrame.TextRange.Text = "[Strategy Performance by Market Regime]"
    ppShape.TextFrame.TextRange.Font.Italic = True
    ppShape.TextFrame.TextRange.Font.Color.RGB = RGB(128, 128, 128)
    ppShape.TextFrame.TextRange.ParagraphFormat.Alignment = 2 ' Center
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "Translating model predictions into a practical trading strategy required careful consideration of signal generation, position sizing, and risk management. For signal generation, I implemented a probability threshold system where a trade signal is generated only when the model's predicted probability exceeds 0.60. This threshold was determined through extensive backtesting to optimize the trade-off between signal frequency and accuracy. Position sizing employed a dynamic allocation approach based on model confidence, using linear scaling to allocate more capital to higher-confidence trades. To enhance reliability, I developed a technical confirmation framework that validates model predictions using traditional indicators: Moving Averages (50 and 200-day), RSI, MACD, and Bollinger Bands. This hybrid approach leverages both the pattern recognition capabilities of deep learning and the established reliability of technical analysis." & vbCrLf & _
    "Risk management was implemented through three key protocols: a fixed stop-loss at 2% below entry price to limit downside risk, a take-profit target at 5% above entry to secure gains, and a maximum holding period of 30 days to prevent capital from being tied up in non-performing positions. The strategy also incorporates market regime adaptation, adjusting parameters based on detected market conditions such as bull markets, bear markets, or sideways markets. Performance evaluation used standard financial metrics including the Sharpe Ratio to measure risk-adjusted returns, Maximum Drawdown to quantify downside risk, Win Rate to assess prediction accuracy, and Profit Factor to measure overall profitability. This chart shows how the strategy performed across different market regimes, highlighting its adaptability to changing conditions."
    
    ' -------------------------
    ' Slide 10: Experimental Results
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Experimental Results"
    
    ' Content
    Dim resultsContent As String
    resultsContent = " Top performer: Walmart (WMT) - 48.18% return, 72.73% win rate, 3.38% maximum drawdown" & vbCrLf & _
                     " Strong performers: Mastercard (MA) - 19.45% risk-adjusted return, 50% win rate" & vbCrLf & _
                     " Portfolio average across all stocks: 15.4% return, 1.85 Sharpe ratio, 58.6% win rate" & vbCrLf & _
                     " Risk management effectiveness: Maximum drawdowns < 5% for top performers" & vbCrLf & _
                     " Market condition sensitivity: Better performance in stable market environments" & vbCrLf & _
                     " Trading frequency finding: Selective trading (10-15 trades) outperformed high-frequency trading (40+ trades)"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = resultsContent
    
    ' Add placeholder for chart
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 340, 500, 100)
    ppShape.TextFrame.TextRange.Text = "[Performance Metrics Heatmap]"
    ppShape.TextFrame.TextRange.Font.Italic = True
    ppShape.TextFrame.TextRange.Font.Color.RGB = RGB(128, 128, 128)
    ppShape.TextFrame.TextRange.ParagraphFormat.Alignment = 2 ' Center
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "The experimental results demonstrated the effectiveness of our hybrid approach across a range of S&P 500 stocks. The standout performer was Walmart (WMT), which achieved an impressive 48.18% return with a 72.73% win rate and a remarkably low maximum drawdown of just 3.38%. This exemplifies the model's ability to generate reliable trading signals for stable, large-cap stocks. Other strong performers included Mastercard (MA), which delivered a 19.45% risk-adjusted return with a 50% win rate. Looking at the portfolio as a whole, the average performance across all tested stocks was a 15.4% return with a Sharpe ratio of 1.85 and a win rate of 58.6%. These metrics compare favorably to both traditional technical analysis approaches and market benchmarks. The risk management protocols proved particularly effective, with top-performing stocks maintaining maximum drawdowns below 5%," & vbCrLf & _
    "highlighting the strategy's ability to control downside risk while capturing upside potential. Analysis of performance across different market conditions revealed better results in stable market environments compared to highly volatile periods, suggesting opportunities for further optimization in high-volatility scenarios. Perhaps the most surprising finding was related to trading frequency: selective trading strategies with fewer trades (10-15 over the test period) significantly outperformed high-frequency approaches with 40+ trades. This contradicts the common assumption that more frequent trading leads to higher returns and suggests that focusing on high-quality signals is more important than trading volume. The performance metrics heatmap visualizes these results across multiple stocks and metrics, providing a comprehensive view of strategy performance."
    
    ' -------------------------
    ' Slide 11: Performance Analysis
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Performance Analysis"
    
    ' Content
    Dim perfAnalysisContent As String
    perfAnalysisContent = " Comparative analysis: Strategy outperformed baseline by 12.3% while maintaining lower volatility" & vbCrLf & _
                          " Stock category analysis: Large-cap retail (WMT) and financial (MA, JPM) sectors showed strongest performance" & vbCrLf & _
                          " Technology sector challenges: High-volatility stocks like NVIDIA showed mixed results (-21.65% return, 31.82% win rate)" & vbCrLf & _
                          " Win rate correlation: Strong positive relationship between win rate and total return (R² = 0.78)" & vbCrLf & _
                          " Risk-return profile: Top performers maintained exceptional risk-adjusted returns (Sharpe ratios > 2.0)"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = perfAnalysisContent
    
    ' Add placeholder for chart
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 340, 500, 100)
    ppShape.TextFrame.TextRange.Text = "[Scatter Plot: Win Rate vs. Total Return]"
    ppShape.TextFrame.TextRange.Font.Italic = True
    ppShape.TextFrame.TextRange.Font.Color.RGB = RGB(128, 128, 128)
    ppShape.TextFrame.TextRange.ParagraphFormat.Alignment = 2 ' Center
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "Deeper analysis of the performance results revealed several important patterns and relationships. Compared to our baseline strategy using traditional technical analysis alone, the hybrid approach outperformed by a substantial 12.3% while simultaneously maintaining lower volatility, demonstrating the value of integrating deep learning with established technical methods. When analyzing performance by stock category, we found that large-cap retail stocks like Walmart and financial sector stocks like Mastercard and JPMorgan Chase showed the strongest and most consistent performance. In contrast, the technology sector presented challenges, with high-volatility stocks like NVIDIA showing mixed results—a negative 21.65% return and a low win rate of 31.82%. This suggests that the model may require sector-specific optimization or additional features to handle the unique characteristics of technology stocks." & vbCrLf & _
    "One of the most striking findings was the negative correlation between trading frequency and performance. NVIDIA, with 44 trades, produced a negative 21.65% return, while Walmart, with just 11 trades, achieved a positive 48.18% return. This inverse relationship was consistent across the dataset and challenges conventional wisdom about algorithmic trading. The scatter plot displayed here shows the strong positive relationship between win rate and total return, with an R-squared value of 0.78, indicating that improving predictive accuracy directly translates to better financial performance. From a risk-return perspective, the top-performing stocks maintained exceptional risk-adjusted returns with Sharpe ratios exceeding 2.0, well above the threshold typically considered excellent in investment management. These findings collectively suggest that a selective, high-quality trading approach focused on stable large-cap stocks provides the best results with this hybrid model."
    
    ' -------------------------
    ' Slide 12: Key Findings & Implementation Insights
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Key Findings & Implementation Insights"
    
    ' Content
    Dim insightsContent As String
    insightsContent = " Trading Frequency: Quality over quantity - selective trading (10-15 trades) significantly outperforms high-frequency trading" & vbCrLf & _
                      " Stock Selection: Large-cap stability (WMT, MA, JPM) delivers consistent performance compared to volatile tech stocks" & vbCrLf & _
                      " Attention Mechanism: Successfully identifies the most important temporal patterns and market regime shifts" & vbCrLf & _
                      " SMOTE Balancing: Dramatically improves model's ability to identify profitable trading opportunities" & vbCrLf & _
                      " Risk Management: Confidence-based position sizing and adaptive stop-losses maintain low drawdowns" & vbCrLf & _
                      " Technical Confirmation: Integration of model predictions with technical indicators produces robust trading signals"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = insightsContent
    
    ' Add placeholder for image
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 340, 500, 100)
    ppShape.TextFrame.TextRange.Text = "[Cumulative Return Comparison for WMT]"
    ppShape.TextFrame.TextRange.Font.Italic = True
    ppShape.TextFrame.TextRange.Font.Color.RGB = RGB(128, 128, 128)
    ppShape.TextFrame.TextRange.ParagraphFormat.Alignment = 2 ' Center
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "The key findings from this research provide practical insights for implementing hybrid deep learning models in real-world trading scenarios." & vbCrLf & _
    "First, regarding trading frequency, our results clearly demonstrate that quality trumps quantity. Selective trading approaches with just 10-15 well-chosen trades significantly outperformed high-frequency strategies with 40+ trades. This suggests that focusing on high-confidence signals and being patient for optimal entry points is more effective than frequent trading." & vbCrLf & _
    "Second, stock selection proved critical to performance. Large-cap stocks with stable characteristics, particularly in the retail and financial sectors like Walmart, Mastercard, and JPMorgan Chase, delivered the most consistent results. These stocks typically have lower volatility, higher liquidity, and more predictable behavior than technology stocks, which showed more variable performance." & vbCrLf & _
    "Third, the attention mechanism component of our model was particularly valuable, successfully identifying the most important temporal patterns and market regime shifts. This helped the model adapt to changing market conditions and focus on the most relevant features at different times." & vbCrLf & _
    "Fourth, the implementation of SMOTE for handling class imbalance dramatically improved the model's ability to identify profitable trading opportunities by preventing bias toward the majority class." & vbCrLf & _
    "Fifth, our risk management approach, combining confidence-based position sizing with adaptive stop-losses, was crucial for maintaining low drawdowns even during market turbulence. " & vbCrLf & _
    "Finally, the integration of model predictions with traditional technical indicators produced more robust trading signals than either approach alone, highlighting the value of this hybrid methodology. This chart shows the cumulative return comparison for Walmart, illustrating the substantial outperformance of our approach compared to both the market benchmark and traditional technical analysis."
    
    ' -------------------------
    ' Slide 13: Limitations & Future Research
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Limitations & Future Research"
    
    ' Content
    Dim limitationsContent As String
    limitationsContent = " Market Condition Sensitivity: Variable performance across different market regimes" & vbCrLf & _
                         " Model Complexity: Potential overfitting in certain market conditions requiring regular recalibration" & vbCrLf & _
                         " Volatility Challenges: Limited effectiveness in high-volatility stocks and extreme market conditions" & vbCrLf & _
                         " Trading Volume Constraints: Some stocks show insufficient trading activity for reliable signal generation" & vbCrLf & _
                         " Risk Management Trade-offs: Balance between return potential and risk control" & vbCrLf & _
                         " Future Research Directions: Enhanced market regime detection, adaptive parameter optimization, additional data sources (sentiment, alternative data)"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = limitationsContent
    
    ' Add placeholder for chart
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 340, 500, 100)
    ppShape.TextFrame.TextRange.Text = "[Performance Variation Across Market Regimes]"
    ppShape.TextFrame.TextRange.Font.Italic = True
    ppShape.TextFrame.TextRange.Font.Color.RGB = RGB(128, 128, 128)
    ppShape.TextFrame.TextRange.ParagraphFormat.Alignment = 2 ' Center
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "While the results of this research are promising, it's important to acknowledge several limitations and areas for future improvement." & vbCrLf & _
    "First, the model shows market condition sensitivity, with variable performance across different market regimes. Performance was strongest in stable, trending markets but less reliable during highly volatile periods or rapid regime shifts. This suggests the need for regime-specific models or adaptive parameters that can adjust to changing market conditions in real-time." & vbCrLf & _
    "Second, model complexity introduces challenges related to potential overfitting in certain market conditions. The hybrid architecture, while powerful, contains many parameters that require regular recalibration as market dynamics evolve." & vbCrLf & _
    "Third, volatility challenges were evident in the limited effectiveness of the approach when applied to high-volatility stocks like those in the technology sector, as well as during extreme market conditions such as the COVID-19 crash." & vbCrLf & _
    "Fourth, trading volume constraints affected some stocks with insufficient trading activity for reliable signal generation, limiting the universe of applicable securities. " & vbCrLf & _
    "Fifth, risk management involves inherent trade-offs between return potential and risk control, with more aggressive settings potentially yielding higher returns but also higher drawdowns. " & vbCrLf & _
    "Future research directions should address these limitations through enhanced market regime detection techniques that can more accurately identify and adapt to changing market conditions, adaptive parameter optimization that dynamically adjusts model parameters based on current market characteristics, and the integration of additional data sources such as sentiment analysis and alternative data to capture aspects of market behavior not reflected in price and volume alone. The chart displayed here illustrates the performance variation across different market regimes, highlighting the need for regime-specific optimization." & vbCrLf & _
    "These limitations, while significant, also present opportunities for further refinement and improvement of the hybrid approach."
    
    ' -------------------------
    ' Slide 14: Conclusion & Recommendations
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Conclusion & Recommendations"
    
    ' Content
    Dim conclusionContent As String
    conclusionContent = " The hybrid CNN-BiLSTM model with attention mechanism demonstrates superior performance compared to traditional methods" & vbCrLf & _
                        " Best results achieved in stable, large-cap stocks (WMT: 48.18% return, 72.73% win rate, 3.38% maximum drawdown)" & vbCrLf & _
                        " Selective trading strategy (10-15 trades) with strong risk management consistently outperforms high-frequency approaches" & vbCrLf & _
                        " Integration of deep learning predictions with technical analysis indicators produces more robust trading signals" & vbCrLf & _
                        " SMOTE balancing technique successfully addresses class imbalance issues in trading signal generation" & vbCrLf & _
                        " Recommendations: Implement model with selective focus on stable large-cap stocks, emphasize quality over quantity in trade execution, maintain strict risk management protocols"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = conclusionContent
    
    ' Add placeholder for chart
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 340, 500, 100)
    ppShape.TextFrame.TextRange.Text = "[Performance Dashboard Summary]"
    ppShape.TextFrame.TextRange.Font.Italic = True
    ppShape.TextFrame.TextRange.Font.Color.RGB = RGB(128, 128, 128)
    ppShape.TextFrame.TextRange.ParagraphFormat.Alignment = 2 ' Center
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "In conclusion, this research demonstrates that a hybrid CNN-BiLSTM model with an attention mechanism can significantly enhance trading signal generation compared to traditional technical analysis methods. The empirical results show that this approach delivers superior risk-adjusted returns while maintaining lower drawdowns. The best results were achieved in stable, large-cap stocks such as Walmart, which produced an impressive 48.18% return with a 72.73% win rate and a minimal 3.38% maximum drawdown. " & vbCrLf & _
"One of the most significant findings was that a selective trading strategy executing only 10-15 high-quality trades consistently outperformed high-frequency approaches with more frequent trading. " & vbCrLf & _
"This challenges conventional wisdom in algorithmic trading and suggests that model confidence and signal quality should be prioritized over trading volume. " & vbCrLf & _
"The integration of deep learning predictions with traditional technical analysis indicators proved particularly effective, producing more robust trading signals than either approach alone. " & vbCrLf & _
"This hybrid methodology leverages the pattern recognition capabilities of neural networks while maintaining the interpretability and reliability of established technical indicators. " & vbCrLf & _
"Additionally, the SMOTE balancing technique successfully addressed class imbalance issues in trading signal generation, dramatically improving the model's ability to identify profitable opportunities. " & vbCrLf & _
"Based on these findings, I recommend implementing this hybrid model with a selective focus on stable large-cap stocks, particularly in the retail and financial sectors, while avoiding highly volatile technology stocks." & vbCrLf & _
" Practitioners should emphasize quality over quantity in trade execution, waiting for high-confidence signals rather than trading frequently." & vbCrLf & _
" Finally, maintaining strict risk management protocols, including position sizing based on model confidence and appropriate stop-loss levels, is crucial for controlling drawdowns and achieving consistent returns. This performance dashboard summary visualizes the key metrics across different stocks and strategies, highlighting the effectiveness of our hybrid approach."
    
    ' -------------------------
    ' Slide 15: References
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "References"
    
    ' Content
    Dim referencesContent As String
    referencesContent = " Huang, J., Chai, J., & Cho, S. (2020). Deep learning in finance and banking: A literature review and classification." & vbCrLf & _
                        " Shah, J., Vaidya, D., & Shah, M. (2022). A comprehensive review of multiple hybrid deep learning approaches for stock prediction." & vbCrLf & _
                        " Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2020). Financial time series forecasting with deep learning: A systematic literature review." & vbCrLf & _
                        " Wu, J. M.-T., Li, Z., Herencsar, N., Vo, B., & Lin, J. C.-W. (2023). A graph-based CNN-LSTM stock price prediction algorithm with leading indicators." & vbCrLf & _
                        " Saud, S. & Shakya, S. (2024). Intelligent stock trading strategies using long short-term memory networks."
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = referencesContent
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "This research builds upon a substantial body of literature in the fields of deep learning," & vbCrLf & _
 "financial time series analysis, and algorithmic trading. Huang, Chai, and Cho's 2020 review provided a comprehensive classification of deep learning applications in finance and banking," & vbCrLf & _
 "establishing the theoretical foundation for this work. Shah, Vaidya, and Shah's 2022 paper specifically reviewed hybrid deep learning approaches for stock prediction, " & vbCrLf & _
 "highlighting the advantages of combining different neural network architectures. Sezer, Gudelek, and Ozbayoglu's 2020 systematic literature review of financial time series forecasting with deep " & vbCrLf & _
 "learning offered valuable insights into methodology and evaluation frameworks. Wu and colleagues' 2023 research on graph-based CNN-LSTM stock price prediction with leading indicators informed our approach to feature engineering and model architecture. Finally, Saud and Shakya's 2024 work on intelligent stock trading strategies using LSTM " & vbCrLf & _
 "networks provided important benchmarks for performance comparison." & vbCrLf & _
 "These references, along with many others cited in the full paper, contribute to the theoretical and methodological framework of this research, positioning it within the broader context of advanced analytical approaches to financial markets."
    
    ' -------------------------
    ' Slide 16: Acknowledgments
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 2) ' 2 = Title and content
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Acknowledgments"
    
    ' Content
    Dim ackContent As String
    ackContent = " Dr. [Professor Name] - Project Advisor" & vbCrLf & _
                 " UNT Advanced Data Analytics Department" & vbCrLf & _
                 " AI Research Assistance: Claude 3.5 Sonnet (Anthropic), GitHub Copilot" & vbCrLf & _
                 " Data Sources: Yahoo Finance API, Alpha Vantage API"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = ackContent
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "I would like to express my sincere gratitude to several individuals and organizations who made this research possible." & vbCrLf & _
 "First, I want to thank my project advisor, Dr. [Professor Name], whose guidance, expertise, and feedback were invaluable throughout the research process. " & vbCrLf & _
 "Their insights into both the theoretical foundations and practical applications of deep learning in finance significantly strengthened this work. " & vbCrLf & _
 "I also want to acknowledge the support of the University of North Texas Advanced Data Analytics Department, which provided the resources, academic environment, and technical infrastructure necessary for conducting this research. This project benefited from AI research assistance tools, including Claude 3.5 Sonnet from Anthropic and GitHub Copilot, " & vbCrLf & _
 "which helped with data analysis, code development, and manuscript preparation. The data for this research was sourced primarily from the Yahoo Finance API and Alpha Vantage API, and I appreciate the accessibility and quality of these data sources. Finally, I want to thank my fellow students and colleagues who provided feedback, engaged in constructive discussions, and contributed to the refinement of ideas presented in this research."
    
    ' -------------------------
    ' Slide 17: Thank You & Questions
    ' -------------------------
    slideIndex = slideIndex + 1
    Set ppSlide = ppPres.Slides.Add(slideIndex, 1) ' 1 = Title slide layout
    
    ' Title
    ppSlide.Shapes.Title.TextFrame.TextRange.Text = "Thank You & Questions"
    
    ' Content
    Dim thanksContent As String
    thanksContent = "Thank you for your attention!" & vbCrLf & vbCrLf & _
                    "Questions and Discussion" & vbCrLf & vbCrLf & _
                    "Contact Information:" & vbCrLf & _
                    "Biniam Abebe" & vbCrLf & _
                    "University of North Texas" & vbCrLf & _
                    "ADTA 5900 - Advanced Data Analytics Capstone" & vbCrLf & _
                    "Email: BiniamAbebe@my.unt.edu"
    
    ppSlide.Shapes(2).TextFrame.TextRange.Text = thanksContent
    
    ' Change title slide background color to UNT green
    ppSlide.Background.Fill.ForeColor.RGB = untGreen
    
    ' Change title text color to white
    ppSlide.Shapes.Title.TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)
    ppSlide.Shapes(2).TextFrame.TextRange.Font.Color.RGB = RGB(255, 255, 255)
    
    ' Add speaking notes
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = "Thank you all for your attention and engagement during this presentation. " & vbCrLf & _
"I hope this research has provided valuable insights into the application of hybrid deep learning models for trading signal generation. " & vbCrLf & _
"I'm now happy to open the floor for questions and discussion. I'm particularly prepared to discuss the practical implementation considerations of the model, " & vbCrLf & _
"including computational requirements and integration with existing trading systems; the significance of the attention mechanism in improving prediction accuracy and adaptability; " & vbCrLf & _
"risk management considerations and how they can be customized for different investment objectives; and future research directions that could further enhance this approach. " & vbCrLf & _
"If you have questions about specific aspects of the methodology, results, or implications, I'd be delighted to address them. For those interested in following up after today's presentation, " & vbCrLf & _
"my contact information is displayed here. Thank you again for your time and attention."
    
    ' Add slide numbers to all slides except the title and thank you slides
    For i = 2 To slideIndex - 1
        ppPres.Slides(i).HeadersFooters.SlideNumber.Visible = True
    Next i
    
    ' Save the presentation
    ppPres.SaveAs "Deep_Learning_Trading_Signal_Generation.pptx"
    
    ' Message to confirm completion
    MsgBox "Presentation created successfully!", vbInformation, "Completed"
    
    ' Release objects
    Set ppShape = Nothing
    Set ppSlide = Nothing
    Set ppPres = Nothing
    Set ppApp = Nothing
End Sub

Sub AddSlideNumber(ppSlide As Object)
    ' Add slide number
    With ppSlide.HeadersFooters
        .SlideNumber.Visible = msoTrue
    End With
End Sub

Sub FormatSlideTitle(ppSlide As Object, titleText As String)
    ' Format slide title
    With ppSlide.Shapes.Title.TextFrame.TextRange
        .Text = titleText
        .Font.Size = 32
        .Font.Bold = msoTrue
    End With
End Sub

Sub AddNoteToSlide(ppSlide As Object, noteText As String)
    ' Add note to slide
    ppSlide.NotesPage.Shapes(2).TextFrame.TextRange.Text = noteText
End Sub

Sub AddImagePlaceholder(ppSlide As Object, imageName As String)
    ' Add a placeholder for images
    Dim ppShape As Object
    Set ppShape = ppSlide.Shapes.AddTextbox(1, 100, 340, 500, 100)
    
    With ppShape.TextFrame.TextRange
        .Text = "[" & imageName & "]"
        .Font.Italic = True
        .Font.Color.RGB = RGB(128, 128, 128)
        .ParagraphFormat.Alignment = 2 ' Center
    End With
End Sub


