import pytest
import streamlit as st
from unittest.mock import Mock, patch
from src.streamlit_app import FinancialIQApp
from src.rag_system import RAGSystem
from src.visualization import FinancialVisualizer
from src.logger import FinancialIQLogger

@pytest.fixture
def mock_rag_system():
    return Mock(spec=RAGSystem)

@pytest.fixture
def mock_visualizer():
    return Mock(spec=FinancialVisualizer)

@pytest.fixture
def mock_logger():
    return Mock(spec=FinancialIQLogger)

@pytest.fixture
def app(mock_rag_system, mock_visualizer, mock_logger):
    with patch('streamlit.session_state', {}):
        app = FinancialIQApp(
            rag_system=mock_rag_system,
            visualizer=mock_visualizer,
            logger=mock_logger
        )
        yield app

@pytest.fixture
def sample_query_result():
    return {
        'answer': 'The revenue for 2023 was $1,000,000',
        'sources': ['test.pdf'],
        'confidence': 0.95,
        'context': {
            'financial_metrics': {
                'revenue': 1000000,
                'net_income': 500000
            }
        }
    }

def test_init(app):
    """Test app initialization"""
    assert app.rag_system is not None
    assert app.visualizer is not None
    assert app.logger is not None
    assert 'uploaded_files' in st.session_state
    assert 'query_history' in st.session_state

@patch('streamlit.file_uploader')
def test_file_upload(mock_uploader, app):
    """Test file upload handling"""
    mock_file = Mock()
    mock_file.name = 'test.pdf'
    mock_uploader.return_value = [mock_file]
    
    with patch('tempfile.NamedTemporaryFile'):
        app.handle_file_upload()
        
        app.rag_system.process_documents.assert_called_once()
        app.logger.info.assert_called_with('Files uploaded successfully')

@patch('streamlit.text_input')
def test_query_input(mock_input, app, sample_query_result):
    """Test query input handling"""
    mock_input.return_value = 'What is the revenue?'
    app.rag_system.query.return_value = sample_query_result
    
    app.handle_query()
    
    app.rag_system.query.assert_called_once_with('What is the revenue?')
    assert len(st.session_state['query_history']) == 1
    app.logger.info.assert_called_with('Query processed successfully')

def test_display_results(app, sample_query_result):
    """Test results display"""
    st.session_state['query_history'] = [
        {
            'query': 'What is the revenue?',
            'result': sample_query_result
        }
    ]
    
    with patch('streamlit.write') as mock_write:
        app.display_results()
        
        mock_write.assert_called()
        app.visualizer.create_financial_metrics_comparison.assert_called_once()

@patch('streamlit.sidebar')
def test_sidebar_controls(mock_sidebar, app):
    """Test sidebar controls"""
    with patch('streamlit.selectbox') as mock_select:
        mock_select.return_value = 'Financial Metrics'
        
        app.render_sidebar()
        
        mock_select.assert_called()
        assert st.session_state.get('visualization_type') == 'Financial Metrics'

def test_error_handling(app):
    """Test error handling in app"""
    app.rag_system.query.side_effect = Exception('Query failed')
    
    with patch('streamlit.error') as mock_error:
        app.handle_query()
        
        mock_error.assert_called_with('Error processing query: Query failed')
        app.logger.error.assert_called_with('Query failed: Query failed')

def test_visualization_update(app, sample_query_result):
    """Test visualization updates"""
    st.session_state['visualization_type'] = 'Financial Trends'
    st.session_state['query_history'] = [
        {
            'query': 'What is the revenue trend?',
            'result': sample_query_result
        }
    ]
    
    with patch('streamlit.plotly_chart') as mock_plot:
        app.update_visualization()
        
        app.visualizer.create_financial_trends.assert_called_once()
        mock_plot.assert_called_once()

def test_session_state_management(app):
    """Test session state management"""
    app.clear_session()
    
    assert st.session_state['uploaded_files'] == []
    assert st.session_state['query_history'] == []
    app.logger.info.assert_called_with('Session state cleared')

@patch('streamlit.download_button')
def test_export_results(mock_download, app, sample_query_result):
    """Test results export functionality"""
    st.session_state['query_history'] = [
        {
            'query': 'What is the revenue?',
            'result': sample_query_result
        }
    ]
    
    app.export_results()
    
    mock_download.assert_called_once()
    app.logger.info.assert_called_with('Results exported successfully')

def test_cache_management(app):
    """Test cache management in app"""
    with patch('streamlit.button') as mock_button:
        mock_button.return_value = True
        
        app.manage_cache()
        
        app.rag_system.cache_manager.clear_expired_cache.assert_called_once()
        app.logger.info.assert_called_with('Cache cleaned successfully') 