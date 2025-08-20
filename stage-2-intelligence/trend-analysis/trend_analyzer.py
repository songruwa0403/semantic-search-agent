"""
📈 Trend Analysis System
Temporal pattern analysis for fitness discussion trends

This module implements comprehensive trend analysis including:
- Time-series analysis of discussion patterns
- Seasonal trend detection and forecasting
- Category-based trend comparison
- Anomaly detection in discussion patterns

Author: ML Engineering Portfolio Project
"""

import numpy as np
import pandas as pd
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Time series and statistical analysis
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrendResult:
    """Container for trend analysis results"""
    temporal_patterns: Dict[str, Dict]  # time-based analysis
    category_trends: Dict[str, Dict]    # category-specific trends
    seasonal_patterns: Dict[str, Dict]  # seasonal analysis
    anomalies: List[Dict]              # detected anomalies
    forecasts: Dict[str, Dict]         # trend forecasts
    insights: List[str]                # key insights
    metadata: Dict[str, any]           # analysis metadata

class TrendAnalyzer:
    """
    Comprehensive trend analysis system for fitness discussions
    
    This class provides temporal analysis capabilities including:
    - Time-series pattern detection
    - Seasonal trend analysis
    - Category-based trend comparison
    - Anomaly detection and forecasting
    """
    
    def __init__(self, 
                 time_column: str = 'created_utc',
                 text_column: str = 'text',
                 min_data_points: int = 10):
        """
        Initialize trend analysis system
        
        Args:
            time_column: Column name for timestamp data
            text_column: Column name for text content
            min_data_points: Minimum data points required for trend analysis
        """
        self.time_column = time_column
        self.text_column = text_column
        self.min_data_points = min_data_points
        
        # Analysis results
        self.trend_result = None
        self.is_fitted = False
    
    def analyze_trends(self,
                      df: pd.DataFrame,
                      categories: Optional[Dict[str, List[str]]] = None,
                      time_granularity: str = 'daily',
                      cache_path: Optional[str] = None) -> TrendResult:
        """
        Comprehensive trend analysis of fitness discussions
        
        Args:
            df: DataFrame with fitness discussions and timestamps
            categories: Optional category assignments for each text
            time_granularity: Time grouping ('daily', 'weekly', 'monthly')
            cache_path: Optional path for caching results
            
        Returns:
            TrendResult with comprehensive temporal analysis
        """
        logger.info(f"📈 Starting trend analysis on {len(df)} discussions...")
        
        # Check for cached results
        if cache_path and Path(cache_path).exists():
            logger.info("📂 Loading cached trend analysis...")
            return self._load_cached_results(cache_path)
        
        # Prepare temporal data
        df_temporal = self._prepare_temporal_data(df, time_granularity)
        
        # Core trend analyses
        temporal_patterns = self._analyze_temporal_patterns(df_temporal)
        seasonal_patterns = self._analyze_seasonal_patterns(df_temporal)
        category_trends = self._analyze_category_trends(df_temporal, categories) if categories else {}
        anomalies = self._detect_anomalies(df_temporal)
        forecasts = self._generate_forecasts(df_temporal)
        insights = self._generate_insights(temporal_patterns, seasonal_patterns, category_trends, anomalies)
        
        # Create result object
        self.trend_result = TrendResult(
            temporal_patterns=temporal_patterns,
            category_trends=category_trends,
            seasonal_patterns=seasonal_patterns,
            anomalies=anomalies,
            forecasts=forecasts,
            insights=insights,
            metadata={
                'total_discussions': len(df),
                'time_range': {
                    'start': df_temporal.index.min().isoformat(),
                    'end': df_temporal.index.max().isoformat()
                },
                'granularity': time_granularity,
                'analysis_date': datetime.now().isoformat()
            }
        )
        
        self.is_fitted = True
        
        # Cache results
        if cache_path:
            self._save_cached_results(cache_path)
        
        logger.info("🎉 Trend analysis complete!")
        return self.trend_result
    
    def _prepare_temporal_data(self, df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Prepare and aggregate data for temporal analysis"""
        logger.info(f"🔄 Preparing temporal data with {granularity} granularity...")
        
        # Convert timestamps
        if self.time_column in df.columns:
            # Handle Unix timestamps
            if df[self.time_column].dtype in ['int64', 'float64']:
                df['datetime'] = pd.to_datetime(df[self.time_column], unit='s')
            else:
                df['datetime'] = pd.to_datetime(df[self.time_column])
        else:
            # Create dummy timestamps if no time column
            logger.warning(f"Time column '{self.time_column}' not found. Creating dummy timestamps.")
            date_range = pd.date_range(
                start='2024-01-01', 
                periods=len(df), 
                freq='H'
            )
            df['datetime'] = date_range
        
        # Aggregate by time granularity
        if granularity == 'daily':
            freq = 'D'
        elif granularity == 'weekly':
            freq = 'W'
        elif granularity == 'monthly':
            freq = 'M'
        else:
            freq = 'D'
        
        # Group and aggregate
        df_grouped = df.set_index('datetime').groupby(pd.Grouper(freq=freq)).agg({
            self.text_column: ['count', lambda x: x.str.len().sum() if hasattr(x, 'str') else len(x)]
        }).reset_index()
        
        # Flatten column names
        df_grouped.columns = ['datetime', 'discussion_count', 'total_chars']
        df_grouped = df_grouped.set_index('datetime')
        
        # Fill missing dates with zeros
        date_range = pd.date_range(
            start=df_grouped.index.min(),
            end=df_grouped.index.max(),
            freq=freq
        )
        df_grouped = df_grouped.reindex(date_range, fill_value=0)
        
        logger.info(f"✅ Temporal data prepared: {len(df_grouped)} time periods")
        return df_grouped
    
    def _analyze_temporal_patterns(self, df_temporal: pd.DataFrame) -> Dict:
        """Analyze overall temporal patterns"""
        logger.info("🔄 Analyzing temporal patterns...")
        
        discussion_counts = df_temporal['discussion_count'].values
        
        patterns = {
            'total_discussions': int(discussion_counts.sum()),
            'avg_daily_discussions': float(discussion_counts.mean()),
            'peak_discussions': int(discussion_counts.max()),
            'peak_date': df_temporal[df_temporal['discussion_count'] == discussion_counts.max()].index[0].isoformat(),
            'trend_direction': self._calculate_trend_direction(discussion_counts),
            'volatility': float(discussion_counts.std()),
            'growth_rate': self._calculate_growth_rate(discussion_counts),
            'activity_distribution': self._analyze_activity_distribution(discussion_counts)
        }
        
        return patterns
    
    def _analyze_seasonal_patterns(self, df_temporal: pd.DataFrame) -> Dict:
        """Analyze seasonal and cyclic patterns"""
        logger.info("🔄 Analyzing seasonal patterns...")
        
        # Add time-based features
        df_temporal['dayofweek'] = df_temporal.index.dayofweek
        df_temporal['month'] = df_temporal.index.month
        df_temporal['hour'] = df_temporal.index.hour
        
        seasonal = {
            'day_of_week_pattern': self._analyze_day_of_week_pattern(df_temporal),
            'monthly_pattern': self._analyze_monthly_pattern(df_temporal),
            'seasonality_strength': self._calculate_seasonality_strength(df_temporal['discussion_count']),
            'cyclic_patterns': self._detect_cyclic_patterns(df_temporal['discussion_count'])
        }
        
        return seasonal
    
    def _analyze_category_trends(self, df_temporal: pd.DataFrame, categories: Dict) -> Dict:
        """Analyze trends by category"""
        logger.info("🔄 Analyzing category-specific trends...")
        
        category_trends = {}
        
        # For this implementation, we'll simulate category trends
        # In practice, this would use actual category assignments
        fitness_categories = [
            'strength', 'nutrition', 'equipment', 'motivation', 
            'injury', 'beginner', 'advanced', 'cardio'
        ]
        
        for category in fitness_categories:
            # Simulate category-specific data
            # In practice, filter discussions by category
            category_pattern = self._simulate_category_trend(df_temporal, category)
            
            category_trends[category] = {
                'trend_direction': category_pattern['trend'],
                'growth_rate': category_pattern['growth'],
                'peak_periods': category_pattern['peaks'],
                'correlation_with_overall': category_pattern['correlation']
            }
        
        return category_trends
    
    def _detect_anomalies(self, df_temporal: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in discussion patterns"""
        logger.info("🔄 Detecting anomalies...")
        
        discussion_counts = df_temporal['discussion_count'].values
        
        # Z-score based anomaly detection
        z_scores = np.abs(stats.zscore(discussion_counts))
        anomaly_threshold = 2.5
        
        anomalies = []
        for i, (date, count) in enumerate(zip(df_temporal.index, discussion_counts)):
            if z_scores[i] > anomaly_threshold:
                anomalies.append({
                    'date': date.isoformat(),
                    'discussion_count': int(count),
                    'z_score': float(z_scores[i]),
                    'type': 'spike' if count > discussion_counts.mean() else 'drop',
                    'severity': 'high' if z_scores[i] > 3 else 'medium'
                })
        
        return anomalies
    
    def _generate_forecasts(self, df_temporal: pd.DataFrame) -> Dict:
        """Generate simple trend forecasts"""
        logger.info("🔄 Generating forecasts...")
        
        discussion_counts = df_temporal['discussion_count'].values
        
        # Simple linear trend forecast
        if len(discussion_counts) >= self.min_data_points:
            x = np.arange(len(discussion_counts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, discussion_counts)
            
            # Forecast next 7 periods
            forecast_periods = 7
            forecast_x = np.arange(len(discussion_counts), len(discussion_counts) + forecast_periods)
            forecast_values = slope * forecast_x + intercept
            
            forecasts = {
                'linear_trend': {
                    'slope': float(slope),
                    'r_squared': float(r_value ** 2),
                    'forecast_values': forecast_values.tolist(),
                    'confidence': 'high' if r_value ** 2 > 0.7 else 'medium' if r_value ** 2 > 0.3 else 'low'
                },
                'moving_average': {
                    'window': 7,
                    'latest_ma': float(np.mean(discussion_counts[-7:])),
                    'forecast': float(np.mean(discussion_counts[-7:]))
                }
            }
        else:
            forecasts = {
                'linear_trend': {'error': 'Insufficient data for trend analysis'},
                'moving_average': {'error': 'Insufficient data for moving average'}
            }
        
        return forecasts
    
    def _generate_insights(self, temporal_patterns: Dict, seasonal_patterns: Dict, 
                          category_trends: Dict, anomalies: List) -> List[str]:
        """Generate human-readable insights from analysis"""
        insights = []
        
        # Temporal insights
        if temporal_patterns['trend_direction'] == 'increasing':
            insights.append(f"📈 Discussion activity is growing at {temporal_patterns['growth_rate']:.1%} rate")
        elif temporal_patterns['trend_direction'] == 'decreasing':
            insights.append(f"📉 Discussion activity is declining at {abs(temporal_patterns['growth_rate']):.1%} rate")
        
        # Peak activity insight
        peak_date = datetime.fromisoformat(temporal_patterns['peak_date'])
        insights.append(f"🔥 Peak activity occurred on {peak_date.strftime('%B %d, %Y')} with {temporal_patterns['peak_discussions']} discussions")
        
        # Seasonal insights
        if 'day_of_week_pattern' in seasonal_patterns:
            peak_day = max(seasonal_patterns['day_of_week_pattern'], key=seasonal_patterns['day_of_week_pattern'].get)
            insights.append(f"📅 Highest activity occurs on {peak_day}s")
        
        # Anomaly insights
        if anomalies:
            high_anomalies = [a for a in anomalies if a['severity'] == 'high']
            if high_anomalies:
                insights.append(f"⚠️ Detected {len(high_anomalies)} significant activity spikes/drops")
        
        # Category insights
        if category_trends:
            growing_categories = [cat for cat, data in category_trends.items() 
                                if data['trend_direction'] == 'increasing']
            if growing_categories:
                insights.append(f"🚀 Growing categories: {', '.join(growing_categories[:3])}")
        
        return insights
    
    def _calculate_trend_direction(self, values: np.ndarray) -> str:
        """Calculate overall trend direction"""
        if len(values) < 2:
            return 'stable'
        
        x = np.arange(len(values))
        slope, _, r_value, p_value, _ = stats.linregress(x, values)
        
        if p_value < 0.05:  # Statistically significant
            if slope > 0:
                return 'increasing'
            else:
                return 'decreasing'
        return 'stable'
    
    def _calculate_growth_rate(self, values: np.ndarray) -> float:
        """Calculate compound growth rate"""
        if len(values) < 2:
            return 0.0
        
        first_val = max(values[0], 1)  # Avoid division by zero
        last_val = values[-1]
        periods = len(values) - 1
        
        growth_rate = (last_val / first_val) ** (1/periods) - 1
        return float(growth_rate)
    
    def _analyze_activity_distribution(self, values: np.ndarray) -> Dict:
        """Analyze distribution of activity levels"""
        return {
            'low_activity_days': int(np.sum(values < np.percentile(values, 25))),
            'medium_activity_days': int(np.sum((values >= np.percentile(values, 25)) & 
                                             (values <= np.percentile(values, 75)))),
            'high_activity_days': int(np.sum(values > np.percentile(values, 75))),
            'consistency_score': float(1 - (np.std(values) / (np.mean(values) + 1e-8)))
        }
    
    def _analyze_day_of_week_pattern(self, df_temporal: pd.DataFrame) -> Dict:
        """Analyze day-of-week patterns"""
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_avg = df_temporal.groupby('dayofweek')['discussion_count'].mean()
        
        return {day_names[i]: float(day_avg.get(i, 0)) for i in range(7)}
    
    def _analyze_monthly_pattern(self, df_temporal: pd.DataFrame) -> Dict:
        """Analyze monthly patterns"""
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_avg = df_temporal.groupby('month')['discussion_count'].mean()
        
        return {month_names[i-1]: float(month_avg.get(i, 0)) for i in range(1, 13)}
    
    def _calculate_seasonality_strength(self, values: pd.Series) -> float:
        """Calculate strength of seasonal patterns"""
        try:
            # Simple seasonality calculation using autocorrelation
            if len(values) < 14:
                return 0.0
            
            # Check for weekly seasonality (7-day cycle)
            autocorr_7 = values.autocorr(lag=7)
            return float(abs(autocorr_7)) if not np.isnan(autocorr_7) else 0.0
        except:
            return 0.0
    
    def _detect_cyclic_patterns(self, values: pd.Series) -> Dict:
        """Detect cyclic patterns in the data"""
        patterns = {
            'weekly_cycle': self._calculate_seasonality_strength(values),
            'detected_cycles': []
        }
        
        # Simple peak detection for cycles
        if len(values) >= 14:
            peaks, _ = find_peaks(values, height=values.mean())
            if len(peaks) > 1:
                cycle_lengths = np.diff(peaks)
                avg_cycle = float(np.mean(cycle_lengths)) if len(cycle_lengths) > 0 else 0
                patterns['detected_cycles'].append({
                    'type': 'discussion_peaks',
                    'average_length': avg_cycle,
                    'count': len(peaks)
                })
        
        return patterns
    
    def _simulate_category_trend(self, df_temporal: pd.DataFrame, category: str) -> Dict:
        """Simulate category-specific trends (placeholder for real implementation)"""
        # This would be replaced with actual category filtering in practice
        base_pattern = df_temporal['discussion_count'].values
        
        # Simulate category-specific variations
        np.random.seed(hash(category) % 1000)
        noise = np.random.normal(0, 0.1, len(base_pattern))
        category_pattern = base_pattern * (1 + noise)
        
        trend_direction = self._calculate_trend_direction(category_pattern)
        growth_rate = self._calculate_growth_rate(category_pattern)
        correlation = float(np.corrcoef(base_pattern, category_pattern)[0, 1])
        
        peaks, _ = find_peaks(category_pattern, height=np.mean(category_pattern))
        
        return {
            'trend': trend_direction,
            'growth': growth_rate,
            'peaks': len(peaks),
            'correlation': correlation
        }
    
    def visualize_trends(self, 
                        save_path: Optional[str] = None,
                        show_plot: bool = True) -> go.Figure:
        """Create comprehensive trend visualization"""
        if not self.is_fitted:
            raise ValueError("Must run analyze_trends() before visualization")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Discussion Volume Over Time',
                'Day of Week Pattern', 
                'Monthly Pattern',
                'Growth Trends by Category'
            ),
            specs=[[{"secondary_y": True}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Time series plot (placeholder - would use actual temporal data)
        sample_dates = pd.date_range('2024-01-01', periods=30, freq='D')
        sample_values = np.random.poisson(20, 30)
        
        fig.add_trace(
            go.Scatter(x=sample_dates, y=sample_values, name="Discussions", line=dict(color='blue')),
            row=1, col=1
        )
        
        # Day of week pattern
        if 'day_of_week_pattern' in self.trend_result.seasonal_patterns:
            dow_data = self.trend_result.seasonal_patterns['day_of_week_pattern']
            fig.add_trace(
                go.Bar(x=list(dow_data.keys()), y=list(dow_data.values()), name="Day Pattern"),
                row=1, col=2
            )
        
        # Monthly pattern
        if 'monthly_pattern' in self.trend_result.seasonal_patterns:
            month_data = self.trend_result.seasonal_patterns['monthly_pattern']
            fig.add_trace(
                go.Bar(x=list(month_data.keys()), y=list(month_data.values()), name="Month Pattern"),
                row=2, col=1
            )
        
        # Category trends
        if self.trend_result.category_trends:
            categories = list(self.trend_result.category_trends.keys())[:6]  # Top 6
            growth_rates = [self.trend_result.category_trends[cat]['growth_rate'] * 100 
                          for cat in categories]
            
            fig.add_trace(
                go.Scatter(x=categories, y=growth_rates, mode='markers+lines', 
                          name="Category Growth", marker=dict(size=10)),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Fitness Discussion Trend Analysis",
            height=800,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"📊 Trend visualization saved to {save_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def get_trend_statistics(self) -> Dict:
        """Get comprehensive trend analysis statistics"""
        if not self.is_fitted:
            raise ValueError("Must run analyze_trends() before getting statistics")
        
        result = self.trend_result
        
        stats = {
            'temporal_summary': result.temporal_patterns,
            'seasonal_summary': result.seasonal_patterns,
            'anomaly_count': len(result.anomalies),
            'high_severity_anomalies': len([a for a in result.anomalies if a['severity'] == 'high']),
            'category_trends_count': len(result.category_trends),
            'insights_count': len(result.insights),
            'key_insights': result.insights[:5],  # Top 5 insights
            'forecast_confidence': result.forecasts.get('linear_trend', {}).get('confidence', 'unknown'),
            'metadata': result.metadata
        }
        
        return stats
    
    def save_results(self, save_path: str):
        """Save trend analysis results to JSON"""
        if not self.is_fitted:
            raise ValueError("Must run analyze_trends() before saving")
        
        # Convert results to JSON-serializable format
        results_dict = {
            'temporal_patterns': self.trend_result.temporal_patterns,
            'seasonal_patterns': self.trend_result.seasonal_patterns,
            'category_trends': self.trend_result.category_trends,
            'anomalies': self.trend_result.anomalies,
            'forecasts': self.trend_result.forecasts,
            'insights': self.trend_result.insights,
            'statistics': self.get_trend_statistics(),
            'metadata': self.trend_result.metadata
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"💾 Trend analysis results saved to {save_path}")
    
    def _save_cached_results(self, cache_path: str):
        """Save analysis state for fast loading"""
        cache_data = {
            'trend_result': self.trend_result,
            'is_fitted': self.is_fitted,
            'config': {
                'time_column': self.time_column,
                'text_column': self.text_column,
                'min_data_points': self.min_data_points
            }
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"💾 Cached trend analysis to {cache_path}")
    
    def _load_cached_results(self, cache_path: str) -> TrendResult:
        """Load cached trend analysis results"""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.trend_result = cache_data['trend_result']
        self.is_fitted = cache_data['is_fitted']
        
        logger.info("📂 Loaded cached trend analysis")
        return self.trend_result

# Example usage and testing
if __name__ == "__main__":
    print("📈 Trend Analysis System Initialized")
    print("Ready for temporal pattern analysis!")
