# === Database Utilities for FarmSmart ===
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json

class FarmSmartDB:
    """Database manager for FarmSmart application"""
    
    def __init__(self, db_path='farmsmart.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Retraining history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retraining_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT NOT NULL,
                accuracy REAL,
                new_samples INTEGER,
                training_duration REAL
            )
        ''')
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                cpu_usage REAL,
                memory_usage REAL,
                active_connections INTEGER,
                model_uptime_hours REAL,
                total_predictions INTEGER
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                avg_confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, image_name, predicted_class, confidence):
        """Save a prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (image_name, predicted_class, confidence)
            VALUES (?, ?, ?)
        ''', (image_name, predicted_class, confidence))
        conn.commit()
        conn.close()
    
    def save_retraining_record(self, dataset_name, status, accuracy=None, 
                             new_samples=None, training_duration=None):
        """Save retraining record to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if status == 'completed':
            cursor.execute('''
                INSERT INTO retraining_history 
                (dataset_name, end_time, status, accuracy, new_samples, training_duration)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (dataset_name, datetime.now(), status, accuracy, new_samples, training_duration))
        else:
            cursor.execute('''
                INSERT INTO retraining_history (dataset_name, status)
                VALUES (?, ?)
            ''', (dataset_name, status))
        
        conn.commit()
        conn.close()
    
    def save_system_metrics(self, cpu_usage, memory_usage, active_connections, 
                          model_uptime, total_predictions):
        """Save system metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO system_metrics 
            (cpu_usage, memory_usage, active_connections, model_uptime_hours, total_predictions)
            VALUES (?, ?, ?, ?, ?)
        ''', (cpu_usage, memory_usage, active_connections, model_uptime, total_predictions))
        conn.commit()
        conn.close()
    
    def get_prediction_stats(self, days=7):
        """Get prediction statistics for visualizations"""
        conn = sqlite3.connect(self.db_path)
        
        # Class distribution
        class_dist = pd.read_sql_query('''
            SELECT predicted_class, COUNT(*) as count
            FROM predictions
            WHERE upload_time >= datetime('now', '-{} days')
            GROUP BY predicted_class
            ORDER BY count DESC
        '''.format(days), conn)
        
        # Daily predictions
        daily_stats = pd.read_sql_query('''
            SELECT DATE(upload_time) as date, COUNT(*) as count
            FROM predictions
            WHERE upload_time >= datetime('now', '-{} days')
            GROUP BY DATE(upload_time)
            ORDER BY date DESC
        '''.format(days), conn)
        
        # Confidence distribution
        confidence_stats = pd.read_sql_query('''
            SELECT confidence
            FROM predictions
            WHERE upload_time >= datetime('now', '-{} days')
            ORDER BY upload_time DESC
        '''.format(days), conn)
        
        # Recent predictions
        recent_predictions = pd.read_sql_query('''
            SELECT image_name, predicted_class, confidence, upload_time
            FROM predictions
            ORDER BY upload_time DESC
            LIMIT 10
        ''', conn)
        
        conn.close()
        
        return {
            'class_distribution': class_dist,
            'daily_stats': daily_stats,
            'confidence_stats': confidence_stats,
            'recent_predictions': recent_predictions
        }
    
    def get_retraining_history(self, limit=10):
        """Get retraining history"""
        conn = sqlite3.connect(self.db_path)
        history = pd.read_sql_query('''
            SELECT dataset_name, start_time, end_time, status, accuracy, new_samples
            FROM retraining_history
            ORDER BY start_time DESC
            LIMIT {}
        '''.format(limit), conn)
        conn.close()
        return history
    
    def get_system_metrics_history(self, hours=24):
        """Get system metrics history"""
        conn = sqlite3.connect(self.db_path)
        metrics = pd.read_sql_query('''
            SELECT timestamp, cpu_usage, memory_usage, active_connections, model_uptime_hours
            FROM system_metrics
            WHERE timestamp >= datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        '''.format(hours), conn)
        conn.close()
        return metrics
    
    def get_dashboard_stats(self):
        """Get comprehensive dashboard statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Total predictions
        total_pred = pd.read_sql_query('SELECT COUNT(*) as count FROM predictions', conn).iloc[0]['count']
        
        # Average confidence
        avg_conf = pd.read_sql_query('SELECT AVG(confidence) as avg_conf FROM predictions', conn).iloc[0]['avg_conf']
        
        # Recent activity (last 24 hours)
        recent_activity = pd.read_sql_query('''
            SELECT COUNT(*) as count
            FROM predictions
            WHERE upload_time >= datetime('now', '-24 hours')
        ''', conn).iloc[0]['count']
        
        # Most common prediction
        most_common = pd.read_sql_query('''
            SELECT predicted_class, COUNT(*) as count
            FROM predictions
            GROUP BY predicted_class
            ORDER BY count DESC
            LIMIT 1
        ''', conn)
        
        # Retraining success rate
        retraining_stats = pd.read_sql_query('''
            SELECT 
                COUNT(*) as total_retrains,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_retrains
            FROM retraining_history
        ''', conn)
        
        conn.close()
        
        return {
            'total_predictions': total_pred,
            'average_confidence': round(avg_conf, 2) if avg_conf else 0,
            'recent_activity': recent_activity,
            'most_common_prediction': most_common.iloc[0]['predicted_class'] if not most_common.empty else 'None',
            'retraining_success_rate': round(
                retraining_stats.iloc[0]['successful_retrains'] / max(retraining_stats.iloc[0]['total_retrains'], 1) * 100, 1
            ) if not retraining_stats.empty else 0
        }
    
    def export_data(self, format='json'):
        """Export database data for analysis"""
        stats = self.get_prediction_stats()
        dashboard_stats = self.get_dashboard_stats()
        
        export_data = {
            'dashboard_stats': dashboard_stats,
            'prediction_stats': {
                'class_distribution': stats['class_distribution'].to_dict('records'),
                'daily_stats': stats['daily_stats'].to_dict('records'),
                'recent_predictions': stats['recent_predictions'].to_dict('records')
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        if format == 'json':
            return json.dumps(export_data, indent=2, default=str)
        else:
            return export_data

# Global database instance
db = FarmSmartDB() 