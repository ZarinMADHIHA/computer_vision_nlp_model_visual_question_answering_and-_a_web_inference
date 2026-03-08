"""
Database module for VQA experiments logging
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class VQADatabase:
    """Database manager for VQA experiments and results"""

    def __init__(self, db_path: str = "vqa_experiments.db"):
        """Initialize database connection and create tables"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def _create_tables(self):
        """Create database tables if they don't exist"""
        # VQAExperiments table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS VQAExperiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                hyperparameters TEXT NOT NULL,
                train_loss REAL,
                val_loss REAL,
                metrics TEXT,
                timestamp TEXT NOT NULL
            )
        """)

        # GeneratedAnswers table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS GeneratedAnswers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                image_url TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                ground_truth TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES VQAExperiments(id)
            )
        """)

        self.conn.commit()

    def insert_experiment(
        self,
        model_name: str,
        hyperparameters: Dict[str, Any],
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Insert a new experiment record

        Args:
            model_name: Name of the model
            hyperparameters: Dictionary of hyperparameters
            train_loss: Training loss
            val_loss: Validation loss
            metrics: Dictionary of evaluation metrics

        Returns:
            experiment_id: ID of inserted experiment
        """
        timestamp = datetime.now().isoformat()
        hyperparameters_json = json.dumps(hyperparameters)
        metrics_json = json.dumps(metrics) if metrics else None

        self.cursor.execute("""
            INSERT INTO VQAExperiments
            (model_name, hyperparameters, train_loss, val_loss, metrics, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (model_name, hyperparameters_json, train_loss, val_loss,
              metrics_json, timestamp))

        self.conn.commit()
        return self.cursor.lastrowid

    def update_experiment(
        self,
        experiment_id: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Update experiment with training results"""
        updates = []
        values = []

        if train_loss is not None:
            updates.append("train_loss = ?")
            values.append(train_loss)

        if val_loss is not None:
            updates.append("val_loss = ?")
            values.append(val_loss)

        if metrics is not None:
            updates.append("metrics = ?")
            values.append(json.dumps(metrics))

        if updates:
            query = f"UPDATE VQAExperiments SET {', '.join(updates)} WHERE id = ?"
            values.append(experiment_id)
            self.cursor.execute(query, values)
            self.conn.commit()

    def insert_answer(
        self,
        experiment_id: int,
        image_url: str,
        question: str,
        answer: str,
        ground_truth: Optional[str] = None
    ) -> int:
        """
        Insert a generated answer

        Args:
            experiment_id: ID of the experiment
            image_url: Path to image
            question: Question text
            answer: Generated answer
            ground_truth: Ground truth answer (optional)

        Returns:
            answer_id: ID of inserted answer
        """
        timestamp = datetime.now().isoformat()

        self.cursor.execute("""
            INSERT INTO GeneratedAnswers
            (experiment_id, image_url, question, answer, ground_truth, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (experiment_id, image_url, question, answer, ground_truth, timestamp))

        self.conn.commit()
        return self.cursor.lastrowid

    def get_experiment(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve experiment by ID"""
        self.cursor.execute("""
            SELECT * FROM VQAExperiments WHERE id = ?
        """, (experiment_id,))

        row = self.cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'model_name': row[1],
                'hyperparameters': json.loads(row[2]),
                'train_loss': row[3],
                'val_loss': row[4],
                'metrics': json.loads(row[5]) if row[5] else None,
                'timestamp': row[6]
            }
        return None

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Retrieve all experiments"""
        self.cursor.execute("SELECT * FROM VQAExperiments ORDER BY timestamp DESC")

        experiments = []
        for row in self.cursor.fetchall():
            experiments.append({
                'id': row[0],
                'model_name': row[1],
                'hyperparameters': json.loads(row[2]),
                'train_loss': row[3],
                'val_loss': row[4],
                'metrics': json.loads(row[5]) if row[5] else None,
                'timestamp': row[6]
            })
        return experiments

    def get_answers_by_experiment(self, experiment_id: int) -> List[Dict[str, Any]]:
        """Retrieve all answers for an experiment"""
        self.cursor.execute("""
            SELECT * FROM GeneratedAnswers WHERE experiment_id = ?
        """, (experiment_id,))

        answers = []
        for row in self.cursor.fetchall():
            answers.append({
                'id': row[0],
                'experiment_id': row[1],
                'image_url': row[2],
                'question': row[3],
                'answer': row[4],
                'ground_truth': row[5],
                'timestamp': row[6]
            })
        return answers

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
