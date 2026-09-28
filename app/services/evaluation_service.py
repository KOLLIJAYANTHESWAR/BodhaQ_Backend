"""
EvaluationService — deterministic quiz scoring.

Scoring is entirely algorithmic. The LLM is NOT involved in calculating
scores or identifying mistakes — this is application logic.
"""

from __future__ import annotations

import time
from app.models.responses import (
    MistakeDetail,
    QuizEvaluationResponse,
    WeakTopicItem,
    WeakTopicsResponse,
    QuizHistoryItem,
)
from app.services.quiz_service import quiz_service
from app.services.document_service import document_service


class EvaluationService:
    """Service responsible for deterministic quiz evaluation."""

    def __init__(self) -> None:
        self._evaluations: dict[str, QuizEvaluationResponse] = {}

    def evaluate(
        self,
        quiz_id: str,
        user_answers: dict[str, str],
    ) -> QuizEvaluationResponse:
        """
        Compare user answers against stored correct answers.

        Args:
            quiz_id:
                The quiz being submitted.

            user_answers:
                Mapping of question ID to selected answer letter.
                Example: {"1": "A", "2": "C"}.

        Returns:
            QuizEvaluationResponse containing score, percentage,
            and mistake details.

        Raises:
            ValueError:
                If the quiz does not exist.
        """

        stored = quiz_service.get_stored_quiz(quiz_id)

        if stored is None:
            raise ValueError(
                f"Quiz '{quiz_id}' not found. "
                "It may have expired because the server restarted."
            )

        private_answers: dict[str, dict] = stored["answers"]

        total = len(private_answers)
        score = 0
        mistakes: list[MistakeDetail] = []

        for question_id, metadata in private_answers.items():
            correct_answer = str(
                metadata["correct_answer"]
            ).strip().upper()

            user_answer = str(
                user_answers.get(question_id, "")
            ).strip().upper()

            if user_answer == correct_answer:
                score += 1
                continue

            mistakes.append(
                MistakeDetail(
                    question_id=int(question_id),
                    question=metadata["question"],
                    correct_answer=correct_answer,
                    user_answer=user_answer or "(no answer)",
                    explanation=metadata["explanation"],
                    topic=metadata["topic"],
                )
            )

        percentage = (
            round((score / total) * 100, 1)
            if total > 0
            else 0.0
        )

        evaluation = QuizEvaluationResponse(
            quiz_id=quiz_id,
            score=score,
            total=total,
            percentage=percentage,
            mistakes=mistakes,
            timestamp=time.time(),
        )

        # Store the actual evaluation so weak-topic analysis
        # can use the user's submitted answers.
        self._evaluations[quiz_id] = evaluation

        return evaluation

    def get_evaluation(
        self,
        quiz_id: str,
    ) -> QuizEvaluationResponse | None:
        """
        Return the most recent evaluation for a quiz.

        Returns None if the quiz has not been submitted yet.
        """

        return self._evaluations.get(quiz_id)

    def get_aggregated_gaps(
        self,
        limit: int = 10,
    ) -> WeakTopicsResponse:
        """
        Aggregate learning gaps from the most recent completed quizzes.
        """
        
        # Get the most recent `limit` evaluations
        evals = list(self._evaluations.values())
        evals.sort(key=lambda x: x.timestamp, reverse=True)
        recent_evals = evals[:limit]
        
        recent_quizzes: list[QuizHistoryItem] = []
        topic_stats: dict[str, dict] = {}
        
        for eval in recent_evals:
            stored = quiz_service.get_stored_quiz(eval.quiz_id)
            if not stored:
                continue
            
            # Determine title for history
            source_type = stored.get("source_type", "")
            source_id = stored.get("source_id", "")
            
            if source_type == "document":
                doc = document_service.get_document(source_id)
                title = doc.filename if doc else source_id
            else:
                title = source_id
                
            recent_quizzes.append(
                QuizHistoryItem(
                    quiz_id=eval.quiz_id,
                    title=title,
                    score=eval.score,
                    total=eval.total,
                    percentage=eval.percentage,
                    timestamp=eval.timestamp,
                )
            )
            
            private_answers = stored["answers"]
            mistake_ids = {m.question_id for m in eval.mistakes}
            
            for question_id, metadata in private_answers.items():
                topic = metadata.get("topic") or "General"
                
                if topic not in topic_stats:
                    topic_stats[topic] = {
                        "correct": 0,
                        "total": 0,
                        "source_type": source_type,
                        "source_id": source_id,
                        "quizzes": set(),
                    }
                    
                topic_stats[topic]["total"] += 1
                topic_stats[topic]["quizzes"].add(eval.quiz_id)
                
                if int(question_id) not in mistake_ids:
                    topic_stats[topic]["correct"] += 1

        weak_topics: list[WeakTopicItem] = []
        
        for topic, stats in topic_stats.items():
            total = stats["total"]
            correct = stats["correct"]
            accuracy = round((correct / total) * 100, 1) if total > 0 else 0.0
            
            if accuracy < 60.0:
                status = "Needs Practice"
            elif accuracy < 80.0:
                status = "Improving"
            else:
                status = "Learned"
                
            weak_topics.append(
                WeakTopicItem(
                    topic=topic,
                    accuracy=accuracy,
                    status=status,
                    source_type=stats["source_type"],
                    source_id=stats["source_id"],
                    quiz_count=len(stats["quizzes"]),
                )
            )
            
        weak_topics.sort(key=lambda item: (item.accuracy, -item.quiz_count))
        
        return WeakTopicsResponse(
            weak_topics=weak_topics,
            recent_quizzes=recent_quizzes,
        )


# Module-level singleton.
evaluation_service = EvaluationService()