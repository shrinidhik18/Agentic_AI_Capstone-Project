from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import math

class CalendarDateTool:
    """
    Precision tool for date arithmetic, day calculations, and schedule allocation.
    """
    @staticmethod
    def calculate_study_days(
        exam_date_str: str,
        start_date_str: Optional[str] = None,
        weekday_hours: float = 2.0,
        weekend_hours: float = 4.0
    ) -> Dict[str, Any]:
        """
        Computes total days remaining, breakdown of weekdays vs weekends, total available study hours,
        and generates a day-by-day timetable template.
        """
        try:
            # Parse exam date
            exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d").date()
        except ValueError:
            # Fallback format parsing (e.g. Oct 15, 2026 or 2026-10-15)
            try:
                exam_date = datetime.strptime(exam_date_str, "%b %d, %Y").date()
            except ValueError:
                exam_date = datetime.now().date() + timedelta(days=14)

        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            except ValueError:
                start_date = datetime.now().date()
        else:
            start_date = datetime.now().date()

        if exam_date <= start_date:
            days_remaining = 1
        else:
            days_remaining = (exam_date - start_date).days

        schedule_days = []
        total_hours = 0.0
        weekdays_count = 0
        weekends_count = 0

        curr_date = start_date
        while curr_date <= exam_date:
            is_weekend = curr_date.weekday() >= 5 # 5=Sat, 6=Sun
            daily_hours = weekend_hours if is_weekend else weekday_hours
            total_hours += daily_hours
            
            if is_weekend:
                weekends_count += 1
            else:
                weekdays_count += 1

            schedule_days.append({
                "date": curr_date.strftime("%Y-%m-%d"),
                "day_name": curr_date.strftime("%A"),
                "is_weekend": is_weekend,
                "available_hours": daily_hours,
                "allocated_subject": "",
                "time_slot": "Morning / Evening" if is_weekend else "Evening"
            })
            curr_date += timedelta(days=1)

        return {
            "start_date": start_date.strftime("%Y-%m-%d"),
            "exam_date": exam_date.strftime("%Y-%m-%d"),
            "days_remaining": days_remaining,
            "weekdays_count": weekdays_count,
            "weekends_count": weekends_count,
            "total_available_hours": total_hours,
            "daily_breakdown": schedule_days
        }

    @staticmethod
    def distribute_subjects(
        subjects: List[str],
        schedule_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Distributes subjects evenly across available study days.
        """
        days = schedule_info["daily_breakdown"]
        if not subjects or not days:
            return schedule_info

        num_subjects = len(subjects)
        num_days = len(days)

        allocated_schedule = []
        for idx, day_info in enumerate(days):
            # Assign subject round-robin or last day revision
            if idx == num_days - 1:
                assigned_sub = f"Final Revision ({', '.join(subjects)})"
            else:
                assigned_sub = subjects[idx % num_subjects]
            
            day_copy = dict(day_info)
            day_copy["allocated_subject"] = assigned_sub
            allocated_schedule.append(day_copy)

        schedule_info["allocated_schedule"] = allocated_schedule
        return schedule_info


class CalculatorTool:
    """
    Mathematical helper for academic calculations like attendance, GPA, study hour requirements.
    """
    @staticmethod
    def calculate_attendance_needed(
        current_conducted: int,
        current_attended: int,
        target_percent: float = 75.0
    ) -> Dict[str, Any]:
        """Calculates how many future classes student must attend to reach target percentage."""
        current_pct = (current_attended / current_conducted * 100) if current_conducted > 0 else 0.0
        
        if current_pct >= target_percent:
            bunkable = math.floor((current_attended - (target_percent / 100.0 * current_conducted)) / (target_percent / 100.0))
            return {
                "status": "Safe",
                "current_pct": round(current_pct, 2),
                "needed_classes": 0,
                "bunkable_classes": max(0, bunkable)
            }
        else:
            # target = (current_attended + x) / (current_conducted + x)
            # x * (1 - target) = target * current_conducted - current_attended
            needed = math.ceil((target_percent * current_conducted - 100 * current_attended) / (100 - target_percent))
            return {
                "status": "Shortage",
                "current_pct": round(current_pct, 2),
                "needed_classes": max(0, needed),
                "bunkable_classes": 0
            }
