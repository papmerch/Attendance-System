"""
attendance_manager.py — Employee Attendance Tracking

Logs recognized faces to a CSV file with timestamps.
Prevents duplicate entries within a configurable cooldown window.
Supports daily / monthly / custom date reporting.

Attendance CSV columns:
    name, date, check_in_time, status
"""

import os
import csv
import datetime
import threading
from collections import defaultdict


class AttendanceManager:
    def __init__(self, csv_path=None, cooldown_hours=4):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        attendance_dir = os.path.join(base_dir, "attendance")
        os.makedirs(attendance_dir, exist_ok=True)
        self.csv_path = csv_path or os.path.join(attendance_dir, "Attendance.csv")
        self.cooldown_hours = cooldown_hours

        self._lock = threading.Lock()

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "date", "check_in_time", "status"])

    def _now(self):
        return datetime.datetime.now()

    def _today_str(self):
        return self._now().strftime("%Y-%m-%d")

    def mark_attendance(self, name):
        """
        Record attendance if the cooldown window has passed since the last
        entry for this person today.
        Returns (is_new, message).
        """
        now = self._now()
        today = self._today_str()
        window_start = now - datetime.timedelta(hours=self.cooldown_hours)

        with self._lock:
            rows = []
            last_time = None
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["name"] == name and row["date"] == today:
                        t = datetime.datetime.strptime(
                            f"{row['date']} {row['check_in_time']}", "%Y-%m-%d %H:%M:%S"
                        )
                        if last_time is None or t > last_time:
                            last_time = t
                    rows.append(row)

            if last_time is not None and last_time > window_start:
                return False, f"Already recorded ({last_time.strftime('%H:%M')})"

            check_in = now.strftime("%H:%M:%S")
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, today, check_in, "Present"])

        return True, f"Attendance recorded at {check_in}"

    def get_today_attendance(self):
        today = self._today_str()
        records = []
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["date"] == today:
                    records.append(row)
        return records

    def get_attendance_by_date(self, date_str):
        records = []
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["date"] == date_str:
                    records.append(row)
        return records

    def get_monthly_report(self, year_month):
        records = []
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["date"].startswith(year_month):
                    records.append(row)
        return records

    def get_all_attendance(self):
        records = []
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        return records

    def get_summary(self, records):
        summary = defaultdict(list)
        for r in records:
            summary[r["name"]].append(r["check_in_time"])
        return dict(summary)

    def print_report(self, records, title="Attendance Report"):
        if not records:
            print(f"\n  No records found for {title}.")
            return
        summary = self.get_summary(records)
        print(f"\n  {'=' * 50}")
        print(f"  {title}")
        print(f"  {'=' * 50}")
        print(f"  {'Name':<30} {'Check-ins'}")
        print(f"  {'-' * 50}")
        for name in sorted(summary):
            times = ", ".join(summary[name])
            print(f"  {name:<30} {times}")
        print(f"  {'-' * 50}")
        print(f"  Total employees present: {len(summary)}")
        print(f"  {'=' * 50}")
