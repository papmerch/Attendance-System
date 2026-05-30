"""
attendance_report.py — CLI tool for generating attendance reports.

Usage:
    python attendance_report.py --today
    python attendance_report.py --date 2026-05-30
    python attendance_report.py --month 2026-05
    python attendance_report.py --all
    python attendance_report.py --help
"""

import sys
from attendance_manager import AttendanceManager


def main():
    mgr = AttendanceManager()

    if "--today" in sys.argv:
        records = mgr.get_today_attendance()
        mgr.print_report(records, title=f"Attendance for {mgr._today_str()}")
    elif "--date" in sys.argv:
        idx = sys.argv.index("--date")
        if idx + 1 < len(sys.argv):
            date_str = sys.argv[idx + 1]
            records = mgr.get_attendance_by_date(date_str)
            mgr.print_report(records, title=f"Attendance for {date_str}")
    elif "--month" in sys.argv:
        idx = sys.argv.index("--month")
        if idx + 1 < len(sys.argv):
            ym = sys.argv[idx + 1]
            records = mgr.get_monthly_report(ym)
            mgr.print_report(records, title=f"Attendance for {ym}")
    elif "--all" in sys.argv:
        records = mgr.get_all_attendance()
        mgr.print_report(records, title="Full Attendance History")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
