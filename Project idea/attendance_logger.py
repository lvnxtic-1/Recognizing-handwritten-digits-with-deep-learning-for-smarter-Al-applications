import pandas as pd

def save_attendance(ids, filename="attendance.csv"):
    df = pd.DataFrame(ids, columns=["Student ID"])
    df.to_csv(filename, index=False)
