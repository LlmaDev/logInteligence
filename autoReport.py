import pandas as pd
import os

def parse_message_line(line):
    line = line.strip()
    if "->" not in line:
        return None

    try:
        dt_part, details = line.split(" -> ", 1)
        dt_be, hour_be = dt_part.split(" ", 1)

        # Remove trailing '$' and split into exactly 7 fields
        details = details.rstrip("$")
        parts = details.split("-")
        if len(parts) != 7:
            return None  # skip malformed

        Status, farm, Command, percent, init_angle, curr_angle, rtc = parts

        return {
            "DtBe": dt_be.strip(),
            "HourBe": hour_be.strip(),
            "Status": Status.strip(),
            "FarmName": farm.strip(),
            "Command": Command.strip(),
            "Percentimeter": percent.strip(),
            "InitialAngle": init_angle.strip(),
            "CurrentAngle": curr_angle.strip(),
            "RTC": rtc.strip()
        }
    except ValueError:
        return None

excel_file = "autoReport.xlsx"
all_rows = []

# Loop through multiple folders
for i in range(1, 31):  # adjust range according to your folder count
    
    path = f"./resources/julho/{i}/MESSAGE.txt"
    if not os.path.exists(path):
        continue

    with open(path, "r") as file:
        for line in file:
            row = parse_message_line(line)
            if row:
                all_rows.append(row)

# Create DataFrame
df = pd.DataFrame(all_rows)

# Append to existing Excel file if it exists
try:
    old_df = pd.read_excel(excel_file)
    df = pd.concat([old_df, df], ignore_index=True)
except FileNotFoundError:
    pass

# Save updated spreadsheet
df.to_excel(excel_file, index=False)
print(f"Saved updated report to {excel_file}")

