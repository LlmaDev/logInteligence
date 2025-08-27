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
                # ✅ filter only if second digit of Command is "6"
        if len(Command) < 2 or Command[1] != "6":
            return None

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
    

##excel_file = "autoReport.xlsx"
all_rows = []
meses = ["blank","janeiro","fevereiro","março","abril","maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
pivotList = ["Pivo4","Pivo13","Pivo15"]

for k in range(3):
    # Loop through multiple folders
    for j in range(13):
            
        for i in range(1, 32):  # adjust range according to your folder count
                
            path = f"./resources/logs/{pivotList[k]}/{meses[j]}/{i}/MESSAGE.txt"
            if not os.path.exists(path):
                print("entered: ", path) 
                continue

            with open(path, "r") as file:
                for line in file:
                    row = parse_message_line(line)
                   ## print(row)
                    if row:
                        all_rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(all_rows)
    excel_file = f"autoReport{pivotList[k]}.xlsx"
    df.to_excel(excel_file, index=False)
    print(f"Saved updated report to {excel_file}")
'''
    # Append to existing Excel file if it exists
    try:
        old_df = pd.read_excel(excel_file)
        df = pd.concat([old_df, df], ignore_index=True)
    except FileNotFoundError:
        pass
'''
    # Save updated spreadsheet
    