import pandas as pd
import os

# Load your Excel file
df = pd.read_excel("autoReport.xlsx")

rows_to_save = []
save_next_two = False

def parseMessageLineAction(line):
    line = line.strip()
    if "->" not in line:
        return None

    try:
        dt_part, details = line.split(" -> ", 1)
        dt_be, hour_be = dt_part.split(" ", 1)

        # Clean up
        details = details.strip('"').rstrip("$")
        parts = details.split("-")

        if len(parts) < 5:
            return None  # malformed line

        Status, farm, Command, percent, owner = parts[:5]

        return {
            "DtBe": dt_be.strip(),
            "HourBe": hour_be.strip(),
            "Status": Status.strip(),          # renamed
            "FarmName": farm.strip(),
            "Command": Command.strip(),          # renamed
            "Percentimeter": percent.strip(),
            "Owner": owner.strip()
        }
    except ValueError:
        return None

i = 0
while i < len(df):
    Command = str(df.loc[i, "Command"]).strip()

    # Step 1: find a row where 2nd digit is '6'
    if len(Command) >= 2 and Command[1] == "6":
        rows_to_save.append(df.loc[i])  # save the row

        # Step 2: keep iterating forward to find last digit '2'
        j = i + 1
        while j < len(df):
            cmd_j = str(df.loc[j, "Command"]).strip()
            if len(cmd_j) >= 1 and cmd_j[-1] == "2":
                rows_to_save.append(df.loc[j])  # save this row too
                rows_to_save.append(pd.Series())  # empty separator row
                i = j  # jump forward so we continue AFTER this block
                break
            j += 1

    i += 1

# Convert list of Series back to DataFrame
filtered_df = pd.DataFrame(rows_to_save)

# Save result into new Excel
filtered_df.to_excel("filteredReport.xlsx", index=False)
print("Filtered report saved to filteredReport.xlsx")

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-==-=-=-=-=-=--=-=-==-=-=-=
