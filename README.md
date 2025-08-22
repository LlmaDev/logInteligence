📊 Farm Pivot Log Analyzer

This project is a log parser and analyzer designed to process pivot/farm messages (from MESSAGE.txt), clean and structure the data, and generate Excel reports enriched with charts for visualization.

It is especially useful for monitoring farm irrigation systems, pivot statuses, and command logs — turning raw text into actionable insights.

🚀 Features

✅ Parse MESSAGE.txt files containing pivot/farm status updates

✅ Clean data automatically (removing hidden/illegal characters)

✅ Extract key fields such as:

Status (ON/OFF/ERROR etc.)

FarmName

PivotName

Command issued

Date and Time

✅ Save structured logs into an Excel file

✅ Generate charts for better visualization:

Bar charts (distribution per farm/command/status)

Optional line plots, heatmaps, or pie charts

✅ Handles noisy or malformed input gracefully

📂 Project Structure
📦 farm-pivot-log-analyzer
 ┣ 📜 parser.py        # Main script to parse MESSAGE.txt and save report
 ┣ 📜 MESSAGE.txt      # Input log file with raw pivot/farm data
 ┣ 📜 requirements.txt # Python dependencies
 ┣ 📜 README.md        # Project documentation
 ┗ 📂 output
     ┗ 📊 report.xlsx  # Generated Excel report with charts

⚙️ Requirements

Python 3.8+

Install dependencies:

pip install -r requirements.txt


requirements.txt should include:

pandas
openpyxl

▶️ Usage

Place your raw log file as MESSAGE.txt in the project folder.

Run the parser:

python parser.py


The script will:

Parse the log file

Clean illegal characters

Save the results in output/report.xlsx

📊 Example Output

The generated Excel file includes:

A cleaned table with structured fields (Farm, Pivot, Status, Command, Date, Time)

Charts showing:

Status distribution per farm

Commands usage frequency

### Pivot activity trends

(Example screenshot:)

| Farm  | Pivot    | Status | Command | Date       | Time  |
|-------|----------|--------|---------|------------|-------|
| Alpha | Pivot-01 | ON     | START   | 2025-08-21 | 15:30 |
| Beta  | Pivot-03 | OFF    | STOP    | 2025-08-21 | 16:10 |

🧹 Data Cleaning

To avoid Excel errors (IllegalCharacterError), the parser automatically strips non-printable ASCII characters and ensures clean text output.

📌 Roadmap

 Add more chart options (line, pie, heatmap)

 Build an interactive dashboard (Streamlit or React frontend)

 Export reports in CSV and PDF formats

 Automate daily parsing with cron/Windows Task Scheduler

🤝 Contributing

Pull requests and suggestions are welcome!
Feel free to open an issue if you find a bug or want a new feature.

📜 License

This project is released under the MIT License.
