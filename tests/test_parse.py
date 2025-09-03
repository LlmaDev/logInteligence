from autoReport import parse_message_line
import pytest

valid_line = "2025-09-02 15:23:01 <--> OK-Farm1-16-45-90-95-123456$"

def test_parse_valid_line_returns_dict():
    result = parse_message_line(valid_line)
    assert isinstance(result, dict)
    assert result["Status"] == "OK"
    assert result["FarmName"] == "Farm1"
    assert result["Command"] == 16

def test_parse_returns_correct_datetime():
    result = parse_message_line(valid_line)
    assert result["DtBe"] == "2025-09-02"
    assert result["HourBe"] == "15:23:01"

def test_parse_percentimeter_numeric():
    result = parse_message_line(valid_line)
    assert result["Percentimeter"] == "45"

def test_parse_angles_correct():
    result = parse_message_line(valid_line)
    assert result["InitialAngle"] == "90"
    assert result["CurrentAngle"] == "95"

def test_parse_rtc_field():
    result = parse_message_line(valid_line)
    assert result["RTC"] == "123456"

def test_parse_command_type_int():
    result = parse_message_line(valid_line)
    assert isinstance(result["Command"], int)

def test_parse_line_with_extra_spaces():
    line = " 2025-09-02 15:23:01 -> OK-Farm1-16-45-90-95-123456$ "
    result = parse_message_line(line)
    assert result["Status"] == "OK"

def test_parse_returns_none_for_missing_arrow():
    line = "2025-09-02 15:23:01 OK-Farm1-16-45-90-95-123456$"
    assert parse_message_line(line) is None

def test_parse_returns_none_for_wrong_command():
    line = "2025-09-02 15:23:01 -> OK-Farm1-12-45-90-95-123456$"
    assert parse_message_line(line) is None

def test_parse_strips_hash_and_dollar():
    line = "2025-09-02 15:23:01 -> OK-Farm1-16-45-90-95-123456#$"
    result = parse_message_line(line)
    assert result["RTC"] == "123456"

def test_parse_missing_fields_returns_none():
    line = "2025-09-02 15:23:01 -> OK-Farm1-16-45-90$"
    assert parse_message_line(line) is None

def test_parse_non_digit_command_returns_none():
    line = "2025-09-02 15:23:01 -> OK-Farm1-1a-45-90-95-123456$"
    assert parse_message_line(line) is None

def test_parse_empty_line_returns_none():
    assert parse_message_line("") is None

def test_parse_only_arrow_returns_none():
    assert parse_message_line("->") is None

def test_parse_none_input_returns_none():
    assert parse_message_line(None) is None

def test_parse_invalid_split_returns_none():
    line = "2025-09-02 15:23:01 -> - - - - - -"
    assert parse_message_line(line) is None

def test_parse_command_without_6_second_digit():
    line = "2025-09-02 15:23:01 -> OK-Farm1-12-45-90-95-123456$"
    assert parse_message_line(line) is None

def test_parse_angle_non_numeric():
    line = "2025-09-02 15:23:01 -> OK-Farm1-16-45-A-95-123456$"
    result = parse_message_line(line)
    assert result["InitialAngle"] == "A"  # still parsed as string

def test_parse_percentimeter_non_numeric():
    line = "2025-09-02 15:23:01 -> OK-Farm1-16-XX-90-95-123456$"
    result = parse_message_line(line)
    assert result["Percentimeter"] == "XX"

def test_parse_with_incomplete_datetime():
    line = "2025-09-02 -> OK-Farm1-16-45-90-95-123456$"
    assert parse_message_line(line) is None
