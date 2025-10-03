"""
Root Cause Analysis Driver

Reads from dataset/input.jsonl, calls the appropriate converted analysis script,
and returns structured JSON results.
"""

import json
import os
import sys
from datetime import datetime

# Import the analysis functions (with error handling)
try:
    from STS_Root_Cause_Analysis_Error import analyze_error_root_cause
    from STS_Root_Cause_Analysis_Latency import analyze_latency_root_cause
    ANALYSIS_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Analysis functions not available: {e}")
    print("💡 Required dependencies may not be installed")
    ANALYSIS_FUNCTIONS_AVAILABLE = False
    analyze_error_root_cause = None
    analyze_latency_root_cause = None


def determine_analysis_type(alarm_rules):
    """
    Determine the analysis type based on alarm rules

    Args:
        alarm_rules: List of alarm rule strings

    Returns:
        str: Analysis type ("error" or "latency")
    """
    if not alarm_rules:
        return "error"  # Default to error analysis

    # Convert to lowercase for case-insensitive matching
    alarm_rules_lower = [rule.lower() for rule in alarm_rules]

    # Check for error-related rules
    error_indicators = ["error", "failure", "exception", "status"]
    for rule in alarm_rules_lower:
        if any(indicator in rule for indicator in error_indicators):
            return "error"

    # Check for latency-related rules
    latency_indicators = ["rt", "latency", "response", "duration", "time"]
    for rule in alarm_rules_lower:
        if any(indicator in rule for indicator in latency_indicators):
            return "latency"

    # Default to error analysis if no clear indicators
    return "error"




def run_error_analysis(start_time, end_time, candidate_root_causes):
    """
    Run error analysis using the imported function

    Args:
        start_time: Start time for analysis
        end_time: End time for analysis
        candidate_root_causes: List of candidate root causes to limit analysis to

    Returns:
        list: Root cause candidates array or empty array
    """
    if not ANALYSIS_FUNCTIONS_AVAILABLE or analyze_error_root_cause is None:
        print("❌ Analysis function not available - required dependencies not installed")
        return []

    try:
        # Call the error analysis function directly
        result = analyze_error_root_cause(start_time, end_time, candidate_root_causes)
        return result

    except Exception as e:
        print(f"❌ Error analysis failed: {e}")
        return []


def run_latency_analysis(anomaly_start_time, anomaly_end_time, candidate_root_causes):
    """
    Run latency analysis using the imported function

    Args:
        anomaly_start_time: Start time of anomaly period
        anomaly_end_time: End time of anomaly period
        candidate_root_causes: List of candidate root causes to limit analysis to

    Returns:
        list: Root cause candidates array or empty array
    """
    if not ANALYSIS_FUNCTIONS_AVAILABLE or analyze_latency_root_cause is None:
        print("❌ Analysis function not available - required dependencies not installed")
        return []

    try:
        # Call the latency analysis function directly
        result = analyze_latency_root_cause(anomaly_start_time, anomaly_end_time, candidate_root_causes)
        return result

    except Exception as e:
        print(f"❌ Latency analysis failed: {e}")
        return []


def read_input_data(input_file_path):
    """
    Read and parse input data from JSONL file

    Args:
        input_file_path: Path to the input JSONL file

    Returns:
        list: List of parsed JSON objects
    """
    data = []

    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Failed to parse line: {line[:100]}... Error: {e}")
                        continue

        print(f"✅ Successfully read {len(data)} records from {input_file_path}")
        return data

    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file_path}")
        return []
    except Exception as e:
        print(f"❌ Failed to read input file: {e}")
        return []


def process_single_problem(problem_data):
    """
    Process a single problem from the input data

    Args:
        problem_data: Dictionary containing problem information

    Returns:
        dict: Processing result with root cause candidate
    """
    try:
        problem_id = problem_data.get("problem_id", "unknown")
        time_range = problem_data.get("time_range", "")
        candidate_root_causes = problem_data.get("candidate_root_causes", [])
        alarm_rules = problem_data.get("alarm_rules", [])

        print(f"\n🔍 Processing problem {problem_id}")
        print(f"   Time range: {time_range}")
        print(f"   Alarm rules: {alarm_rules}")
        print(f"   Candidates: {len(candidate_root_causes)} possible root causes")

        # Determine analysis type
        analysis_type = determine_analysis_type(alarm_rules)
        print(f"   Analysis type: {analysis_type}")

        # Parse time range
        if " ~ " not in time_range:
            return {
                "problem_id": problem_id,
                "root_cause": "unknown",
                "confidence": "low",
                "evidence": False,
                "error": f"Invalid time range format: {time_range}"
            }

        start_time, end_time = time_range.split(' ~ ')
        start_time = start_time.strip()
        end_time = end_time.strip()

        # Validate time format
        try:
            datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return {
                "problem_id": problem_id,
                "root_cause": "unknown",
                "confidence": "low",
                "evidence": False,
                "error": f"Invalid time format in: {time_range}"
            }

        # Run analysis with candidate validation
        if analysis_type == "error":
            root_cause = run_error_analysis(start_time, end_time, candidate_root_causes)
        elif analysis_type == "latency":
            root_cause = run_latency_analysis(start_time, end_time, candidate_root_causes)
        else:
            root_cause = run_error_analysis(start_time, end_time, candidate_root_causes)

        print(f"   ✅ Root cause candidate: {root_cause}")

        # Handle array returns from analysis functions
        if isinstance(root_cause, list):
            if root_cause:  # Non-empty array
                print(f"   📝 Root cause: {root_cause[0]}")
                return root_cause[0]
            else:  # Empty array
                print(f"   📝 No root cause found (empty result)")
                return "unknown"
        elif root_cause and root_cause != "unknown":
            print(f"   📝 Root cause: {root_cause}")
            return root_cause

        # Return unknown for any invalid/empty results
        return "unknown"

    except Exception as e:
        print(f"❌ Failed to process problem {problem_data.get('problem_id', 'unknown')}: {e}")
        return "unknown"


def process_all_problems(input_data=None, input_file_path="../dataset/input.jsonl", output_file_path=None):
    """
    Process all problems from the input file

    Args:
        input_data: Pre-loaded input data (optional)
        input_file_path: Path to the input JSONL file
        output_file_path: Optional path to save results

    Returns:
        list: List of processing results
    """
    print("🚀 Starting Root Cause Analysis Driver")
    print("="*60)

    # Read input data if not provided
    if input_data is None:
        input_data = read_input_data(input_file_path)

    if not input_data:
        print("❌ No input data to process")
        return []

    # Process each problem
    results = []
    successful_analyses = 0
    failed_analyses = 0

    for i, problem_data in enumerate(input_data, 1):
        print(f"\n📊 Processing problem {i}/{len(input_data)}")
        print("-"*40)

        result = process_single_problem(problem_data)
        results.append(result)

        if result != "unknown":
            successful_analyses += 1
        else:
            failed_analyses += 1

    # Summary
    print(f"\n🏆 Analysis Summary")
    print("="*60)
    print(f"Total problems processed: {len(results)}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {failed_analyses}")

    # Save results in JSONL format (default to output.jsonl)
    output_file = output_file_path if output_file_path else "../dataset/output.jsonl"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                problem_id = input_data[i].get("problem_id", f"problem_{i+1}")
                # Wrap root cause in array format - only valid results
                root_causes = [result] if result and result != "unknown" else []
                output_line = {
                    "problem_id": problem_id,
                    "root_causes": root_causes
                }
                f.write(json.dumps(output_line, ensure_ascii=False) + '\n')
        print(f"✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")

    return results


def get_root_cause_for_problem(problem_id, input_file_path="../dataset/input.jsonl"):
    """
    Get root cause for a specific problem ID

    Args:
        problem_id: The problem ID to analyze
        input_file_path: Path to the input JSONL file

    Returns:
        dict: Root cause analysis result for the specific problem
    """
    input_data = read_input_data(input_file_path)

    for problem_data in input_data:
        if problem_data.get("problem_id") == problem_id:
            print(f"🔍 Found problem {problem_id}, processing...")
            return process_single_problem(problem_data)

    print(f"❌ Problem {problem_id} not found in input data")
    return "unknown"


def main():
    """Main function to run the driver"""
    if len(sys.argv) < 2:
        print("Usage: python root_cause_driver.py <command> [args...]")
        print("Commands:")
        print("  all [output_file]          - Process all problems (outputs to ../dataset/output.jsonl)")
        print("  problem <problem_id>       - Process specific problem")
        print("  test <time_range>          - Test analysis with time range")
        return

    command = sys.argv[1]

    if command == "all":
        # Default to output.jsonl if no output file specified
        output_file = sys.argv[2] if len(sys.argv) > 2 else "../dataset/output.jsonl"
        input_data = read_input_data("../dataset/input.jsonl")
        results = process_all_problems(input_data=input_data, output_file_path=output_file)

        # Print summary
        print(f"\n📋 Final Results Summary:")
        print("-"*40)

        for i, result in enumerate(results):
            problem_id = input_data[i].get("problem_id", f"problem_{i+1}") if i < len(input_data) else f"problem_{i+1}"
            status = "✅" if result != "unknown" else "❌"
            print(f"{status} {problem_id}: {result}")

    elif command == "problem":
        if len(sys.argv) < 3:
            print("Usage: python root_cause_driver.py problem <problem_id>")
            return

        problem_id = sys.argv[2]
        result = get_root_cause_for_problem(problem_id)
        print(result)

    elif command == "test":
        if len(sys.argv) < 3:
            print("Usage: python root_cause_driver.py test '<start_time> ~ <end_time>' [analysis_type]")
            return

        time_range = sys.argv[2]
        analysis_type = sys.argv[3] if len(sys.argv) > 3 else "error"

        print(f"🧪 Testing {analysis_type} analysis with time range: {time_range}")

        if " ~ " not in time_range:
            print("❌ Invalid time range format. Use: 'start_time ~ end_time'")
            return

        start_time, end_time = time_range.split(' ~ ')
        start_time = start_time.strip()
        end_time = end_time.strip()

        if analysis_type == "error":
            result = run_error_analysis(start_time, end_time)
        elif analysis_type == "latency":
            result = run_latency_analysis(start_time, end_time)
        else:
            print(f"❌ Unknown analysis type: {analysis_type}")
            return

        print(result)

    else:
        print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    main()
