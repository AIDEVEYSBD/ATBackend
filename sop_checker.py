import re
import os
from typing import Optional
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def parse_violation_report(response_text: str) -> pd.DataFrame:
    """
    Parse a Markdown table (violations or compliant) from model output into a DataFrame.

    Tolerant to:
    - leading/trailing pipes
    - alignment/separator rows (---, :---:)
    - soft-wrapped lines
    - variable spacing around '|'
    - fenced code blocks

    Returns empty DataFrame if no table detected.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return pd.DataFrame()

    # Remove any visible hidden-reasoning sections the model may include.
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1].strip()

    # Prefer table inside a fenced code block if present
    fenced = re.search(r"```(?:markdown|md|table)?\s*(.*?)```", response_text, flags=re.S | re.I)
    text = fenced.group(1).strip() if fenced else response_text

    raw_lines = [ln.rstrip() for ln in text.splitlines()]
    # Keep only table-ish lines
    table_lines = [ln for ln in raw_lines if ln.count("|") >= 2]

    if not table_lines:
        return pd.DataFrame()

    # Drop alignment/separator lines like |---|:---:|
    def is_sep(ln: str) -> bool:
        s = ln.replace("|", "").replace(" ", "")
        return s != "" and set(s) <= set("-:")

    table_lines = [ln for ln in table_lines if not is_sep(ln)]
    if not table_lines:
        return pd.DataFrame()

    # Join soft-wrapped rows: accumulate until a line ends with '|'
    rows_joined: list[str] = []
    buf = ""
    for ln in table_lines:
        stripped = ln.strip()
        if not buf:
            buf = stripped
        else:
            buf += " " + stripped
        if stripped.endswith("|"):
            rows_joined.append(buf)
            buf = ""
    if buf:
        rows_joined.append(buf)

    # Split into cells
    rows: list[list[str]] = []
    for ln in rows_joined:
        inner = ln.strip().strip("|")
        cells = [c.strip().strip('"') for c in inner.split("|")]
        if any(cells):
            rows.append(cells)

    if len(rows) < 2:
        return pd.DataFrame()

    headers = [(h or "").strip().strip('|') for h in rows[0]]
    ncols = len(headers)

    data = []
    for r in rows[1:]:
        r = (r + [""] * ncols)[:ncols]
        data.append([c.strip() for c in r])

    df = pd.DataFrame(data, columns=headers)

    # Drop fully empty columns created by stray delimiters
    empty_cols = [c for c in df.columns if c == "" or df[c].eq("").all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    # Normalize header whitespace
    df.columns = [c.strip().strip("|") for c in df.columns]
    return df


def compare_logs_and_sop(sop_text: str, log_text: str, model_name: str = "o4-mini") -> str:
    """
    Calls OpenAI to compare SOP and logs and return a Markdown table.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please add it to your .env file.")

    client = OpenAI(api_key=api_key)

    prompt = f"""
### Role
You are a strict compliance auditor specializing in SOP-log matching. Compare logs against SOPs and report only high-confidence violations (≥95% certainty).

### Instructions
1) Analyze each log entry separately.
   - If a log follows all SOP rules → Mark it **Compliant**.
   - If a log deviates → Mark it as a **Violation** and provide full details.

2) Severity definitions:
   - **Critical**: Direct safety risk, regulatory non-compliance, or financial impact.
   - **High**: Major process failure or procedural error affecting quality or efficiency.
   - **Medium**: Minor process deviation but no immediate risk.

3) Output structure (return **only** a single Markdown table). If violations exist, use:

| Severity  | SOP Section | SOP Requirement  | Log Entry (Row #) | Deviation Details | Confidence |
|-----------|-------------|------------------|-------------------|-------------------|------------|
| Critical/High/Medium | X.X | "Exact SOP text" | "Log text (Row #X)" | Explanation | 95%+ |

If all logs are compliant, use:

| Status    | Details                                  |
|-----------|------------------------------------------|
| Compliant | All operations match SOP requirements    |

### Logs
{log_text}

### SOP
{sop_text}

### Rules
- Do **NOT** modify the table schema.
- Only report deviations with ≥95% confidence.
- Quote exact text from logs and SOPs in the table.
- Reject uncertain findings.
- Do **not** report anomalies that are physical actions defined in the SOP but not present in the logs.

### Formatting
Return **only** the table inside a fenced code block as:

```markdown
<the table goes here>
```

No extra commentary or text outside the fenced block.
""".strip()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.choices[0].message.content or ""
        # Extract content after </think> if it exists
        if "</think>" in result:
            result = result.split("</think>")[-1].strip()
        return result.strip()
    except Exception as e:
        return f"Error processing analysis: {e}"


# Example usage function for testing
def test_sop_checker() -> None:
    """
    Example function to test the SOP checker.
    Replace with your actual SOP and log data.
    """
    sample_sop = """
1. All equipment must be checked before use
2. Safety protocols must be followed at all times
3. All actions must be logged with timestamps
""".strip()

    sample_log = """
10:00 - Started equipment check
10:05 - Equipment check completed
10:10 - Safety protocols reviewed
10:15 - Process started without logging timestamp
""".strip()

    result = compare_logs_and_sop(sample_sop, sample_log)
    print("Analysis Result:")
    print(result)

    # Parse the result into a DataFrame
    df = parse_violation_report(result)
    print("\nParsed DataFrame:")
    if df.empty:
        print("(empty)")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    # Uncomment the line below to test the function
    test_sop_checker()
    pass
