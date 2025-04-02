import sys

from streamlit.web import cli as stcli

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        "src/screening_rag/preprocess/streamlit_reporter.py",
    ]
    sys.exit(stcli.main())
