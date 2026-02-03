import traceback
import sys

class ChurnPipelineException(Exception):
    """
    Custom exception for the Churn Pipeline.
    Captures the error message, file name, line number, and code line.
    """

    def __init__(self, message: str):
        
        tb = traceback.extract_tb(sys.exc_info()[2])[-1] if sys.exc_info()[2] else None

        if tb:
            filename = tb.filename
            lineno = tb.lineno
            code_line = tb.line
            full_message = (
                f"{message}\n"
                f"File: {filename}, Line: {lineno}\n"
                f"Code: {code_line}"
            )
        else:
            # If no traceback, just show message
            full_message = message

        super().__init__(full_message)
