"""
data_loader.py
--------------
Handles reading uploaded CSV / Excel files safely, including multi-sheet
Excel workbooks, and returns friendly errors instead of raw stack traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


class DataLoadError(Exception):
    """Raised when a file cannot be parsed into a usable DataFrame."""


@dataclass
class LoadResult:
    file_name: str
    file_size_bytes: int
    sheets: Dict[str, pd.DataFrame] = field(default_factory=dict)
    is_excel: bool = False

    @property
    def sheet_names(self):
        return list(self.sheets.keys())

    def get(self, sheet_name: Optional[str] = None) -> pd.DataFrame:
        if sheet_name is None:
            sheet_name = self.sheet_names[0]
        return self.sheets[sheet_name]


def _friendly_error(msg: str) -> DataLoadError:
    return DataLoadError(f"⚠ {msg}")


def load_uploaded_file(uploaded_file) -> LoadResult:
    """
    Load a Streamlit UploadedFile object (CSV or Excel) into a LoadResult.
    Raises DataLoadError with a user-friendly message on failure.
    """
    if uploaded_file is None:
        raise _friendly_error("No file was uploaded.")

    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    file_size = len(file_bytes)

    if file_size == 0:
        raise _friendly_error("The uploaded file is empty. Please upload a file that contains data.")

    lower_name = file_name.lower()

    try:
        if lower_name.endswith(".csv") or lower_name.endswith(".tsv"):
            sep = "\t" if lower_name.endswith(".tsv") else None
            import io
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, engine="python",
                              on_bad_lines="skip")
            if df.shape[1] == 1 and sep is None:
                # Might be semicolon-delimited (common in EU locale exports)
                df_semicolon = pd.read_csv(io.BytesIO(file_bytes), sep=";", engine="python",
                                            on_bad_lines="skip")
                if df_semicolon.shape[1] > 1:
                    df = df_semicolon
            df = _drop_fully_unnamed_columns(df)
            if df.empty or df.shape[1] == 0:
                raise _friendly_error(
                    "We couldn't find any usable columns in this CSV file. "
                    "Please check the file format."
                )
            return LoadResult(file_name=file_name, file_size_bytes=file_size,
                               sheets={"Sheet1": df}, is_excel=False)

        elif lower_name.endswith(".xlsx") or lower_name.endswith(".xls") or lower_name.endswith(".xlsm"):
            import io
            engine = "openpyxl" if lower_name.endswith((".xlsx", ".xlsm")) else None
            xls = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
            sheets = {}
            for sheet in xls.sheet_names:
                sheet_df = xls.parse(sheet)
                sheet_df = _drop_fully_unnamed_columns(sheet_df)
                if not sheet_df.empty and sheet_df.shape[1] > 0:
                    sheets[sheet] = sheet_df
            if not sheets:
                raise _friendly_error(
                    "This Excel file doesn't contain any sheets with usable data."
                )
            return LoadResult(file_name=file_name, file_size_bytes=file_size,
                               sheets=sheets, is_excel=True)

        else:
            raise _friendly_error(
                "Unsupported file format. Please upload a .csv, .xlsx, or .xls file."
            )

    except DataLoadError:
        raise
    except pd.errors.EmptyDataError:
        raise _friendly_error("The uploaded file has no data to parse.")
    except pd.errors.ParserError:
        raise _friendly_error(
            "We couldn't parse this file — it may be corrupted or not a valid CSV."
        )
    except Exception as exc:  # noqa: BLE001 - convert any parsing failure to a friendly message
        raise _friendly_error(f"We couldn't read this file. Details: {exc}")


def _drop_fully_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in df.columns if not str(c).startswith("Unnamed:") or df[c].notna().any()]
    return df[keep_cols]


def dataframe_memory_mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)
