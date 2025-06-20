import pyodbc
import pymssql
import pandas as pd
import json # For handling potential issues with serializing specific column types to JSON for logging/reporting
import sys # For checking Python version

# Pre-requisite Check for Python Version ---
if sys.version_info < (3, 12):
    print("Warning: Python 3.12+ is recommended for this script. You are using Python", sys.version)

#
# IMPORTANT: Replace these placeholders with your actual database connection details.

# Sybase Connection Details
# Ensure you have the correct ODBC driver installed for Sybase.
# Common drivers include 'Adaptive Server Enterprise', 'Sybase ASE ODBC Driver',
# or sometimes '{ODBC Driver 17 for SQL Server}' if Sybase is accessed via a SQL Server-compatible driver.
# You may need to experiment to find the correct DRIVER string for your setup.
SYBASE_SERVER = 'your_sybase_server_address'
SYBASE_DATABASE = 'your_sybase_database_name'
SYBASE_USERNAME = 'your_sybase_username'
SYBASE_PASSWORD = 'your_sybase_password'
SYBASE_DRIVER = '{ODBC Driver 17 for SQL Server}' # <<< ADJUST THIS IF NEEDED >>>

# SQL Server Connection Details
SQLSERVER_SERVER = 'your_sqlserver_server_address'
SQLSERVER_DATABASE = 'your_sqlserver_database_name'
SQLSERVER_USERNAME = 'your_sqlserver_username'
SQLSERVER_PASSWORD = 'your_sqlserver_password'

# Output Excel Report Name
REPORT_FILE = 'database_comparison_report.xlsx'

# Helper Functions for Database Operations ---

def get_sybase_connection():
    """Establishes and returns a connection to the Sybase database."""
    try:
        conn_str = (
            f"DRIVER={SYBASE_DRIVER};"
            f"SERVER={SYBASE_SERVER};"
            f"DATABASE={SYBASE_DATABASE};"
            f"UID={SYBASE_USERNAME};"
            f"PWD={SYBASE_PASSWORD};"
        )
        # autocommit=True is often useful for simple read operations to avoid explicit commits.
        conn = pyodbc.connect(conn_str, autocommit=True)
        print("Connected to Sybase successfully.")
        return conn
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"Error connecting to Sybase: SQLSTATE={sqlstate} - Message: {ex.args[1]}")
        return None
    except Exception as ex:
        print(f"An unexpected error occurred while connecting to Sybase: {ex}")
        return None

def get_sqlserver_connection():
    """Establishes and returns a connection to the SQL Server database."""
    try:
        # as_dict=False means rows are returned as tuples, which is suitable for pandas.read_sql.
        conn = pymssql.connect(
            server=SQLSERVER_SERVER,
            user=SQLSERVER_USERNAME,
            password=SQLSERVER_PASSWORD,
            database=SQLSERVER_DATABASE,
            as_dict=False
        )
        print("Connected to SQL Server successfully.")
        return conn
    except Exception as ex:
        print(f"Error connecting to SQL Server: {ex}")
        return None

def get_tables(conn, db_type):
    """
    Retrieves a list of table names from the given database connection.
    Args:
        conn: Database connection object (pyodbc or pymssql).
        db_type: 'sybase' or 'sqlserver' to determine the appropriate system query.
    Returns:
        A list of table names (strings). Returns an empty list if an error occurs.
    """
    try:
        cursor = conn.cursor()
        if db_type == 'sybase':
            # Sybase: Query sys.sysobjects for user tables (type 'U').
            query = "SELECT name FROM sys.sysobjects WHERE type = 'U' ORDER BY name"
        elif db_type == 'sqlserver':
            # SQL Server: Query INFORMATION_SCHEMA.TABLES for base tables.
            query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME"
        else:
            raise ValueError("Unsupported database type. Use 'sybase' or 'sqlserver'.")

        cursor.execute(query)
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables
    except Exception as ex:
        print(f"Error getting tables from {db_type}: {ex}")
        return []

def get_table_schema(conn, table_name, db_type):
    """
    Retrieves the schema (column names and data types) for a given table.
    Args:
        conn: Database connection object.
        table_name: Name of the table.
        db_type: 'sybase' or 'sqlserver'.
    Returns:
        A list of dictionaries, each representing a column (e.g., {'COLUMN_NAME': 'id', 'DATA_TYPE': 'int'}).
        Returns an empty list if an error occurs.
    """
    try:
        cursor = conn.cursor()
        if db_type == 'sybase':
            # Sybase: Joins syscolumns (for column names) and systypes (for data types)
            # using sysobjects to filter by table name.
            query = f"""
            SELECT
                c.name AS COLUMN_NAME,
                t.name AS DATA_TYPE
            FROM
                syscolumns c
            JOIN
                sysobjects o ON c.id = o.id
            JOIN
                systypes t ON c.type = t.type
            WHERE
                o.name = '{table_name}'
            ORDER BY c.colid;
            """
        elif db_type == 'sqlserver':
            # SQL Server: Queries INFORMATION_SCHEMA.COLUMNS.
            query = f"""
            SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{table_name}' ORDER BY ORDINAL_POSITION;
            """
        else:
            raise ValueError("Unsupported database type.")

        cursor.execute(query)
        # Fetch results and format into a list of dictionaries.
        schema = [{'COLUMN_NAME': row[0], 'DATA_TYPE': row[1]} for row in cursor.fetchall()]
        cursor.close()
        return schema
    except Exception as ex:
        print(f"Error getting schema for {table_name} from {db_type}: {ex}")
        return []

def get_primary_keys(conn, table_name, db_type):
    """
    Retrieves primary key column names for a given table.
    Args:
        conn: Database connection object.
        table_name: Name of the table.
        db_type: 'sybase' or 'sqlserver'.
    Returns:
        A list of primary key column names (strings). Returns an empty list if no PKs found or an error occurs.
    """
    pk_columns = []
    try:
        cursor = conn.cursor()
        if db_type == 'sybase':
            # Sybase: Attempts to use the sp_pkeys stored procedure.
            # This is the most direct way for Sybase. Output structure can vary slightly by Sybase version.
            # Assumes COLUMN_NAME is at index 3 in sp_pkeys result.
            try:
                # Ensure table_name is properly quoted/escaped if it contains special characters
                cursor.execute(f"EXEC sp_pkeys '{table_name}'")
                results = cursor.fetchall()
                if results:
                    # sp_pkeys typically returns: TABLE_CAT, TABLE_SCHEM, TABLE_NAME, COLUMN_NAME, KEY_SEQ, PK_NAME
                    # We are interested in COLUMN_NAME (index 3).
                    pk_columns = [row[3] for row in results if len(row) > 3]
                else:
                    print(f"No primary keys reported by sp_pkeys for {table_name} in Sybase.")
            except pyodbc.ProgrammingError as pe:
                # This error can occur if sp_pkeys is not available or has different permissions.
                print(f"Warning: Could not execute sp_pkeys for table {table_name} in Sybase. Error: {pe}. "
                      "This table will be compared using all common columns if no PKs are found.")
                pk_columns = [] # Fallback to empty list
            except Exception as e:
                print(f"Warning: Unexpected error calling sp_pkeys for {table_name} in Sybase: {e}. "
                      "This table will be compared using all common columns if no PKs are found.")
                pk_columns = []

        elif db_type == 'sqlserver':
            # SQL Server: Queries INFORMATION_SCHEMA.TABLE_CONSTRAINTS and KEY_COLUMN_USAGE.
            query = f"""
            SELECT kcu.COLUMN_NAME
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
            ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
            WHERE tc.TABLE_NAME = '{table_name}' AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            ORDER BY kcu.ORDINAL_POSITION;
            """
            cursor.execute(query)
            pk_columns = [row[0] for row in cursor.fetchall()]
        else:
            raise ValueError("Unsupported database type.")

        cursor.close()
        return pk_columns
    except Exception as ex:
        print(f"Error getting primary keys for {table_name} from {db_type}: {ex}")
        return []

def get_row_count(conn, table_name):
    """
    Retrieves the row count for a given table.
    Args:
        conn: Database connection object.
        table_name: Name of the table.
    Returns:
        The row count (int), or -1 if an error occurs (e.g., table not found, permission issue).
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        cursor.close()
        return count
    except Exception as ex:
        print(f"Error getting row count for {table_name}: {ex}")
        return -1

def clean_data_for_comparison(df):
    """
    Cleans a DataFrame for comparison: trims strings, handles NULLs/empty strings.
    The requirement is "Handle NULLs and empty strings as equal."
    This function replaces pandas' NaN (for numbers) and None (for objects) with an empty string,
    then trims all string columns.
    Args:
        df: pandas DataFrame.
    Returns:
        Cleaned pandas DataFrame.
    """
    # Create a copy to avoid modifying the original DataFrame in place
    df_cleaned = df.copy()

    # Fill all NaN/None values with empty string across the entire DataFrame
    df_cleaned = df_cleaned.fillna('')

    # Iterate through columns identified as object (strings) or explicitly string type
    for col in df_cleaned.select_dtypes(include=['object', 'string']).columns:
        # Ensure column is treated as string before stripping
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
    return df_cleaned

def compare_tables(syb_conn, sql_conn, table_name):
    """
    Compares schema, row counts, and data for a given table between Sybase and SQL Server.
    Args:
        syb_conn: Sybase database connection object.
        sql_conn: SQL Server database connection object.
        table_name: Name of the table to compare.
    Returns:
        A dictionary containing detailed comparison results for the table.
    """
    results = {
        'table_name': table_name,
        'schema_match': False,
        'row_count_syb': -1,
        'row_count_sql': -1,
        'rows_missing_in_target': 0,
        'rows_added_in_target': 0,
        'value_mismatches_count': 0,
        'overall_status': 'UNKNOWN', # Initial status
        'detailed_differences': []    # Stores row-level differences for the report
    }

    print(f"\n--- Comparing table: {table_name} ---")

    # --- 1. Compare Schema ---
    syb_schema = get_table_schema(syb_conn, table_name, 'sybase')
    sql_schema = get_table_schema(sql_conn, table_name, 'sqlserver')

    if not syb_schema or not sql_schema:
        print(f"Skipping schema comparison for {table_name} due to missing schema information (e.g., table not found or permission issue).")
        results['overall_status'] = 'SCHEMA_ERROR'
        results['detailed_differences'].append({
            'Action': 'SCHEMA_ERROR',
            'Column': 'N/A',
            'Source_Value': 'Could not retrieve Sybase schema.',
            'Target_Value': 'Could not retrieve SQL Server schema.',
            'Row_Identifier': 'N/A'
        })
        return results

    # Convert schema to DataFrames for easier comparison
    syb_schema_df = pd.DataFrame(syb_schema).astype(str)
    sql_schema_df = pd.DataFrame(sql_schema).astype(str)

    # Sort schemas by column name to ensure consistent order for comparison
    syb_schema_df = syb_schema_df.sort_values(by='COLUMN_NAME').reset_index(drop=True)
    sql_schema_df = sql_schema_df.sort_values(by='COLUMN_NAME').reset_index(drop=True)

    if syb_schema_df.equals(sql_schema_df):
        results['schema_match'] = True
        print(f"Schema for {table_name}: MATCH")
    else:
        results['schema_match'] = False
        print(f"Schema for {table_name}: MISMATCH")
        # Log detailed schema differences
        merged_schema = pd.merge(syb_schema_df, sql_schema_df, on='COLUMN_NAME', how='outer', suffixes=('_SYB', '_SQL'))

        # Identify columns with differing data types
        type_diffs = merged_schema[merged_schema['DATA_TYPE_SYB'] != merged_schema['DATA_TYPE_SQL']]
        # Filter out rows where one of the data types is NaN (indicating column missing)
        type_diffs = type_diffs.dropna(subset=['DATA_TYPE_SYB', 'DATA_TYPE_SQL'])
        if not type_diffs.empty:
            for idx, row in type_diffs.iterrows():
                results['detailed_differences'].append({
                    'Action': 'SCHEMA MISMATCH - DATA TYPE',
                    'Column': row['COLUMN_NAME'],
                    'Source_Value': row['DATA_TYPE_SYB'],
                    'Target_Value': row['DATA_TYPE_SQL'],
                    'Row_Identifier': 'N/A'
                })

        # Identify columns missing in Sybase (present in SQL Server only)
        missing_in_syb = merged_schema[merged_schema['DATA_TYPE_SYB'].isnull()]
        if not missing_in_syb.empty:
            for idx, row in missing_in_syb.iterrows():
                results['detailed_differences'].append({
                    'Action': 'SCHEMA MISMATCH - COLUMN MISSING IN SYBASE',
                    'Column': row['COLUMN_NAME'],
                    'Source_Value': 'N/A',
                    'Target_Value': row['DATA_TYPE_SQL'],
                    'Row_Identifier': 'N/A'
                })

        # Identify columns missing in SQL Server (present in Sybase only)
        missing_in_sql = merged_schema[merged_schema['DATA_TYPE_SQL'].isnull()]
        if not missing_in_sql.empty:
            for idx, row in missing_in_sql.iterrows():
                results['detailed_differences'].append({
                    'Action': 'SCHEMA MISMATCH - COLUMN MISSING IN SQLSERVER',
                    'Column': row['COLUMN_NAME'],
                    'Source_Value': row['DATA_TYPE_SYB'],
                    'Target_Value': 'N/A',
                    'Row_Identifier': 'N/A'
                })

    # --- 2. Compare Row Counts ---
    results['row_count_syb'] = get_row_count(syb_conn, table_name)
    results['row_count_sql'] = get_row_count(sql_conn, table_name)
    print(f"Row count for {table_name}: Sybase={results['row_count_syb']}, SQL Server={results['row_count_sql']}")

    # If row counts are -1, it indicates an error in fetching.
    if results['row_count_syb'] == -1 or results['row_count_sql'] == -1:
        print(f"Skipping data comparison for {table_name} due to row count fetch errors.")
        results['overall_status'] = 'ROW_COUNT_ERROR'
        return results

    # --- 3. Compare Data ---
    # Only proceed with data comparison if schemas match (or if a partial comparison is acceptable,
    # but for migration verification, schema match is crucial).
    if not results['schema_match']:
        print(f"Skipping data comparison for {table_name} due to schema mismatch.")
        results['overall_status'] = 'SCHEMA_MISMATCH'
        return results

    syb_pk_columns = get_primary_keys(syb_conn, table_name, 'sybase')
    sql_pk_columns = get_primary_keys(sql_conn, table_name, 'sqlserver')

    # Determine the key columns for merging DataFrames
    common_pk_columns = []
    if syb_pk_columns and sql_pk_columns:
        # Find intersection of PKs. If not all PKs are common, warn and fallback.
        common_pk_columns = list(set(syb_pk_columns) & set(sql_pk_columns))
        if not common_pk_columns:
            print(f"Warning: Primary keys reported by Sybase ({syb_pk_columns}) and SQL Server ({sql_pk_columns}) "
                  f"for {table_name} do not overlap. Will compare all common columns.")
    else:
        print(f"Warning: Could not determine consistent primary keys for {table_name} (Sybase PKs: {syb_pk_columns}, SQL Server PKs: {sql_pk_columns}). "
              "Will compare all common columns.")

    try:
        syb_df = pd.read_sql(f"SELECT * FROM {table_name}", syb_conn)
        sql_df = pd.read_sql(f"SELECT * FROM {table_name}", sql_conn)
    except Exception as ex:
        print(f"Error fetching data for {table_name} into DataFrames: {ex}")
        results['overall_status'] = 'DATA_FETCH_ERROR'
        results['detailed_differences'].append({
            'Action': 'DATA FETCH ERROR',
            'Column': 'N/A',
            'Source_Value': f"Error: {ex}",
            'Target_Value': f"Error: {ex}",
            'Row_Identifier': 'N/A'
        })
        return results

    # Clean dataframes for consistent comparison
    syb_df_cleaned = clean_data_for_comparison(syb_df.copy())
    sql_df_cleaned = clean_data_for_comparison(sql_df.copy())

    # Get common columns AFTER cleaning (as cleaning might affect dtypes for select_dtypes)
    # This also acts as a safeguard if schemas were considered "matching" but column casing differs.
    common_columns_data = list(set(syb_df_cleaned.columns) & set(sql_df_cleaned.columns))
    
    if not common_columns_data:
        print(f"Warning: No common data columns found between Sybase and SQL Server for table {table_name}. Cannot perform row-by-row comparison.")
        results['overall_status'] = 'NO_COMMON_DATA_COLUMNS'
        results['detailed_differences'].append({
            'Action': 'NO_COMMON_DATA_COLUMNS',
            'Column': 'N/A',
            'Source_Value': 'N/A',
            'Target_Value': 'N/A',
            'Row_Identifier': 'N/A'
        })
        return results

    syb_df_cleaned = syb_df_cleaned[common_columns_data]
    sql_df_cleaned = sql_df_cleaned[common_columns_data]

    # Determine final key columns for merging. If common_pk_columns exist and are in common_columns_data, use them.
    # Otherwise, use ALL common_columns_data as the merge key.
    key_columns = [col for col in common_pk_columns if col in common_columns_data]
    if not key_columns:
        key_columns = common_columns_data # Fallback to all common columns if no valid PKs

    # Ensure key columns actually exist in both dataframes (after potential column trimming/selection)
    if not all(col in syb_df_cleaned.columns for col in key_columns):
        print(f"Error: Some identified key columns are not present in Sybase DataFrame after cleaning for {table_name}. "
              f"Missing: {set(key_columns) - set(syb_df_cleaned.columns)}")
        results['overall_status'] = 'KEY_COLUMN_MISSING'
        return results
    if not all(col in sql_df_cleaned.columns for col in key_columns):
        print(f"Error: Some identified key columns are not present in SQL Server DataFrame after cleaning for {table_name}. "
              f"Missing: {set(key_columns) - set(sql_df_cleaned.columns)}")
        results['overall_status'] = 'KEY_COLUMN_MISSING'
        return results
    
    # Outer merge to find all rows (present in source, target, or both)
    # The 'indicator=True' adds a '_merge' column indicating where the row came from.
    merged_df = pd.merge(syb_df_cleaned, sql_df_cleaned, on=key_columns, how='outer', indicator=True, suffixes=('_syb', '_sql'))

    # --- Identify rows missing in target (present in source only) ---
    missing_in_target = merged_df[merged_df['_merge'] == 'left_only'].copy()
    results['rows_missing_in_target'] = len(missing_in_target)
    if not missing_in_target.empty:
        for idx, row in missing_in_target.iterrows():
            pk_values = {col: str(row[col]) for col in key_columns} # Capture PKs as identifier
            # Capture the full source row for detailed reporting
            source_row_data = {c.replace('_syb', ''): str(row[c]) for c in row.index if c.endswith('_syb')}
            results['detailed_differences'].append({
                'Action': 'MISSING IN TARGET',
                'Column': 'N/A', # Not a column-specific mismatch
                'Source_Value': source_row_data,
                'Target_Value': 'N/A', # Not present in target
                'Row_Identifier': pk_values
            })

    # --- Identify rows added in target (present in target only) ---
    added_in_target = merged_df[merged_df['_merge'] == 'right_only'].copy()
    results['rows_added_in_target'] = len(added_in_target)
    if not added_in_target.empty:
        for idx, row in added_in_target.iterrows():
            pk_values = {col: str(row[col]) for col in key_columns}
            # Capture the full target row for detailed reporting
            target_row_data = {c.replace('_sql', ''): str(row[c]) for c in row.index if c.endswith('_sql')}
            results['detailed_differences'].append({
                'Action': 'ADDED IN TARGET',
                'Column': 'N/A',
                'Source_Value': 'N/A', # Not present in source
                'Target_Value': target_row_data,
                'Row_Identifier': pk_values
            })

    # --- Identify value mismatches in common rows (present in both) ---
    common_rows = merged_df[merged_df['_merge'] == 'both'].copy()
    mismatches_found = 0
    if not common_rows.empty:
        # Columns to compare for value differences (exclude keys and the merge indicator)
        data_columns_for_comparison = [col for col in common_columns_data if col not in key_columns]

        for idx, row in common_rows.iterrows():
            row_pk_values = {col: str(row[col]) for col in key_columns}
            for col in data_columns_for_comparison:
                syb_col_name = f"{col}_syb"
                sql_col_name = f"{col}_sql"

                syb_val = row[syb_col_name] if syb_col_name in row.index else None
                sql_val = row[sql_col_name] if sql_col_name in row.index else None

                # Cleaned values should already be string and trimmed from clean_data_for_comparison
                # So direct comparison is appropriate here.
                if syb_val != sql_val:
                    mismatches_found += 1
                    # Use json.dumps to handle complex types (e.g., lists, dicts if they somehow exist)
                    # or special characters that might break Excel. Fallback to str().
                    try:
                        syb_val_str = json.dumps(syb_val)
                    except TypeError:
                        syb_val_str = str(syb_val)
                    try:
                        sql_val_str = json.dumps(sql_val)
                    except TypeError:
                        sql_val_str = str(sql_val)

                    results['detailed_differences'].append({
                        'Action': 'VALUE MISMATCH',
                        'Column': col,
                        'Source_Value': syb_val_str,
                        'Target_Value': sql_val_str,
                        'Row_Identifier': row_pk_values
                    })
    results['value_mismatches_count'] = mismatches_found

    print(f"Comparison Summary for {table_name}:")
    print(f"  Rows missing in target: {results['rows_missing_in_target']}")
    print(f"  Rows added in target: {results['rows_added_in_target']}")
    print(f"  Value mismatches: {results['value_mismatches_count']}")

    # Determine overall status
    if not results['schema_match'] or \
       results['row_count_syb'] != results['row_count_sql'] or \
       results['rows_missing_in_target'] > 0 or \
       results['rows_added_in_target'] > 0 or \
       results['value_mismatches_count'] > 0:
        results['overall_status'] = 'MISMATCHED'
    else:
        results['overall_status'] = 'MATCHED'

    return results

def generate_excel_report(comparison_results, report_file):
    """
    Generates an Excel report with a summary sheet and detailed sheets for mismatches.
    Args:
        comparison_results: List of dictionaries, each containing comparison results for a table.
        report_file: Path to the output Excel file (e.g., 'database_comparison_report.xlsx').
    """
    print(f"\nGenerating Excel report: {report_file}")
    # Using openpyxl engine for writing Excel files.
    writer = pd.ExcelWriter(report_file, engine='openpyxl')

    # --- Summary Sheet ---
    summary_data = []
    for res in comparison_results:
        summary_data.append({
            'Table Name': res['table_name'],
            'Schema Match': 'Yes' if res['schema_match'] else 'No',
            'Source Row Count': res['row_count_syb'],
            'Target Row Count': res['row_count_sql'],
            'Rows Missing in Target': res['rows_missing_in_target'],
            'Rows Added in Target': res['rows_added_in_target'],
            'Value Mismatches': res['value_mismatches_count'],
            'Overall Status': res['overall_status']
        })
    summary_df = pd.DataFrame(summary_data)
    # Write the summary DataFrame to a sheet named 'Summary'. Do not include the DataFrame index.
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

    # --- Detail Sheets for Mismatches ---
    for res in comparison_results:
        # Create a detail sheet only if there are recorded differences for the table.
        if res['detailed_differences']:
            detail_df = pd.DataFrame(res['detailed_differences'])
            # Create a valid Excel sheet name (max 31 characters, no invalid chars like /, \, ?, *, [, ])
            # We'll truncate and add a suffix if the table name is too long.
            sheet_name = f"{res['table_name']}_Details"
            if len(sheet_name) > 31:
                # Truncate and ensure it's still unique enough.
                sheet_name = sheet_name[:25] + "_Det" # e.g., "very_long_table_name_Det"

            # Convert 'Row_Identifier' dictionary to a JSON string for display in Excel.
            # This makes the PK values readable in the report.
            if 'Row_Identifier' in detail_df.columns:
                detail_df['Row_Identifier'] = detail_df['Row_Identifier'].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
                )
            
            # Convert Source_Value and Target_Value columns to string to handle various types gracefully in Excel.
            for col_name in ['Source_Value', 'Target_Value']:
                if col_name in detail_df.columns:
                    detail_df[col_name] = detail_df[col_name].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else str(x))

            detail_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save the Excel file. This commits all sheets to the file.
    writer.close()
    print(f"Excel report '{report_file}' generated successfully.")

def main():
    """
    Main function to orchestrate the database comparison process.
    Establishes connections, finds common tables, compares them, and generates the report.
    Ensures connections are closed in a finally block.
    """
    syb_conn = None
    sql_conn = None
    all_comparison_results = []

    try:
        # Attempt to connect to both databases.
        syb_conn = get_sybase_connection()
        sql_conn = get_sqlserver_connection()

        # If either connection fails, print an error and exit.
        if not syb_conn:
            print("Failed to establish Sybase connection. Please check your connection details and ODBC driver setup. Exiting.")
            return
        if not sql_conn:
            print("Failed to establish SQL Server connection. Please check your connection details. Exiting.")
            return

        # Get lists of tables from both databases.
        syb_tables = set(get_tables(syb_conn, 'sybase'))
        sql_tables = set(get_tables(sql_conn, 'sqlserver'))

        # Find tables common to both source and target.
        common_tables = sorted(list(syb_tables.intersection(sql_tables)))

        if not common_tables:
            print("No common tables found between Sybase and SQL Server based on provided database names. Please ensure tables exist and are accessible.")
            return

        print(f"\nFound {len(common_tables)} common tables for comparison: {', '.join(common_tables)}")

        # Iterate through common tables and perform comparison.
        for table_name in common_tables:
            results = compare_tables(syb_conn, sql_conn, table_name)
            all_comparison_results.append(results)

    except Exception as e:
        print(f"An unhandled error occurred during the comparison process: {e}")
    finally:
        # Ensure database connections are closed, even if errors occur.
        if syb_conn:
            try:
                syb_conn.close()
                print("Sybase connection closed.")
            except Exception as e:
                print(f"Error closing Sybase connection: {e}")
        if sql_conn:
            try:
                sql_conn.close()
                print("SQL Server connection closed.")
            except Exception as e:
                print(f"Error closing SQL Server connection: {e}")

    # Generate the Excel report if there are any comparison results.
    if all_comparison_results:
        generate_excel_report(all_comparison_results, REPORT_FILE)
    else:
        print("No comparison results were generated. Report will not be created.")

if __name__ == "__main__":
    main()

