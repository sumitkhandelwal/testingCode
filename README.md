import sys
import pyodbc
import pymssql
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

def get_sybase_connection(dsn, uid, pwd):
    """
    Establishes and returns a connection to Sybase database using pyodbc.
    Replace DSN, UID, PWD with your actual Sybase connection details.
    Ensure you have the correct Sybase ODBC driver installed and configured.
    """
    try:
        # Example DSN-based connection. If not using DSN, provide a direct connection string like:
        # conn_str = "DRIVER={Sybase Adaptive Server Enterprise};SERVER=your_sybase_server:port;DATABASE=your_sybase_db;UID=your_sybase_user;PWD=your_sybase_password"
        conn_str = f"DSN={dsn};UID={uid};PWD={pwd}"
        conn = pyodbc.connect(conn_str, autocommit=True) # autocommit is often suitable for read operations
        print("Connected to Sybase successfully.")
        return conn
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"ERROR: Sybase Connection Failed: {sqlstate} - {ex.args[1]}")
        sys.exit(1) # Exit if connection fails

def get_mssql_connection(server, user, password, database):
    """
    Establishes and returns a connection to MSSQL database using pymssql.
    Replace SERVER, USER, PASSWORD, DATABASE with your actual MSSQL connection details.
    """
    try:
        conn = pymssql.connect(server=server, user=user, password=password, database=database)
        print("Connected to MSSQL successfully.")
        return conn
    except Exception as ex:
        print(f"ERROR: MSSQL Connection Failed: {ex}")
        sys.exit(1) # Exit if connection fails

def get_table_schema(db_cursor, db_type, table_name):
    """
    Fetches schema information (column names and data types) for a given table.
    Returns a list of dictionaries with 'name' and 'type'.
    """
    schema = []
    try:
        if db_type == 'sybase':
            # For Sybase, using sp_columns stored procedure.
            # The output columns can vary slightly by Sybase version,
            # but COLUMN_NAME (index 3) and TYPE_NAME (index 5) are common.
            db_cursor.execute(f"EXEC sp_columns '{table_name}'")
            for row in db_cursor:
                # Adjust indices if your sp_columns output differs
                schema.append({'name': row[3], 'type': row[5]})
        elif db_type == 'mssql':
            # For MSSQL, using INFORMATION_SCHEMA.COLUMNS view.
            db_cursor.execute(
                f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_NAME = '{table_name}' ORDER BY ORDINAL_POSITION"
            )
            for row in db_cursor:
                schema.append({'name': row[0], 'type': row[1]})
        else:
            print(f"WARNING: Unsupported database type '{db_type}' for schema retrieval.")
            return None
        return schema
    except Exception as ex:
        print(f"ERROR: Failed to fetch schema for {db_type} table '{table_name}': {ex}")
        return None

def get_record_count(db_cursor, table_name):
    """
    Fetches the total record count for a given table.
    """
    try:
        db_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = db_cursor.fetchone()[0]
        return count
    except Exception as ex:
        print(f"ERROR: Failed to fetch record count for table '{table_name}': {ex}")
        return -1 # Return -1 to indicate an error

def get_table_data(db_cursor, table_name, db_type, schema):
    """
    Fetches all data from a given table and returns it as a sorted list of tuples.
    Data is processed to ensure consistent comparison (e.g., binary to hex, dates to ISO strings).
    Rows are sorted to ensure reliable row-by-row comparison.
    It now requires schema to correctly order by all columns.
    """
    try:
        order_by_clause = ""
        if schema:
            column_names = [col['name'] for col in schema]
            if column_names:
                # Quote column names for safety in case they contain spaces or special chars.
                # Double quotes are generally more portable across systems for this purpose.
                # Consider specific quoting rules for your Sybase version if issues arise.
                order_by_clause = "ORDER BY " + ", ".join([f'"{col}"' for col in column_names])
            else:
                print(f"WARNING: No columns found in schema for table '{table_name}', cannot use ORDER BY.")
        else:
            print(f"WARNING: Schema not provided or empty for table '{table_name}', cannot use ORDER BY.")


        query = f"SELECT * FROM {table_name} {order_by_clause}"
        db_cursor.execute(query)
        data = db_cursor.fetchall()

        processed_data = []
        for row in data:
            processed_row = []
            for item in row:
                if isinstance(item, (bytes, bytearray)):
                    # Convert binary data to hex string for consistent comparison
                    processed_row.append(item.hex())
                elif hasattr(item, 'isoformat'): # Handles datetime.date, datetime.datetime objects
                    processed_row.append(item.isoformat())
                else:
                    processed_row.append(item)
            processed_data.append(tuple(processed_row))

        # Sort the list of tuples for reliable comparison, crucial if database order isn't guaranteed
        # by the ORDER BY clause (e.g., if columns aren't unique enough) or if it's a large table.
        processed_data.sort()
        return processed_data
    except Exception as ex:
        print(f"ERROR: Failed to fetch data for {db_type} table '{table_name}': {ex}")
        return None

def write_to_excel(table_name, summary_results, schema_diffs, data_diffs, sybase_count, mssql_count):
    """
    Writes the comparison results to an Excel file.
    """
    filename = f"{table_name}.xlsx"
    workbook = openpyxl.Workbook()

    # --- Summary Sheet ---
    summary_sheet = workbook.active
    summary_sheet.title = "Summary"
    summary_sheet['A1'] = "Comparison Summary"
    summary_sheet['A1'].font = Font(bold=True, size=16)
    summary_sheet['A3'] = "Table Name:"
    summary_sheet['B3'] = table_name
    summary_sheet['A4'] = "Record Counts Match:"
    summary_sheet['B4'] = "YES" if summary_results['counts_match'] else "NO"
    summary_sheet['A5'] = "Schemas (Columns & Types) Match:"
    summary_sheet['B5'] = "YES" if summary_results['schemas_match'] else "NO"
    summary_sheet['A6'] = "Data in Columns Match:"
    summary_sheet['B6'] = "YES" if summary_results['data_matches'] else "NO"

    # Auto-width for Summary Sheet
    for col in summary_sheet.columns:
        max_length = 0
        column = col[0].column_letter # Get the column name
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        summary_sheet.column_dimensions[column].width = adjusted_width

    # --- Record Count Sheet ---
    rc_sheet = workbook.create_sheet("Record Count")
    rc_sheet['A1'] = "Record Count Comparison"
    rc_sheet['A1'].font = Font(bold=True, size=14)
    rc_sheet['A3'] = "Database"
    rc_sheet['B3'] = "Count"
    rc_sheet['A3'].font = Font(bold=True)
    rc_sheet['B3'].font = Font(bold=True)
    rc_sheet['A4'] = "Sybase"
    rc_sheet['B4'] = sybase_count
    rc_sheet['A5'] = "MSSQL"
    rc_sheet['B5'] = mssql_count

    # Auto-width for Record Count Sheet
    for col in rc_sheet.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        rc_sheet.column_dimensions[column].width = adjusted_width


    # --- Schema Differences Sheet ---
    schema_sheet = workbook.create_sheet("Schema Differences")
    schema_sheet['A1'] = "Schema Comparison Details"
    schema_sheet['A1'].font = Font(bold=True, size=14)
    schema_sheet.append(["Status", "Column Name", "Sybase Type", "MSSQL Type"])
    schema_sheet.cell(row=2, column=1).font = Font(bold=True)
    schema_sheet.cell(row=2, column=2).font = Font(bold=True)
    schema_sheet.cell(row=2, column=3).font = Font(bold=True)
    schema_sheet.cell(row=2, column=4).font = Font(bold=True)


    if not schema_diffs:
        schema_sheet.append(["", "No schema discrepancies found."])
    else:
        for diff in schema_diffs:
            schema_sheet.append([diff['status'], diff['column_name'], diff.get('sybase_type', ''), diff.get('mssql_type', '')])

    # Auto-width for Schema Sheet
    for col in schema_sheet.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        schema_sheet.column_dimensions[column].width = adjusted_width

    # --- Data Discrepancies Sheet ---
    data_sheet = workbook.create_sheet("Data Discrepancies")
    data_sheet['A1'] = "Data Mismatch Details"
    data_sheet['A1'].font = Font(bold=True, size=14)
    data_sheet.append(["Row Index (Sorted)", "Column Index", "Sybase Value", "MSSQL Value"])
    data_sheet.cell(row=2, column=1).font = Font(bold=True)
    data_sheet.cell(row=2, column=2).font = Font(bold=True)
    data_sheet.cell(row=2, column=3).font = Font(bold=True)
    data_sheet.cell(row=2, column=4).font = Font(bold=True)

    if not data_diffs:
        data_sheet.append(["", "", "No data discrepancies found."])
    else:
        for diff in data_diffs:
            data_sheet.append([diff['row_index'], diff['col_index'], diff['sybase_value'], diff['mssql_value']])

    # Auto-width for Data Sheet
    for col in data_sheet.columns:
        max_length = 0
        column = col[0].column_letter
        for cell in col:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        data_sheet.column_dimensions[column].width = adjusted_width


    try:
        workbook.save(filename)
        print(f"\nComparison report saved to '{filename}' successfully.")
    except Exception as ex:
        print(f"ERROR: Could not save Excel file '{filename}': {ex}")

def main():
    """
    Main function to orchestrate the database comparison process and generate Excel report.
    """
    if len(sys.argv) < 2:
        print("Usage: python compare_db_data.py <table_name>")
        sys.exit(1) # Exit if no table name is provided

    table_name = sys.argv[1]

    # --- Sybase Connection Details (PLACEHOLDERS - REPLACE WITH YOUR ACTUAL DETAILS) ---
    SYBASE_DSN = "YOUR_SYBASE_DSN"       # e.g., "Sybase_Prod_DB" - Must be configured on your OS
    SYBASE_UID = "YOUR_SYBASE_USERNAME"
    SYBASE_PWD = "YOUR_SYBASE_PASSWORD"

    # --- MSSQL Connection Details (PLACEHOLDERS - REPLACE WITH YOUR ACTUAL DETAILS) ---
    MSSQL_SERVER = "YOUR_MSSQL_SERVER_IP_OR_HOSTNAME" # e.g., "localhost", "192.168.1.10", or "your_server_name\SQLEXPRESS"
    MSSQL_DATABASE = "YOUR_MSSQL_DATABASE" # e.g., "SalesDB"
    MSSQL_USER = "YOUR_MSSQL_USERNAME"
    MSSQL_PASSWORD = "YOUR_MSSQL_PASSWORD"

    # --- Establish Database Connections ---
    sybase_conn = get_sybase_connection(SYBASE_DSN, SYBASE_UID, SYBASE_PWD)
    mssql_conn = get_mssql_connection(MSSQL_SERVER, MSSQL_USER, MSSQL_PASSWORD, MSSQL_DATABASE)

    sybase_cursor = sybase_conn.cursor()
    mssql_cursor = mssql_conn.cursor()

    print(f"\n--- Starting Comparison for Table: '{table_name}' ---")

    # Initialize results containers
    summary_results = {
        'counts_match': False,
        'schemas_match': False,
        'data_matches': False
    }
    schema_diffs = []
    data_diffs = []
    sybase_count = -1
    mssql_count = -1

    # --- 1. Compare Record Counts ---
    sybase_count = get_record_count(sybase_cursor, table_name)
    mssql_count = get_record_count(mssql_cursor, table_name)
    if sybase_count != -1 and mssql_count != -1:
        summary_results['counts_match'] = (sybase_count == mssql_count)
        print(f"Record Counts: Sybase={sybase_count}, MSSQL={mssql_count}. Match: {summary_results['counts_match']}")
    else:
        print("Could not retrieve one or both record counts due to prior errors.")

    # --- 2. Compare Schemas (Column Names and Data Types) ---
    sybase_schema = get_table_schema(sybase_cursor, 'sybase', table_name)
    mssql_schema = get_table_schema(mssql_cursor, 'mssql', table_name)

    if sybase_schema is not None and mssql_schema is not None:
        sybase_cols = {col['name'].lower(): col['type'].lower() for col in sybase_schema}
        mssql_cols = {col['name'].lower(): col['type'].lower() for col in mssql_schema}
        all_col_names = sorted(list(set(sybase_cols.keys()).union(mssql_cols.keys())))

        current_schemas_match = True
        for col_name in all_col_names:
            sybase_type = sybase_cols.get(col_name)
            mssql_type = mssql_cols.get(col_name)

            if sybase_type is None:
                schema_diffs.append({
                    'status': 'MSSQL_ONLY',
                    'column_name': col_name,
                    'mssql_type': mssql_type
                })
                current_schemas_match = False
            elif mssql_type is None:
                schema_diffs.append({
                    'status': 'SYBASE_ONLY',
                    'column_name': col_name,
                    'sybase_type': sybase_type
                })
                current_schemas_match = False
            elif sybase_type != mssql_type:
                schema_diffs.append({
                    'status': 'TYPE_MISMATCH',
                    'column_name': col_name,
                    'sybase_type': sybase_type,
                    'mssql_type': mssql_type
                })
                current_schemas_match = False
        summary_results['schemas_match'] = current_schemas_match
        print(f"Schemas (Columns & Types) Match: {summary_results['schemas_match']}")
    else:
        print("Skipping schema comparison due to errors in fetching one or both schemas.")


    # --- 3. Compare Data in Each Column ---
    # Only proceed with detailed data comparison if counts and schemas appear consistent
    if summary_results['counts_match'] and summary_results['schemas_match']:
        print("\nFetching all data for detailed column-by-column comparison. This might take significant time for large tables...")
        sybase_data = get_table_data(sybase_cursor, table_name, 'sybase', sybase_schema)
        mssql_data = get_table_data(mssql_cursor, table_name, 'mssql', mssql_schema)

        if sybase_data is not None and mssql_data is not None:
            current_data_matches = True
            if len(sybase_data) != len(mssql_data):
                print(f"WARNING: Data comparison skipped. Record counts differ (Sybase: {len(sybase_data)}, MSSQL: {len(mssql_data)}).")
                current_data_matches = False # Even if fetching succeeded, counts mismatch means data doesn't match
            elif not sybase_data: # Both are empty if they match here and sybase_data is empty
                print("Both tables are empty. Data comparison considered a match.")
            else:
                num_columns = len(sybase_data[0])
                for i in range(len(sybase_data)):
                    sybase_row = sybase_data[i]
                    mssql_row = mssql_data[i]
                    if sybase_row != mssql_row:
                        current_data_matches = False
                        for j in range(num_columns):
                            sybase_val = sybase_row[j]
                            mssql_val = mssql_row[j]
                            if sybase_val != mssql_val:
                                data_diffs.append({
                                    'row_index': i + 1,
                                    'col_index': j + 1,
                                    'sybase_value': str(sybase_val), # Convert to string for Excel
                                    'mssql_value': str(mssql_val)
                                })
            summary_results['data_matches'] = current_data_matches
            print(f"Data in Columns Match: {summary_results['data_matches']}")
        else:
            print("Skipping detailed data comparison due to errors in fetching one or both datasets.")
    else:
        print("Skipping detailed data comparison because record counts or schemas did not match.")

    # --- Generate Excel Report ---
    write_to_excel(table_name, summary_results, schema_diffs, data_diffs, sybase_count, mssql_count)

    # --- Close Database Connections ---
    sybase_cursor.close()
    sybase_conn.close()
    mssql_cursor.close()
    mssql_conn.close()
    print("\nDatabase connections closed.")

if __name__ == "__main__":
    main()
