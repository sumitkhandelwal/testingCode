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





    # --- 3. Compare Data in Each Column (Value-by-Value) ---

    # Fetch data here so it can be passed to both data comparison sheets

    sybase_data = None

    mssql_data = None

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

    filename = f"{table_name}.xlsx"

    workbook = openpyxl.Workbook()



    write_summary_and_schema_to_excel(workbook, table_name, summary_results, schema_diffs, sybase_count, mssql_count)

    write_data_discrepancies_sheet(workbook, data_diffs)

    

    # Only call full data comparison if data was successfully retrieved

    if sybase_data is not None and mssql_data is not None:

        write_full_data_comparison_sheet(workbook, table_name, sybase_data, mssql_data, sybase_schema)

    else:

        print("Skipping full data comparison sheet generation as data could not be retrieved.")



    try:

        workbook.save(filename)

        print(f"\nComparison report saved to '{filename}' successfully.")

    except Exception as ex:

        print(f"ERROR: Could not save Excel file '{filename}': {ex}")





    # --- Close Database Connections ---

    sybase_cursor.close()

    sybase_conn.close()

    mssql_cursor.close()

    mssql_conn.close()

    print("\nDatabase connections closed.")



if __name__ == "__main__":

    main()

