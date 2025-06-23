def write_full_data_comparison_sheet(workbook, table_name, sybase_data, mssql_data, sybase_schema):

    """

    Writes the full data comparison, row by row based on the first column, to a new Excel sheet.

    Highlights discrepancies and rows unique to one database.

    """

    full_data_sheet = workbook.create_sheet("Full Data Comparison")

    full_data_sheet['A1'] = f"Full Data Comparison for Table: {table_name}"

    full_data_sheet['A1'].font = Font(bold=True, size=14)



    # Prepare column headers

    if sybase_schema:

        # Create a list of column names for the header row

        sybase_column_names = [col['name'] for col in sybase_schema]

        headers = ["Database"] + sybase_column_names # Add a 'Database' column to distinguish Sybase/MSSQL rows

    else:

        # Fallback if schema is not available, just use generic headers

        num_cols = len(sybase_data[0]) if sybase_data else (len(mssql_data[0]) if mssql_data else 0)

        headers = ["Database"] + [f"Column {i+1}" for i in range(num_cols)]



    # Write header row

    full_data_sheet.append(headers)

    # Apply bold font to header row

    for cell in full_data_sheet[2]:

        cell.font = Font(bold=True)



    # Create maps for quick lookup by the first column's value

    # Important: If the first column is not unique, this will only store the last encountered row for that key.

    sybase_map = {row[0]: row for row in sybase_data} if sybase_data else {}

    mssql_map = {row[0]: row for row in mssql_data} if mssql_data else {}



    all_keys = sorted(list(set(sybase_map.keys()).union(mssql_map.keys())))



    current_row_excel = 2 # Start writing data from row 3 (after title and headers)



    for key in all_keys:

        sybase_row_data = sybase_map.get(key)

        mssql_row_data = mssql_map.get(key)



        # Write Sybase row

        if sybase_row_data:

            current_row_excel += 1

            row_to_write = ["Sybase"] + list(sybase_row_data)

            full_data_sheet.append(row_to_write)

            

            # Apply fill and check for mismatches against MSSQL data

            for col_idx in range(len(sybase_row_data)):

                cell = full_data_sheet.cell(row=current_row_excel, column=col_idx + 2) # +2 for "Database" column and 0-index

                if mssql_row_data is None: # Row only in Sybase

                    cell.fill = GREEN_FILL

                elif sybase_row_data[col_idx] != mssql_row_data[col_idx]: # Mismatch for this column

                    cell.fill = RED_FILL

        

        # Write MSSQL row

        if mssql_row_data:

            current_row_excel += 1

            row_to_write = ["MSSQL"] + list(mssql_row_data)

            full_data_sheet.append(row_to_write)

            

            # Apply fill and check for mismatches against Sybase data

            for col_idx in range(len(mssql_row_data)):

                cell = full_data_sheet.cell(row=current_row_excel, column=col_idx + 2) # +2 for "Database" column and 0-index

                if sybase_row_data is None: # Row only in MSSQL

                    cell.fill = BLUE_FILL

                elif sybase_row_data[col_idx] != mssql_row_data[col_idx]: # Mismatch for this column

                    cell.fill = RED_FILL



        # Add a separator row for better readability if both were present or if it's the end of a block

        current_row_excel += 1

        full_data_sheet.append([''] * len(headers)) # Empty row as separator

        # Optional: Add a subtle border

        for col_idx in range(1, len(headers) + 1):

            full_data_sheet.cell(row=current_row_excel, column=col_idx).border = Border(top=Side(style='dotted'))





    # Auto-width for Full Data Comparison Sheet

    for col_idx, column in enumerate(full_data_sheet.iter_cols()):

        max_length = 0

        for cell in column:

            try:

                if cell.value is not None and len(str(cell.value)) > max_length:

                    max_length = len(str(cell.value))

            except:

                pass

        adjusted_width = (max_length + 2) * 1.2

        full_data_sheet.column_dimensions[get_column_letter(col_idx + 1)].width = adjusted_width
