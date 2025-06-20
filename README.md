import argparse
import pyodbc
import pymssql
import pandas as pd
from deepdiff import DeepDiff
import json
import os
import sys

class DatabaseComparer:

    def __init__(self, sybase_config, mssql_config):
        """
        Initializes the DatabaseComparer with database connection configurations.

        Args:
            sybase_config (dict): Dictionary with Sybase connection details
                                  (e.g., 'driver', 'server', 'database', 'uid', 'pwd').
            mssql_config (dict): Dictionary with MSSQL connection details
                                 (e.g., 'server', 'database', 'user', 'password').
        """
        self.sybase_config = sybase_config
        self.mssql_config = mssql_config
        self.sybase_conn = None
        self.mssql_conn = None

    def _connect_sybase(self):
        """Establishes a connection to the Sybase database using pyodbc."""
        try:
            conn_str = (
                f"DRIVER={{{self.sybase_config['driver']}}};"
                f"SERVER={self.sybase_config['server']};"
                f"DATABASE={self.sybase_config['database']};"
                f"UID={self.sybase_config['uid']};"
                f"PWD={self.sybase_config['pwd']}"
            )
            self.sybase_conn = pyodbc.connect(conn_str)
            print("Connected to Sybase successfully.")
        except pyodbc.Error as ex:
            sqlstate = ex.args[0]
            print(f"Error connecting to Sybase: {sqlstate} - {ex.args[1]}")
            self.sybase_conn = None # Ensure connection is None on failure
        except KeyError as ke:
            print(f"Missing Sybase configuration key: {ke}. Please check your config.json.")
            self.sybase_conn = None


    def _connect_mssql(self):
        """Establishes a connection to the MSSQL database using pymssql."""
        try:
            self.mssql_conn = pymssql.connect(
                server=self.mssql_config['server'],
                database=self.mssql_config['database'],
                user=self.mssql_config['user'],
                password=self.mssql_config['password'],
                as_dict=False # Keep it as list of tuples for consistency with pyodbc fetchall
            )
            print("Connected to MSSQL successfully.")
        except Exception as ex:
            print(f"Error connecting to MSSQL: {ex}")
            self.mssql_conn = None # Ensure connection is None on failure
        except KeyError as ke:
            print(f"Missing MSSQL configuration key: {ke}. Please check your config.json.")
            self.mssql_conn = None

    def close_connections(self):
        """Closes both database connections."""
        if self.sybase_conn:
            self.sybase_conn.close()
            print("Sybase connection closed.")
        if self.mssql_conn:
            self.mssql_conn.close()
            print("MSSQL connection closed.")

    def get_row_count(self, cursor, table_name):
        """Fetches the total row count for a given table."""
        try:
            query = f"SELECT COUNT(*) FROM {table_name}"
            cursor.execute(query)
            return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error getting row count for {table_name}: {e}")
            return None

    def get_table_data(self, cursor, table_name):
        """Fetches all data from a table into a Pandas DataFrame."""
        try:
            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            data = cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            print(f"Error fetching data for {table_name}: {e}")
            return None

    def _get_sybase_schema_details(self, cursor, table_name):
        """
        Retrieves detailed schema information for a table from Sybase.
        Returns a dictionary containing columns, indexes, PK, FKs, and constraints.
        """
        schema_info = {
            "columns": [],
            "indexes": [],
            "primary_key": [],
            "foreign_keys": [],
            "other_constraints": []
        }

        # Columns
        try:
            col_query = f"""
            SELECT
                c.name AS column_name,
                t.name AS data_type,
                c.length AS max_length,
                CASE WHEN c.isnullable = 1 THEN 'YES' ELSE 'NO' END AS is_nullable,
                (SELECT d.text FROM syscomments d WHERE d.id = c.cdefault) AS default_value,
                CASE WHEN OBJECT_PROPERTY(obj.id, 'TableHasIdentity') = 1 AND c.status & 0x80 = 0x80 THEN 'YES' ELSE 'NO' END AS is_identity
            FROM sysobjects obj
            JOIN syscolumns c ON obj.id = c.id
            JOIN systypes t ON c.type = t.type
            WHERE obj.name = '{table_name}' AND obj.type = 'U'
            ORDER BY c.colid;
            """
            cursor.execute(col_query)
            for row in cursor.fetchall():
                schema_info["columns"].append({
                    "column_name": row[0],
                    "data_type": row[1],
                    "max_length": row[2],
                    "is_nullable": row[3],
                    "default_value": row[4],
                    "is_identity": row[5]
                })
        except Exception as e:
            print(f"Error getting Sybase columns for {table_name}: {e}")

        # Primary Key & Indexes
        try:
            # Combined query for PK and other indexes
            idx_query = f"""
            SELECT
                i.name AS index_name,
                CASE
                    WHEN i.indid = 1 THEN 'Clustered'
                    WHEN i.indid = 2 THEN 'NonClustered'
                    WHEN i.indid BETWEEN 1 AND 250 THEN 'NonClustered'
                    ELSE 'Other'
                END AS index_type,
                CASE WHEN (i.status & 2) = 2 THEN 'YES' ELSE 'NO' END AS is_unique,
                COL_NAME(i.id, c.colid) AS column_name,
                INDEX_COLORDER(i.id, i.indid, k.keyno) AS column_order,
                CASE WHEN (INDEX_COLORDER(i.id, i.indid, k.keyno) & 0x8000) = 0x8000 THEN 'DESC' ELSE 'ASC' END AS sort_order,
                CASE WHEN k.keytype = 1 THEN 'YES' ELSE 'NO' END AS is_primary_key
            FROM sysindexes i
            JOIN syskeys k ON i.id = k.id AND i.indid = k.indid
            JOIN syscolumns c ON i.id = c.id AND c.colid IN (k.key1, k.key2, k.key3, k.key4, k.key5, k.key6, k.key7, k.key8, k.key9, k.key10, k.key11, k.key12, k.key13, k.key14, k.key15, k.key16)
            WHERE i.id = OBJECT_ID('{table_name}') AND i.status & 2048 = 0 -- Exclude system indexes
            ORDER BY i.name, INDEX_COLORDER(i.id, i.indid, k.keyno);
            """
            cursor.execute(idx_query)
            current_index_name = None
            current_index_cols = []
            
            for row in cursor.fetchall():
                idx_name, idx_type, is_unique, col_name, col_order, sort_order, is_pk = row
                if idx_name != current_index_name:
                    if current_index_name:
                        # Append previous index
                        schema_info["indexes"].append({
                            "index_name": current_index_name,
                            "index_type": current_index_type,
                            "is_unique": current_is_unique,
                            "columns": current_index_cols
                        })
                        if current_is_pk == 'YES':
                            # Simplify PK to just columns, as DeepDiff compares dicts
                            schema_info["primary_key"].append({
                                "constraint_name": current_index_name,
                                "columns": current_index_cols
                            })
                        elif current_is_unique == 'YES' and current_is_pk == 'NO':
                             schema_info["other_constraints"].append({
                                "type": "UNIQUE",
                                "constraint_name": current_index_name,
                                "columns": current_index_cols
                            })
                    # Start new index
                    current_index_name = idx_name
                    current_index_type = idx_type
                    current_is_unique = is_unique
                    current_is_pk = is_pk
                    current_index_cols = []
                current_index_cols.append({"column_name": col_name, "sort_order": sort_order})
            
            # Append the last index after loop
            if current_index_name:
                schema_info["indexes"].append({
                    "index_name": current_index_name,
                    "index_type": current_index_type,
                    "is_unique": current_is_unique,
                    "columns": current_index_cols
                })
                if current_is_pk == 'YES':
                    schema_info["primary_key"].append({
                        "constraint_name": current_index_name,
                        "columns": current_index_cols
                    })
                elif current_is_unique == 'YES' and current_is_pk == 'NO':
                     schema_info["other_constraints"].append({
                        "type": "UNIQUE",
                        "constraint_name": current_index_name,
                        "columns": current_index_cols
                    })

        except Exception as e:
            print(f"Error getting Sybase indexes/PK for {table_name}: {e}")

        # Foreign Keys
        try:
            fk_query = f"""
            SELECT
                obj.name AS fk_constraint_name,
                COL_NAME(fkc.fkeyid, fkc.fkeycolid) AS fk_column,
                OBJECT_NAME(fkc.rkeyid) AS referenced_table,
                COL_NAME(fkc.rkeyid, fkc.rkeycolid) AS referenced_column
            FROM sysforeignkeys fkc
            JOIN sysobjects obj ON fkc.constrid = obj.id
            WHERE fkc.fkeyid = OBJECT_ID('{table_name}');
            """
            cursor.execute(fk_query)
            current_fk_name = None
            current_fk_columns = []
            current_ref_table = None
            current_ref_columns = []

            for row in cursor.fetchall():
                fk_name, fk_col, ref_table, ref_col = row
                if fk_name != current_fk_name:
                    if current_fk_name:
                        schema_info["foreign_keys"].append({
                            "constraint_name": current_fk_name,
                            "fk_columns": current_fk_columns,
                            "referenced_table": current_ref_table,
                            "referenced_columns": current_ref_columns
                        })
                    current_fk_name = fk_name
                    current_fk_columns = []
                    current_ref_table = ref_table
                    current_ref_columns = []
                current_fk_columns.append(fk_col)
                current_ref_columns.append(ref_col)
            if current_fk_name: # Add last FK
                schema_info["foreign_keys"].append({
                    "constraint_name": current_fk_name,
                    "fk_columns": current_fk_columns,
                    "referenced_table": current_ref_table,
                    "referenced_columns": current_ref_columns
                })

        except Exception as e:
            print(f"Error getting Sybase foreign keys for {table_name}: {e}")

        # Other Constraints (Check constraints - this is much harder in Sybase via sys.tables)
        # Sybase check constraints are stored as text in syscomments. Linking them directly
        # to a constraint name is non-trivial without parsing. Simplified for now.
        # For actual check constraints, you'd need more complex queries or manual parsing.
        
        return schema_info

    def _get_mssql_schema_details(self, cursor, table_name):
        """
        Retrieves detailed schema information for a table from MSSQL.
        Returns a dictionary containing columns, indexes, PK, FKs, and constraints.
        """
        schema_info = {
            "columns": [],
            "indexes": [],
            "primary_key": [],
            "foreign_keys": [],
            "other_constraints": []
        }
        
        # Columns
        try:
            col_query = f"""
            SELECT
                c.name AS column_name,
                t.name AS data_type,
                c.max_length,
                c.precision,
                c.scale,
                CASE WHEN c.is_nullable = 1 THEN 'YES' ELSE 'NO' END AS is_nullable,
                dc.definition AS default_value,
                c.is_identity
            FROM sys.columns c
            JOIN sys.types t ON c.user_type_id = t.user_type_id
            LEFT JOIN sys.default_constraints dc ON c.default_object_id = dc.object_id AND c.object_id = dc.parent_object_id
            WHERE c.object_id = OBJECT_ID('{table_name}')
            ORDER BY c.column_id;
            """
            cursor.execute(col_query)
            for row in cursor.fetchall():
                schema_info["columns"].append({
                    "column_name": row[0],
                    "data_type": row[1],
                    "max_length": row[2],
                    "precision": row[3],
                    "scale": row[4],
                    "is_nullable": row[5],
                    "default_value": row[6],
                    "is_identity": "YES" if row[7] else "NO"
                })
        except Exception as e:
            print(f"Error getting MSSQL columns for {table_name}: {e}")

        # Indexes & Primary Key & Unique Constraints (all from sys.indexes and related views)
        try:
            idx_pk_uq_query = f"""
            SELECT
                ind.name AS index_or_constraint_name,
                ind.type_desc AS type_description,
                ind.is_unique,
                ind.is_primary_key,
                ind.is_unique_constraint,
                col.name AS column_name,
                ic.is_descending_key AS is_descending,
                ic.is_included_column
            FROM sys.indexes ind
            JOIN sys.index_columns ic ON ind.object_id = ic.object_id AND ind.index_id = ic.index_id
            JOIN sys.columns col ON ic.object_id = col.object_id AND ic.column_id = col.column_id
            WHERE ind.object_id = OBJECT_ID('{table_name}')
            ORDER BY ind.name, ic.key_ordinal;
            """
            cursor.execute(idx_pk_uq_query)
            
            # Group results by index/constraint name
            grouped_results = {}
            for row in cursor.fetchall():
                name, type_desc, is_unique, is_pk, is_uq_constraint, col_name, is_descending, is_included = row
                if name not in grouped_results:
                    grouped_results[name] = {
                        "type_desc": type_desc,
                        "is_unique": is_unique,
                        "is_primary_key": is_pk,
                        "is_unique_constraint": is_uq_constraint,
                        "columns": [],
                        "included_columns": []
                    }
                sort_order = "DESC" if is_descending else "ASC"
                if not is_included:
                    grouped_results[name]["columns"].append({"column_name": col_name, "sort_order": sort_order})
                else:
                    grouped_results[name]["included_columns"].append(col_name)

            for name, details in grouped_results.items():
                if details["is_primary_key"]:
                    schema_info["primary_key"].append({
                        "constraint_name": name,
                        "columns": details["columns"]
                    })
                elif details["is_unique_constraint"]:
                    schema_info["other_constraints"].append({
                        "type": "UNIQUE",
                        "constraint_name": name,
                        "columns": details["columns"]
                    })
                else: # Regular index
                    schema_info["indexes"].append({
                        "index_name": name,
                        "index_type": details["type_desc"],
                        "is_unique": "YES" if details["is_unique"] else "NO",
                        "columns": details["columns"],
                        "included_columns": details["included_columns"]
                    })
        except Exception as e:
            print(f"Error getting MSSQL indexes/PK/Unique constraints for {table_name}: {e}")

        # Foreign Keys
        try:
            fk_query = f"""
            SELECT
                fk.name AS fk_constraint_name,
                COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS fk_column,
                OBJECT_NAME(fk.referenced_object_id) AS referenced_table,
                COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS referenced_column
            FROM sys.foreign_keys fk
            JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            WHERE fk.parent_object_id = OBJECT_ID('{table_name}');
            """
            cursor.execute(fk_query)
            current_fk_name = None
            current_fk_columns = []
            current_ref_table = None
            current_ref_columns = []

            for row in cursor.fetchall():
                fk_name, fk_col, ref_table, ref_col = row
                if fk_name != current_fk_name:
                    if current_fk_name:
                        schema_info["foreign_keys"].append({
                            "constraint_name": current_fk_name,
                            "fk_columns": current_fk_columns,
                            "referenced_table": current_ref_table,
                            "referenced_columns": current_ref_columns
                        })
                    current_fk_name = fk_name
                    current_fk_columns = []
                    current_ref_table = ref_table
                    current_ref_columns = []
                current_fk_columns.append(fk_col)
                current_ref_columns.append(ref_col)
            if current_fk_name: # Add last FK
                schema_info["foreign_keys"].append({
                    "constraint_name": current_fk_name,
                    "fk_columns": current_fk_columns,
                    "referenced_table": current_ref_table,
                    "referenced_columns": current_ref_columns
                })
        except Exception as e:
            print(f"Error getting MSSQL foreign keys for {table_name}: {e}")

        # Other Constraints (Check)
        try:
            check_query = f"""
            SELECT
                cc.name AS constraint_name,
                cc.definition AS check_definition
            FROM sys.check_constraints cc
            WHERE cc.parent_object_id = OBJECT_ID('{table_name}');
            """
            cursor.execute(check_query)
            for row in cursor.fetchall():
                schema_info["other_constraints"].append({
                    "type": "CHECK",
                    "constraint_name": row[0],
                    "definition": row[1]
                })

        except Exception as e:
            print(f"Error getting MSSQL other constraints (CHECK) for {table_name}: {e}")

        return schema_info

    def _get_data_quality_metrics(self, df):
        """Calculates various data quality metrics for a DataFrame."""
        if df is None or df.empty:
            return None

        metrics = {}
        for col in df.columns:
            col_series = df[col]
            col_metrics = {
                "total_rows": len(col_series),
                "null_count": col_series.isnull().sum(),
                "unique_count": col_series.nunique()
            }

            # For numeric columns, add min, max, mean
            if pd.api.types.is_numeric_dtype(col_series):
                # Handle potential NaNs in min/max/mean if all values are null
                col_metrics["min"] = col_series.min() if not col_series.dropna().empty else None
                col_metrics["max"] = col_series.max() if not col_series.dropna().empty else None
                col_metrics["mean"] = col_series.mean() if not col_series.dropna().empty else None
            
            metrics[col] = col_metrics
        return metrics

    def compare_table(self, table_name):
        """
        Performs a full comparison of schema and data for a given table.

        Args:
            table_name (str): The name of the table to compare.

        Returns:
            dict: A dictionary containing all comparison results.
        """
        self._connect_sybase()
        self._connect_mssql()

        if not self.sybase_conn or not self.mssql_conn:
            print("Failed to establish one or both database connections. Cannot proceed with comparison.")
            return None

        sybase_cursor = self.sybase_conn.cursor()
        mssql_cursor = self.mssql_conn.cursor()

        comparison_results = {
            "table_name": table_name,
            "row_count_comparison": {},
            "schema_comparison": {},
            "data_quality_comparison": {},
            "data_differences": {},
            "raw_sybase_schema": {}, # For detailed schema comparison later
            "raw_mssql_schema": {}   # For detailed schema comparison later
        }

        print(f"\n--- Starting comparison for table: {table_name} ---")

        # 1. Row Count Comparison
        print("1. Comparing row counts...")
        sybase_count = self.get_row_count(sybase_cursor, table_name)
        mssql_count = self.get_row_count(mssql_cursor, table_name)

        comparison_results["row_count_comparison"] = {
            "sybase_count": sybase_count,
            "mssql_count": mssql_count,
            "match": sybase_count == mssql_count if sybase_count is not None and mssql_count is not None else False
        }
        if sybase_count == mssql_count and sybase_count is not None:
            print(f"   Row counts match: {sybase_count}")
        elif sybase_count is None or mssql_count is None:
             print("   Could not retrieve row counts from one or both databases.")
        else:
            print(f"   Row count mismatch: Sybase={sybase_count}, MSSQL={mssql_count}")


        # 2. Schema Comparison (detailed)
        print("2. Retrieving and comparing schema details (columns, indexes, PK, FK, constraints)...")
        sybase_schema = self._get_sybase_schema_details(sybase_cursor, table_name)
        mssql_schema = self._get_mssql_schema_details(mssql_cursor, table_name)

        comparison_results["raw_sybase_schema"] = sybase_schema
        comparison_results["raw_mssql_schema"] = mssql_schema

        # Use DeepDiff for robust schema comparison
        schema_diff = DeepDiff(sybase_schema, mssql_schema, ignore_order=True,
                               significant_digits=5, # For numeric properties
                               view='tree', # 'report' or 'tree'
                               custom_operators=[
                                   (lambda x: isinstance(x, str), lambda x, y: x.strip().lower() == y.strip().lower())
                               ]) # Case-insensitive string comparison for schema elements

        comparison_results["schema_comparison"] = {
            "diff_found": bool(schema_diff),
            "details": schema_diff.to_json() if schema_diff else "No schema differences found."
        }
        if schema_diff:
            print(f"   Schema differences found. See report for details.")
        else:
            print(f"   No significant schema differences found.")

        # 3. Data Quality and Full Data Comparison
        print("3. Retrieving and comparing data (data quality metrics and row-level diff)...")
        sybase_df = self.get_table_data(sybase_cursor, table_name)
        mssql_df = self.get_table_data(mssql_cursor, table_name)

        if sybase_df is not None and mssql_df is not None:
            # Data Quality Metrics
            sybase_dq = self._get_data_quality_metrics(sybase_df)
            mssql_dq = self._get_data_quality_metrics(mssql_df)

            comparison_results["data_quality_comparison"] = {
                "sybase_metrics": sybase_dq,
                "mssql_metrics": mssql_dq
            }
            print("   Data quality metrics calculated.")

            # Full Data Differences (using DeepDiff on DataFrames)
            # Normalize DataFrames for comparison: sort by primary key (if available) or all columns
            try:
                # Attempt to sort by primary key if available and consistent
                # Retrieve PK column names dynamically
                pk_cols_sybase_names = []
                if sybase_schema and sybase_schema.get('primary_key'):
                    for pk_info in sybase_schema['primary_key']:
                        if 'columns' in pk_info:
                            for col_info in pk_info['columns']:
                                if 'column_name' in col_info:
                                    pk_cols_sybase_names.append(col_info['column_name'])

                pk_cols_mssql_names = []
                if mssql_schema and mssql_schema.get('primary_key'):
                     for pk_info in mssql_schema['primary_key']:
                        if 'columns' in pk_info:
                            for col_info in pk_info['columns']:
                                if 'column_name' in col_info:
                                    pk_cols_mssql_names.append(col_info['column_name'])


                if pk_cols_sybase_names and pk_cols_sybase_names == pk_cols_mssql_names and all(col in sybase_df.columns for col in pk_cols_sybase_names):
                    sybase_df_sorted = sybase_df.sort_values(by=pk_cols_sybase_names).reset_index(drop=True)
                    mssql_df_sorted = mssql_df.sort_values(by=pk_cols_mssql_names).reset_index(drop=True)
                    print(f"   Sorting dataframes by primary key columns: {pk_cols_sybase_names}")
                else:
                    # Fallback to sorting by all common columns
                    common_cols = list(set(sybase_df.columns) & set(mssql_df.columns))
                    if not common_cols:
                        print("   Warning: No common columns found for data comparison. Skipping detailed data diff.")
                        data_diff = "Skipped: No common columns."
                    else:
                        sybase_df_sorted = sybase_df[common_cols].sort_values(by=common_cols).reset_index(drop=True)
                        mssql_df_sorted = mssql_df[common_cols].sort_values(by=common_cols).reset_index(drop=True)
                        print("   Sorting dataframes by all common columns.")
            except Exception as e:
                print(f"   Warning: Could not sort DataFrames for comparison, attempting unsorted diff. Error: {e}")
                sybase_df_sorted = sybase_df
                mssql_df_sorted = mssql_df


            data_diff = DeepDiff(
                sybase_df_sorted.to_dict(orient='records'),
                mssql_df_sorted.to_dict(orient='records'),
                ignore_order=True,              # Ignore row order if PK sorting wasn't perfect or applicable
                ignore_numeric_type_changes=True, # e.g., Sybase INT to MSSQL BIGINT
                significant_digits=5,           # For float comparisons
                view='tree',                    # 'report' provides summary, 'tree' is detailed
                custom_operators=[
                   (lambda x: isinstance(x, str), lambda x, y: str(x).strip().lower() == str(y).strip().lower())
                ] # Case-insensitive string comparison for data values, handles non-string types gracefully
            )

            comparison_results["data_differences"] = {
                "diff_found": bool(data_diff),
                "details": data_diff.to_json() if data_diff else "No data differences found."
            }
            if data_diff:
                print(f"   Data differences found. See report for details.")
            else:
                print(f"   No data differences found.")
        else:
            print("   Skipping data quality and differences due to data retrieval errors.")

        print(f"--- Finished comparison for table: {table_name} ---")
        return comparison_results

    def generate_excel_report(self, comparison_results, output_dir="."):
        """
        Generates an Excel report for the comparison results.
        Beyond these, the script also provides highly detailed comparison reports.

        Args:
            comparison_results (dict): The dictionary containing comparison results.
            output_dir (str): Directory to save the Excel file.
        """
        table_name = comparison_results["table_name"]
        # Updated file naming
        file_path = os.path.join(output_dir, f"{table_name}.xlsx")

        print(f"Generating Excel report: {file_path}")

        try:
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                # Helper function to write potentially empty lists or dicts
                def write_sheet_from_list(data_list, sheet_name, default_message="N/A", key_column='column_name'):
                    if not data_list:
                        pd.DataFrame({"Details": [f"No {sheet_name.replace('_', ' ')} found for this table." if default_message == "N/A" else default_message]}).to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        if all(isinstance(d, dict) for d in data_list):
                            # Try to normalize if it's a list of dictionaries
                            df = pd.json_normalize(data_list, sep='_')
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                        else:
                            # Fallback if not all dicts, just dump as JSON string
                            pd.DataFrame({"Details": [json.dumps(data_list, indent=2)]}).to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Adjust column width for DeepDiff outputs
                        if sheet_name in ['Schema_Comparison', 'Data_Differences']:
                             worksheet = writer.sheets[sheet_name]
                             worksheet.set_column('A:A', 100)


                # 1. Summary Sheet
                summary_data = {
                    "Metric": ["Table Name", "Sybase Row Count", "MSSQL Row Count", "Row Count Match",
                               "Schema Differences Found", "Data Differences Found"],
                    "Value": [
                        table_name,
                        comparison_results["row_count_comparison"].get("sybase_count", "N/A"),
                        comparison_results["row_count_comparison"].get("mssql_count", "N/A"),
                        comparison_results["row_count_comparison"].get("match", "N/A"),
                        comparison_results["schema_comparison"].get("diff_found", "N/A"),
                        comparison_results["data_differences"].get("diff_found", "N/A")
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                # 2. Schema Comparison Sheet (DeepDiff output)
                write_sheet_from_list(
                    json.loads(comparison_results["schema_comparison"]["details"]) if isinstance(comparison_results["schema_comparison"]["details"], str) and "{" in comparison_results["schema_comparison"]["details"] else comparison_results["schema_comparison"]["details"],
                    'Schema_Comparison',
                    default_message="No schema differences found."
                )
                
                # 3. Detailed Schema (Sybase)
                write_sheet_from_list(comparison_results["raw_sybase_schema"].get("columns", []), 'Sybase_Columns')
                write_sheet_from_list(comparison_results["raw_sybase_schema"].get("indexes", []), 'Sybase_Indexes')
                write_sheet_from_list(comparison_results["raw_sybase_schema"].get("primary_key", []), 'Sybase_PK')
                write_sheet_from_list(comparison_results["raw_sybase_schema"].get("foreign_keys", []), 'Sybase_FKs')
                write_sheet_from_list(comparison_results["raw_sybase_schema"].get("other_constraints", []), 'Sybase_Other_Constraints')

                # 4. Detailed Schema (MSSQL)
                write_sheet_from_list(comparison_results["raw_mssql_schema"].get("columns", []), 'MSSQL_Columns')
                write_sheet_from_list(comparison_results["raw_mssql_schema"].get("indexes", []), 'MSSQL_Indexes')
                write_sheet_from_list(comparison_results["raw_mssql_schema"].get("primary_key", []), 'MSSQL_PK')
                write_sheet_from_list(comparison_results["raw_mssql_schema"].get("foreign_keys", []), 'MSSQL_FKs')
                write_sheet_from_list(comparison_results["raw_mssql_schema"].get("other_constraints", []), 'MSSQL_Other_Constraints')

                # 5. Data Quality Metrics Sheet
                sybase_dq_df = pd.DataFrame(comparison_results["data_quality_comparison"].get("sybase_metrics", {})).T.reset_index().rename(columns={'index': 'Column_Name'})
                mssql_dq_df = pd.DataFrame(comparison_results["data_quality_comparison"].get("mssql_metrics", {})).T.reset_index().rename(columns={'index': 'Column_Name'})

                if not sybase_dq_df.empty or not mssql_dq_df.empty:
                    data_quality_df = pd.merge(sybase_dq_df, mssql_dq_df, on='Column_Name', suffixes=('_Sybase', '_MSSQL'), how='outer')
                    data_quality_df.to_excel(writer, sheet_name='Data_Quality_Metrics', index=False)
                else:
                    pd.DataFrame({"Details": ["Data quality metrics could not be retrieved for one or both databases or tables are empty."]}).to_excel(writer, sheet_name='Data_Quality_Metrics', index=False)

                # 6. Data Differences Sheet (DeepDiff output)
                write_sheet_from_list(
                    json.loads(comparison_results["data_differences"]["details"]) if isinstance(comparison_results["data_differences"]["details"], str) and "{" in comparison_results["data_differences"]["details"] else comparison_results["data_differences"]["details"],
                    'Data_Differences',
                    default_message="No data differences found."
                )

            print(f"Excel report saved successfully to {file_path}")

        except Exception as e:
            print(f"Error generating Excel report: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compare a table between Sybase and MSSQL Server using config.json.")
    parser.add_argument("--table_name", required=True, help="Name of the table to compare.")
    parser.add_argument("--config_file", default="config.json", help="Path to the JSON configuration file. Defaults to config.json.")

    args = parser.parse_args()

    # Load configuration from JSON file
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        print("Please create a config.json file with database connection details.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{args.config_file}'. Please check JSON syntax.")
        sys.exit(1)

    sybase_config = config.get("sybase")
    mssql_config = config.get("mssql")
    output_dir = config.get("output_directory", ".") # Default to current dir if not in config

    if not sybase_config:
        print("Error: 'sybase' configuration not found in config.json.")
        sys.exit(1)
    if not mssql_config:
        print("Error: 'mssql' configuration not found in config.json.")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    comparer = DatabaseComparer(sybase_config, mssql_config)
    try:
        results = comparer.compare_table(args.table_name)
        if results:
            comparer.generate_excel_report(results, output_dir)
    finally:
        comparer.close_connections()

if __name__ == "__main__":
    main()
