"""
Excel export module for extracted data output.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from io import BytesIO

logger = logging.getLogger(__name__)

class ExcelExporter:
    """Export extracted data to Excel format."""
    
    def __init__(self):
        """Initialize Excel exporter."""
        self.default_columns = [
            'document_name',
            'page_number',
            'field_type',
            'extracted_value',
            'confidence',
            'source',
            'context',
            'validation_status',
            'processing_date'
        ]
    
    def export_extracted_fields(self, 
                               extracted_data: Dict[str, Dict[str, Any]], 
                               output_path: str = None,
                               include_metadata: bool = True) -> bytes:
        """
        Export extracted field data to Excel.
        
        Args:
            extracted_data: Dictionary with document_name -> fields data
            output_path: Path to save Excel file (optional)
            include_metadata: Whether to include metadata sheets
            
        Returns:
            Excel file as bytes
        """
        try:
            # Create Excel writer
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Main data sheet
                main_data = self._prepare_main_data(extracted_data)
                df_main = pd.DataFrame(main_data)
                df_main.to_excel(writer, sheet_name='Extracted_Fields', index=False)
                
                # Format main sheet
                workbook = writer.book
                worksheet = writer.sheets['Extracted_Fields']
                self._format_main_sheet(workbook, worksheet, df_main)
                
                if include_metadata:
                    # Summary sheet
                    summary_data = self._prepare_summary_data(extracted_data)
                    df_summary = pd.DataFrame(summary_data)
                    df_summary.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Statistics sheet
                    stats_data = self._prepare_statistics_data(extracted_data)
                    df_stats = pd.DataFrame(stats_data)
                    df_stats.to_excel(writer, sheet_name='Statistics', index=False)
                    
                    # Format additional sheets
                    self._format_summary_sheet(workbook, writer.sheets['Summary'])
                    self._format_statistics_sheet(workbook, writer.sheets['Statistics'])
            
            # Get the Excel data
            excel_data = output.getvalue()
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(excel_data)
                logger.info(f"Excel file saved to {output_path}")
            
            return excel_data
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            raise
    
    def export_tables(self, 
                     table_data: List[Dict[str, Any]], 
                     output_path: str = None) -> bytes:
        """
        Export extracted table data to Excel.
        
        Args:
            table_data: List of table dictionaries
            output_path: Path to save Excel file (optional)
            
        Returns:
            Excel file as bytes
        """
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Create a sheet for each table
                for i, table in enumerate(table_data):
                    sheet_name = f"Table_{i+1}_Page_{table.get('page_number', 'Unknown')}"
                    
                    # Convert table to DataFrame
                    if 'headers' in table and 'rows' in table:
                        df = pd.DataFrame(table['rows'], columns=table['headers'])
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        
                        # Format table sheet
                        worksheet = writer.sheets[sheet_name]
                        self._format_table_sheet(workbook, worksheet, df)
                
                # Create summary sheet for tables
                if table_data:
                    table_summary = self._prepare_table_summary(table_data)
                    df_summary = pd.DataFrame(table_summary)
                    df_summary.to_excel(writer, sheet_name='Table_Summary', index=False)
            
            excel_data = output.getvalue()
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(excel_data)
                logger.info(f"Table Excel file saved to {output_path}")
            
            return excel_data
            
        except Exception as e:
            logger.error(f"Error exporting tables to Excel: {e}")
            raise
    
    def export_validation_results(self, 
                                validation_data: Dict[str, Dict[str, Any]], 
                                output_path: str = None) -> bytes:
        """
        Export validation results to Excel.
        
        Args:
            validation_data: Dictionary with document_name -> validation results
            output_path: Path to save Excel file (optional)
            
        Returns:
            Excel file as bytes
        """
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Validation results sheet
                validation_rows = []
                
                for doc_name, fields in validation_data.items():
                    for field_name, validation_result in fields.items():
                        validation_rows.append({
                            'document_name': doc_name,
                            'field_name': field_name,
                            'is_valid': validation_result.is_valid,
                            'confidence': validation_result.confidence,
                            'errors': '; '.join(validation_result.errors),
                            'warnings': '; '.join(validation_result.warnings),
                            'suggestions': '; '.join(validation_result.suggestions)
                        })
                
                df_validation = pd.DataFrame(validation_rows)
                df_validation.to_excel(writer, sheet_name='Validation_Results', index=False)
                
                # Format validation sheet
                workbook = writer.book
                worksheet = writer.sheets['Validation_Results']
                self._format_validation_sheet(workbook, worksheet, df_validation)
            
            excel_data = output.getvalue()
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(excel_data)
                logger.info(f"Validation Excel file saved to {output_path}")
            
            return excel_data
            
        except Exception as e:
            logger.error(f"Error exporting validation results to Excel: {e}")
            raise
    
    def _prepare_main_data(self, extracted_data: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """Prepare main extraction data for Excel export."""
        rows = []
        
        for doc_name, fields in extracted_data.items():
            for field_name, field_data in fields.items():
                if isinstance(field_data, dict):
                    row = {
                        'document_name': doc_name,
                        'field_type': field_name,
                        'extracted_value': field_data.get('value', ''),
                        'confidence': field_data.get('confidence', 0.0),
                        'source': field_data.get('source', ''),
                        'page_number': field_data.get('page_number', 0),
                        'context': field_data.get('context', ''),
                        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                else:
                    row = {
                        'document_name': doc_name,
                        'field_type': field_name,
                        'extracted_value': str(field_data),
                        'confidence': 0.5,
                        'source': 'unknown',
                        'page_number': 0,
                        'context': '',
                        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                rows.append(row)
        
        return rows
    
    def _prepare_summary_data(self, extracted_data: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """Prepare summary data for Excel export."""
        summary_rows = []
        
        for doc_name, fields in extracted_data.items():
            total_fields = len(fields)
            high_confidence = sum(1 for f in fields.values() 
                                if isinstance(f, dict) and f.get('confidence', 0) >= 0.8)
            user_corrections = sum(1 for f in fields.values() 
                                 if isinstance(f, dict) and f.get('source') == 'user_correction')
            
            summary_rows.append({
                'document_name': doc_name,
                'total_fields': total_fields,
                'high_confidence_fields': high_confidence,
                'user_corrections': user_corrections,
                'extraction_rate': f"{(high_confidence/total_fields)*100:.1f}%" if total_fields > 0 else "0%"
            })
        
        return summary_rows
    
    def _prepare_statistics_data(self, extracted_data: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """Prepare statistics data for Excel export."""
        field_stats = {}
        
        # Aggregate statistics by field type
        for doc_name, fields in extracted_data.items():
            for field_name, field_data in fields.items():
                if field_name not in field_stats:
                    field_stats[field_name] = {
                        'field_type': field_name,
                        'total_occurrences': 0,
                        'high_confidence_count': 0,
                        'user_corrections_count': 0,
                        'sources': set()
                    }
                
                field_stats[field_name]['total_occurrences'] += 1
                
                if isinstance(field_data, dict):
                    confidence = field_data.get('confidence', 0)
                    source = field_data.get('source', 'unknown')
                    
                    if confidence >= 0.8:
                        field_stats[field_name]['high_confidence_count'] += 1
                    
                    if source == 'user_correction':
                        field_stats[field_name]['user_corrections_count'] += 1
                    
                    field_stats[field_name]['sources'].add(source)
        
        # Convert to list format
        stats_rows = []
        for field_name, stats in field_stats.items():
            stats_rows.append({
                'field_type': field_name,
                'total_occurrences': stats['total_occurrences'],
                'high_confidence_count': stats['high_confidence_count'],
                'confidence_rate': f"{(stats['high_confidence_count']/stats['total_occurrences'])*100:.1f}%",
                'user_corrections': stats['user_corrections_count'],
                'sources': ', '.join(stats['sources'])
            })
        
        return stats_rows
    
    def _prepare_table_summary(self, table_data: List[Dict[str, Any]]) -> List[Dict]:
        """Prepare table summary data."""
        summary_rows = []
        
        for i, table in enumerate(table_data):
            summary_rows.append({
                'table_number': i + 1,
                'page_number': table.get('page_number', 'Unknown'),
                'row_count': table.get('row_count', 0),
                'column_count': table.get('column_count', 0),
                'headers': ', '.join(table.get('headers', [])),
                'data_cells': table.get('row_count', 0) * table.get('column_count', 0)
            })
        
        return summary_rows
    
    def _format_main_sheet(self, workbook, worksheet, df):
        """Format the main extraction data sheet."""
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        confidence_high = workbook.add_format({'bg_color': '#C6EFCE'})
        confidence_medium = workbook.add_format({'bg_color': '#FFEB9C'})
        confidence_low = workbook.add_format({'bg_color': '#FFC7CE'})
        
        # Set column widths
        worksheet.set_column('A:A', 20)  # document_name
        worksheet.set_column('B:B', 15)  # field_type
        worksheet.set_column('C:C', 25)  # extracted_value
        worksheet.set_column('D:D', 10)  # confidence
        worksheet.set_column('E:E', 15)  # source
        worksheet.set_column('F:F', 10)  # page_number
        worksheet.set_column('G:G', 30)  # context
        worksheet.set_column('H:H', 20)  # processing_date
        
        # Format header row
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Apply conditional formatting for confidence
        if 'confidence' in df.columns:
            conf_col = df.columns.get_loc('confidence')
            worksheet.conditional_format(1, conf_col, len(df), conf_col, {
                'type': 'cell',
                'criteria': '>=',
                'value': 0.8,
                'format': confidence_high
            })
            worksheet.conditional_format(1, conf_col, len(df), conf_col, {
                'type': 'cell',
                'criteria': 'between',
                'minimum': 0.5,
                'maximum': 0.79,
                'format': confidence_medium
            })
            worksheet.conditional_format(1, conf_col, len(df), conf_col, {
                'type': 'cell',
                'criteria': '<',
                'value': 0.5,
                'format': confidence_low
            })
    
    def _format_summary_sheet(self, workbook, worksheet):
        """Format the summary sheet."""
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#B4C6E7',
            'border': 1
        })
        
        # Set column widths
        worksheet.set_column('A:E', 20)
        
        # Apply header format (assuming row 0 contains headers)
        for col in range(5):
            worksheet.write(0, col, '', header_format)
    
    def _format_statistics_sheet(self, workbook, worksheet):
        """Format the statistics sheet."""
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#F2CC8F',
            'border': 1
        })
        
        # Set column widths
        worksheet.set_column('A:F', 20)
        
        # Apply header format
        for col in range(6):
            worksheet.write(0, col, '', header_format)
    
    def _format_validation_sheet(self, workbook, worksheet, df):
        """Format the validation results sheet."""
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#FFD966',
            'border': 1
        })
        
        valid_format = workbook.add_format({'bg_color': '#C6EFCE'})
        invalid_format = workbook.add_format({'bg_color': '#FFC7CE'})
        
        # Set column widths
        worksheet.set_column('A:A', 20)  # document_name
        worksheet.set_column('B:B', 15)  # field_name
        worksheet.set_column('C:C', 10)  # is_valid
        worksheet.set_column('D:D', 10)  # confidence
        worksheet.set_column('E:G', 30)  # errors, warnings, suggestions
        
        # Format header row
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Apply conditional formatting for validation status
        if 'is_valid' in df.columns:
            valid_col = df.columns.get_loc('is_valid')
            worksheet.conditional_format(1, valid_col, len(df), valid_col, {
                'type': 'cell',
                'criteria': '==',
                'value': True,
                'format': valid_format
            })
            worksheet.conditional_format(1, valid_col, len(df), valid_col, {
                'type': 'cell',
                'criteria': '==',
                'value': False,
                'format': invalid_format
            })
    
    def _format_table_sheet(self, workbook, worksheet, df):
        """Format individual table sheets."""
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#E2EFDA',
            'border': 1
        })
        
        # Auto-adjust column widths
        for i, column in enumerate(df.columns):
            # Get the maximum length in this column
            max_length = max(df[column].astype(str).str.len().max(), len(str(column)))
            # Set the column width (with some padding)
            worksheet.set_column(i, i, min(max_length + 2, 50))
        
        # Format header row
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
    
    def create_template(self, field_types: List[str], output_path: str = None) -> bytes:
        """
        Create an Excel template for manual data entry.
        
        Args:
            field_types: List of field types to include in template
            output_path: Path to save template file (optional)
            
        Returns:
            Template Excel file as bytes
        """
        try:
            output = BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Create template data
                template_data = []
                for i in range(10):  # Create 10 empty rows
                    row = {'document_name': f'Document_{i+1}'}
                    for field_type in field_types:
                        row[field_type] = ''
                    template_data.append(row)
                
                df_template = pd.DataFrame(template_data)
                df_template.to_excel(writer, sheet_name='Data_Entry_Template', index=False)
                
                # Format template
                workbook = writer.book
                worksheet = writer.sheets['Data_Entry_Template']
                
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#4F81BD',
                    'font_color': 'white',
                    'border': 1
                })
                
                # Set column widths and format headers
                for col_num, value in enumerate(df_template.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    worksheet.set_column(col_num, col_num, 20)
            
            excel_data = output.getvalue()
            
            if output_path:
                with open(output_path, 'wb') as f:
                    f.write(excel_data)
                logger.info(f"Template saved to {output_path}")
            
            return excel_data
            
        except Exception as e:
            logger.error(f"Error creating template: {e}")
            raise