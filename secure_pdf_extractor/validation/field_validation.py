"""
Field validation module for extracted data quality checks.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of field validation."""
    is_valid: bool
    confidence: float
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

class FieldValidator:
    """Validate extracted field data for quality and correctness."""
    
    def __init__(self):
        """Initialize field validator with validation rules."""
        self.validation_rules = {
            'date': self._validate_date,
            'angebot_number': self._validate_angebot_number,
            'company_name': self._validate_company_name,
            'amount': self._validate_amount,
            'delivery_date': self._validate_date,
            'contact_person': self._validate_contact_person,
            'email': self._validate_email,
            'phone': self._validate_phone,
            'postal_code': self._validate_postal_code,
            'tax_id': self._validate_tax_id
        }
        
        # Common validation patterns
        self.patterns = {
            'date': [
                r'^\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}$',
                r'^\d{2,4}[./\-]\d{1,2}[./\-]\d{1,2}$'
            ],
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^[\+]?[1-9][\d\s\-\(\)]{7,15}$',
            'postal_code': r'^\d{4,5}$',
            'amount': r'^\d+(?:[.,]\d{1,2})?$'
        }
    
    def validate_field(self, field_type: str, value: str, context: str = "") -> ValidationResult:
        """
        Validate a single field value.
        
        Args:
            field_type: Type of field to validate
            value: Field value to validate
            context: Context information for validation
            
        Returns:
            ValidationResult with validation details
        """
        if not value or not value.strip():
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                errors=["Empty value"],
                warnings=[],
                suggestions=["Provide a valid value"]
            )
        
        value = value.strip()
        
        # Get validation function for field type
        validator_func = self.validation_rules.get(field_type, self._validate_generic)
        
        return validator_func(value, context)
    
    def validate_fields(self, fields: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """
        Validate multiple fields.
        
        Args:
            fields: Dictionary of field_name -> field_data
            
        Returns:
            Dictionary of field_name -> ValidationResult
        """
        results = {}
        
        for field_name, field_data in fields.items():
            if isinstance(field_data, dict):
                value = field_data.get('value', '')
                context = field_data.get('context', '')
            else:
                value = str(field_data)
                context = ''
            
            results[field_name] = self.validate_field(field_name, value, context)
        
        return results
    
    def _validate_date(self, value: str, context: str = "") -> ValidationResult:
        """Validate date field."""
        errors = []
        warnings = []
        suggestions = []
        confidence = 0.0
        
        # Check format
        date_patterns = self.patterns['date']
        format_valid = any(re.match(pattern, value) for pattern in date_patterns)
        
        if not format_valid:
            errors.append("Invalid date format")
            suggestions.append("Use format DD.MM.YYYY or DD/MM/YYYY")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        confidence += 0.5
        
        # Try to parse the date
        try:
            # Handle different separators
            if '.' in value:
                parts = value.split('.')
            elif '/' in value:
                parts = value.split('/')
            elif '-' in value:
                parts = value.split('-')
            else:
                parts = []
            
            if len(parts) == 3:
                # Determine if it's DD.MM.YYYY or YYYY.MM.DD
                if len(parts[0]) == 4:  # YYYY.MM.DD
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                else:  # DD.MM.YYYY
                    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                
                # Convert 2-digit years
                if year < 100:
                    if year < 50:
                        year += 2000
                    else:
                        year += 1900
                
                # Validate ranges
                if not (1 <= month <= 12):
                    errors.append(f"Invalid month: {month}")
                elif not (1 <= day <= 31):
                    errors.append(f"Invalid day: {day}")
                elif year < 1900 or year > 2100:
                    warnings.append(f"Unusual year: {year}")
                else:
                    # Try to create actual date
                    try:
                        datetime(year, month, day)
                        confidence = 0.9
                        
                        # Check if date is in reasonable range for business documents
                        current_year = datetime.now().year
                        if year < current_year - 10:
                            warnings.append("Date is more than 10 years in the past")
                        elif year > current_year + 5:
                            warnings.append("Date is more than 5 years in the future")
                        
                    except ValueError as e:
                        errors.append(f"Invalid date: {e}")
            
        except ValueError:
            errors.append("Could not parse date components")
        
        is_valid = len(errors) == 0
        if not is_valid:
            confidence = 0.0
        
        return ValidationResult(is_valid, confidence, errors, warnings, suggestions)
    
    def _validate_angebot_number(self, value: str, context: str = "") -> ValidationResult:
        """Validate angebot/quote number."""
        errors = []
        warnings = []
        suggestions = []
        confidence = 0.0
        
        # Check length
        if len(value) < 3:
            errors.append("Angebot number too short (minimum 3 characters)")
            suggestions.append("Ensure complete number is extracted")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        confidence += 0.3
        
        # Check for valid characters
        if not re.match(r'^[A-Z0-9\-_/\.#]+$', value, re.IGNORECASE):
            errors.append("Contains invalid characters")
            suggestions.append("Should contain only letters, numbers, and common separators")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        confidence += 0.3
        
        # Check for typical patterns
        patterns = [
            r'^[A-Z]{2,4}\d{4,}$',  # Letters followed by numbers
            r'^\d{4,}$',            # All numbers
            r'^[A-Z]\d+$',          # Single letter + numbers
            r'^\d+-\d+$'            # Numbers with dash
        ]
        
        if any(re.match(pattern, value, re.IGNORECASE) for pattern in patterns):
            confidence += 0.4
        else:
            warnings.append("Unusual format for angebot number")
        
        # Check context for validation
        if context and any(keyword in context.lower() for keyword in ['angebot', 'quote', 'offer', 'nr']):
            confidence = min(1.0, confidence + 0.2)
        
        return ValidationResult(True, confidence, errors, warnings, suggestions)
    
    def _validate_company_name(self, value: str, context: str = "") -> ValidationResult:
        """Validate company name."""
        errors = []
        warnings = []
        suggestions = []
        confidence = 0.0
        
        # Check length
        if len(value) < 2:
            errors.append("Company name too short")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        confidence += 0.3
        
        # Check for at least one uppercase letter
        if not any(c.isupper() for c in value):
            warnings.append("No uppercase letters found")
        else:
            confidence += 0.2
        
        # Check for company indicators
        company_indicators = ['gmbh', 'ag', 'ltd', 'inc', 'corp', 'sa', 'bv', 'oy', 'ab']
        if any(indicator in value.lower() for indicator in company_indicators):
            confidence += 0.3
        
        # Check for numbers (unusual in company names)
        if re.search(r'\d', value):
            warnings.append("Contains numbers (unusual for company names)")
        
        # Check for special characters
        if re.search(r'[<>{}[\]|\\]', value):
            warnings.append("Contains unusual special characters")
        
        return ValidationResult(True, confidence, errors, warnings, suggestions)
    
    def _validate_amount(self, value: str, context: str = "") -> ValidationResult:
        """Validate monetary amount."""
        errors = []
        warnings = []
        suggestions = []
        confidence = 0.0
        
        # Remove currency symbols and spaces
        clean_value = re.sub(r'[€$£\s]', '', value)
        
        # Check basic pattern
        if not re.match(self.patterns['amount'], clean_value):
            errors.append("Invalid amount format")
            suggestions.append("Use format: 123.45 or 123,45")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        confidence += 0.5
        
        try:
            # Convert to float for validation
            numeric_value = float(clean_value.replace(',', '.'))
            
            # Check for reasonable ranges
            if numeric_value < 0:
                errors.append("Negative amount")
            elif numeric_value == 0:
                warnings.append("Zero amount")
            elif numeric_value > 1000000:
                warnings.append("Very large amount")
            else:
                confidence += 0.4
            
            # Check decimal places
            if '.' in clean_value or ',' in clean_value:
                decimal_part = clean_value.split('.' if '.' in clean_value else ',')[1]
                if len(decimal_part) > 2:
                    warnings.append("More than 2 decimal places")
                else:
                    confidence += 0.1
            
        except ValueError:
            errors.append("Could not parse numeric value")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, confidence, errors, warnings, suggestions)
    
    def _validate_contact_person(self, value: str, context: str = "") -> ValidationResult:
        """Validate contact person name."""
        errors = []
        warnings = []
        suggestions = []
        confidence = 0.0
        
        # Check for reasonable length
        if len(value) < 2:
            errors.append("Name too short")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        confidence += 0.3
        
        # Check for at least one space (first + last name)
        if ' ' not in value:
            warnings.append("Appears to be only first or last name")
        else:
            confidence += 0.3
        
        # Check for proper capitalization
        words = value.split()
        if all(word[0].isupper() for word in words if word):
            confidence += 0.2
        else:
            warnings.append("Unusual capitalization pattern")
        
        # Check for numbers (unusual in names)
        if re.search(r'\d', value):
            warnings.append("Contains numbers (unusual for person names)")
        
        # Check for titles
        titles = ['mr', 'mrs', 'ms', 'dr', 'prof', 'herr', 'frau']
        if any(title in value.lower().split() for title in titles):
            confidence += 0.2
        
        return ValidationResult(True, confidence, errors, warnings, suggestions)
    
    def _validate_email(self, value: str, context: str = "") -> ValidationResult:
        """Validate email address."""
        errors = []
        warnings = []
        suggestions = []
        
        if not re.match(self.patterns['email'], value):
            errors.append("Invalid email format")
            suggestions.append("Use format: name@domain.com")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        confidence = 0.8
        
        # Check for common domains
        common_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']
        domain = value.split('@')[1].lower()
        if domain in common_domains:
            warnings.append("Personal email domain")
        
        return ValidationResult(True, confidence, errors, warnings, suggestions)
    
    def _validate_phone(self, value: str, context: str = "") -> ValidationResult:
        """Validate phone number."""
        errors = []
        warnings = []
        suggestions = []
        
        if not re.match(self.patterns['phone'], value):
            errors.append("Invalid phone format")
            suggestions.append("Use format: +49 123 456789")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        return ValidationResult(True, 0.7, errors, warnings, suggestions)
    
    def _validate_postal_code(self, value: str, context: str = "") -> ValidationResult:
        """Validate postal code."""
        errors = []
        warnings = []
        suggestions = []
        
        if not re.match(self.patterns['postal_code'], value):
            errors.append("Invalid postal code format")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        return ValidationResult(True, 0.8, errors, warnings, suggestions)
    
    def _validate_tax_id(self, value: str, context: str = "") -> ValidationResult:
        """Validate tax ID."""
        errors = []
        warnings = []
        suggestions = []
        
        # Basic validation for tax ID format
        if len(value) < 5:
            errors.append("Tax ID too short")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        return ValidationResult(True, 0.6, errors, warnings, suggestions)
    
    def _validate_generic(self, value: str, context: str = "") -> ValidationResult:
        """Generic validation for unknown field types."""
        errors = []
        warnings = []
        suggestions = []
        
        # Basic checks
        if len(value.strip()) == 0:
            errors.append("Empty value")
            return ValidationResult(False, 0.0, errors, warnings, suggestions)
        
        confidence = 0.5  # Neutral confidence for unknown types
        
        return ValidationResult(True, confidence, errors, warnings, suggestions)
    
    def get_validation_summary(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """
        Get summary of validation results.
        
        Args:
            validation_results: Dictionary of field -> ValidationResult
            
        Returns:
            Summary statistics
        """
        total_fields = len(validation_results)
        valid_fields = sum(1 for result in validation_results.values() if result.is_valid)
        total_errors = sum(len(result.errors) for result in validation_results.values())
        total_warnings = sum(len(result.warnings) for result in validation_results.values())
        
        average_confidence = (
            sum(result.confidence for result in validation_results.values()) / total_fields
            if total_fields > 0 else 0.0
        )
        
        # Categorize fields by confidence
        high_confidence = sum(1 for result in validation_results.values() if result.confidence >= 0.8)
        medium_confidence = sum(1 for result in validation_results.values() if 0.5 <= result.confidence < 0.8)
        low_confidence = sum(1 for result in validation_results.values() if result.confidence < 0.5)
        
        return {
            'total_fields': total_fields,
            'valid_fields': valid_fields,
            'invalid_fields': total_fields - valid_fields,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'average_confidence': average_confidence,
            'validation_rate': valid_fields / total_fields if total_fields > 0 else 0.0,
            'confidence_distribution': {
                'high': high_confidence,
                'medium': medium_confidence,
                'low': low_confidence
            }
        }